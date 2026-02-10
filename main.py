import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, mean_squared_error, r2_score,
    roc_curve 
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

# --- 1. LOAD DATA ---
orders = pd.read_csv("olist_orders_dataset.csv")
reviews = pd.read_csv("olist_order_reviews_dataset.csv")
items = pd.read_csv("olist_order_items_dataset.csv")

df = orders.merge(reviews, on="order_id").merge(items, on="order_id")

# --- 2. FEATURE ENGINEERING ---
df["is_bad"] = (df["review_score"] < 3).astype(int)

df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"])
df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"])

df["actual_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
df["late"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days
df["late"] = df["late"].apply(lambda x: x if x > 0 else 0)

df = df.dropna(subset=["actual_days", "price", "freight_value"])

X = df[["actual_days", "late", "price", "freight_value"]]
y = df["is_bad"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 3. MODEL PIPELINES ---
pipes = {
    "Logistic Regression": Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Random Forest": Pipeline([
        ("scale", StandardScaler()),
        ("model", RandomForestClassifier())
    ]),
    "Gradient Boosting": Pipeline([
        ("scale", StandardScaler()),
        ("model", GradientBoostingClassifier())
    ])
}

params = {
    "Logistic Regression": {
        "model__C": [0.1, 1, 10],
        "model__penalty": ["l2"]
    },
    "Random Forest": {
        "model__n_estimators": [100, 150],
        "model__max_depth": [4, 6, 8, None]
    },
    "Gradient Boosting": {
        "model__n_estimators": [100, 150],
        "model__learning_rate": [0.05, 0.1]
    }
}

cv = KFold(3, shuffle=True, random_state=42)
final_models = {}

# --- 4. TRAINING & TUNING ---
print("Starting Training...")
for name in pipes:
    print(f"Tuning {name}...")
    search = GridSearchCV(
        pipes[name],
        params[name],
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    final_models[name] = search.best_estimator_

# --- 5. EVALUATION ---
results = []

for name, model in final_models.items():
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    roc = roc_auc_score(y_test, prob)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results.append([name, acc, roc, rmse, r2])

    print("\n==============================")
    print("MODEL:", name)
    print("==============================\n")
    print("Accuracy:", round(acc, 4))
    print("ROC-AUC:", round(roc, 4))
    print("RMSE:", round(rmse, 4))
    print("R² Score:", round(r2, 4))
    print("\nClassification Report Details:\n")
    print(classification_report(y_test, pred))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, pred))

# --- 6. LEADERBOARD ---
res = pd.DataFrame(results, columns=["Model", "Accuracy", "ROC-AUC", "RMSE", "R2"])
print("\nFINAL LEADERBOARD\n")
print(res.sort_values("ROC-AUC", ascending=False))

# --- 7. ROC CURVE PLOT  ---
plt.figure(figsize=(7, 5))
for name, mdl in final_models.items():
    prob = mdl.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=name)

plt.plot([0, 1], [0, 1], '--', color="gray", label="Random Guess")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

# --- 8. SAVE BEST MODEL ---
winner = res.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]
best_model = final_models[winner]
joblib.dump(best_model, "best_model.pkl")

# Save SHAP background set
X_train.sample(200).to_csv("x_background.csv", index=False)

print("\nSaved:", winner, "as best_model.pkl")
print("Saved ROC Curve as 'roc_curve_comparison.png'")
