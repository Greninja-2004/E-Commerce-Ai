import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Anurag's AI",
    page_icon="🤖",
    layout="centered"
)

# --- THE "CYBERPUNK" STYLING ---
st.markdown("""
    <style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400&display=swap');

    /* BACKGROUND - Deep Space Dark Theme */
    .stApp {
        background: radial-gradient(circle at 50% 10%, #2b0a3d 0%, #0e1117 80%);
        color: #e0e0e0;
        font-family: 'Roboto', sans-serif;
    }

    /* HEADERS - Sci-Fi Font */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #00f2ff !important;
        text-shadow: 0 0 10px rgba(0, 242, 255, 0.5);
    }
    
    /* INPUT FIELDS - Neon Borders */
    .stNumberInput, .stDateInput {
        background-color: rgba(0, 0, 0, 0.3) !important;
        border: 1px solid #333;
        border-radius: 5px;
        color: white;
    }
    /* Glow effect on focus */
    .stNumberInput:focus-within, .stDateInput:focus-within {
        border-color: #00f2ff !important;
        box-shadow: 0 0 10px #00f2ff;
    }

    /* PREDICT BUTTON - The "Nuclear" Button */
    .stButton>button {
        background: linear-gradient(45deg, #ff0055, #ff00aa);
        color: white;
        font-family: 'Orbitron', sans-serif;
        font-size: 18px;
        border: none;
        border-radius: 50px;
        padding: 12px 30px;
        transition: all 0.4s ease;
        box-shadow: 0 0 15px rgba(255, 0, 85, 0.6);
        width: 100%;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 25px rgba(255, 0, 85, 1);
    }

    /* RESULT CARDS - Glassmorphism */
    div[data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        font-size: 30px;
        color: #fff;
    }
    div[data-testid="stMetricLabel"] {
        color: #aaa;
    }
    
    /* CUSTOM BANNERS */
    .happy-card {
        background: rgba(0, 255, 157, 0.1);
        border: 1px solid #00ff9d;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 20px rgba(0, 255, 157, 0.2);
    }
    .sad-card {
        background: rgba(255, 0, 85, 0.1);
        border: 1px solid #ff0055;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 20px rgba(255, 0, 85, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_model.pkl")
        return model
    except FileNotFoundError:
        st.error("⚠️ SYSTEM ERROR: Model file missing.")
        return None

model = load_model()

# --- HEADER SECTION ---
st.title("🚀 E-Commerce Satisfaction AI")
st.markdown("### `SYSTEM STATUS: ONLINE`")

st.write("Input logistics parameters to initialize prediction sequence.")

# --- INPUT DASHBOARD ---
with st.container():
    st.markdown("---")
    
    # Row 1: Finances
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 💳 TRANSACTION DATA")
        price = st.number_input("Unit Price (R$)", value=100.0, step=10.0)
    with col2:
        st.markdown("#### 📦 FREIGHT DATA")
        freight_value = st.number_input("Shipping Cost (R$)", value=20.0, step=5.0)

    # Row 2: Timeline
    st.markdown("#### ⏳ TEMPORAL DATA")
    c1, c2, c3 = st.columns(3)
    with c1:
        purchase_date = st.date_input("Purchase Date")
    with c2:
        estimated_date = st.date_input("Est. Delivery")
    with c3:
        delivered_date = st.date_input("Actual Delivery")

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.button("INITIATE ANALYSIS SEQUENCE")

# --- EXECUTION LOGIC ---
if submitted and model is not None:
    # Calculations
    purchase_ts = pd.to_datetime(purchase_date)
    estimated_ts = pd.to_datetime(estimated_date)
    delivered_ts = pd.to_datetime(delivered_date)

    actual_days = (delivered_ts - purchase_ts).days
    late_days_raw = (delivered_ts - estimated_ts).days
    late = max(0, late_days_raw)

    if actual_days < 0:
        st.error("⚠️ CRITICAL ERROR: Temporal Paradox Detected (Delivery < Purchase)")
    else:
        # Prepare Input
        input_data = pd.DataFrame({
            'actual_days': [actual_days],
            'late': [late],
            'price': [price],
            'freight_value': [freight_value]
        })

        # Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # --- RESULTS DISPLAY ---
        st.markdown("---")
        st.markdown("### 📡 ANALYSIS COMPLETE")
        
        # Metrics Row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Cycle Time", f"{actual_days} Days")
        with m2:
            st.metric("Delay Factor", f"{late} Days")
        with m3:
            st.metric("Risk Probability", f"{probability:.1%}")

        st.markdown("<br>", unsafe_allow_html=True)

        # The Big Reveal
        if prediction == 1:
            st.markdown(f"""
                <div class="sad-card">
                    <h2 style="color: #ff0055 !important;">⛔ HIGH CHURN RISK</h2>
                    <p>Customer Sentiment: <b>NEGATIVE</b></p>
                    <p>Recommendation: Immediate Support Intervention</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="happy-card">
                    <h2 style="color: #00ff9d !important;">✅ LOW CHURN RISK</h2>
                    <p>Customer Sentiment: <b>POSITIVE</b></p>
                    <p>Recommendation: No Action Required</p>
                </div>
            """, unsafe_allow_html=True)

        # Feature Importance (Dark Mode Version)
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔍 VIEW NEURAL WEIGHTS"):
            classifier = model.named_steps['model']
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                feature_names = ['Time', 'Lateness', 'Price', 'Freight']
                
                # Dark Mode Plot
                fig, ax = plt.subplots(figsize=(6, 2))
                fig.patch.set_facecolor('#0e1117') # Dark background
                ax.set_facecolor('#0e1117')
                
                # Neon Bars
                sns.barplot(x=importances, y=feature_names, palette=['#00f2ff', '#00f2ff', '#ff0055', '#ff0055'], ax=ax)
                
                # White text for dark mode
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.set_title("Decision Factors", color='white')
                # Remove borders
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white') 
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                st.pyplot(fig)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #555; font-size: 15px; font-family: 'Orbitron', sans-serif;">
        SYSTEM ARCHITECT: <b>ANURAG KARMAKAR</b> | v2.0.45
    </div>
""", unsafe_allow_html=True)
