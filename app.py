# =========================
# app.py
# =========================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time

import plotly.express as px
import plotly.graph_objects as go

import seaborn as sns
import matplotlib.pyplot as plt

# PAGE CONFIG
st.set_page_config(
    page_title="AI Revenue Dashboard",
    page_icon="🚀",
    layout="wide"
)

# LOAD MODEL
model = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/scaler.pkl")
results_df = joblib.load("model/model_results.pkl")

# CSS
st.markdown("""
<style>

.stApp {
    background: linear-gradient(-45deg, #0f172a, #111827, #312e81, #0f766e);
    background-size: 400% 400%;
    animation: gradientMove 15s ease infinite;
    color: white;
}

@keyframes gradientMove {
    0% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
    100% {
        background-position: 0% 50%;
    }
}

.main-title {
    text-align: center;
    font-size: 65px;
    font-weight: bold;
    color: white;
    margin-top: 20px;
    text-shadow:
        0 0 10px #00e5ff,
        0 0 20px #00e5ff,
        0 0 40px #00e5ff;
}

.subtitle {
    text-align: center;
    font-size: 22px;
    color: #cbd5e1;
    margin-bottom: 40px;
}

.kpi-card {
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.prediction-box {
    background: linear-gradient(135deg, #00c853, #00e676);
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: white;
    box-shadow: 0 10px 40px rgba(0,255,100,0.4);
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white;
    font-size: 22px;
    border-radius: 12px;
    padding: 14px;
    border: none;
}

section[data-testid="stSidebar"] {
    background: #111827;
}

</style>
""", unsafe_allow_html=True)

# TITLE
st.markdown(
    '<div class="main-title">🚀 AI Revenue Prediction Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Advanced Marketing Intelligence Platform</div>',
    unsafe_allow_html=True
)

# TABS
analytics_tab, prediction_tab = st.tabs([
    "📊 Data Analytics",
    "🤖 Revenue Prediction"
])

# =========================
# ANALYTICS TAB
# =========================

with analytics_tab:

    st.header("📊 Dataset Analytics")

    data = pd.read_csv("data/train.csv")

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    st.subheader("Dataset Shape")
    st.write(data.shape)

    st.subheader("Missing Values")
    st.write(data.isnull().sum())

    st.subheader("Statistical Summary")
    st.write(data.describe())

    st.subheader("Correlation Heatmap")

    numeric_data = data.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(12,8))

    sns.heatmap(
        numeric_data.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("📈 Model Comparison")

    st.dataframe(results_df)

    comparison_fig = px.bar(
        results_df,
        x="Model",
        y="R2 Score",
        color="Model",
        title="Model Performance Comparison"
    )

    st.plotly_chart(comparison_fig, use_container_width=True)

# =========================
# PREDICTION TAB
# =========================

with prediction_tab:

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info("📊 Smart Analytics")

    with c2:
        st.success("⚡ Real-Time AI")

    with c3:
        st.warning("💰 Revenue Intelligence")

    st.write("")

    # SIDEBAR
    st.sidebar.title("⚙ Campaign Settings")

    region = st.sidebar.selectbox(
        "Region",
        ["North", "South", "East", "West"]
    )

    channel = st.sidebar.selectbox(
        "Marketing Channel",
        ["Email", "Social Media", "TV", "Google Ads"]
    )

    product_category = st.sidebar.selectbox(
        "Product Category",
        ["Electronics", "Fashion", "Home", "Beauty"]
    )

    customer_segment = st.sidebar.selectbox(
        "Customer Segment",
        ["Low", "Medium", "High"]
    )

    st.markdown("## 📈 Campaign Metrics")

    col1, col2 = st.columns(2)

    with col1:
        ad_spend = st.number_input("Ad Spend", min_value=0.0)
        price = st.number_input("Price", min_value=0.0)
        discount_rate = st.number_input("Discount Rate", min_value=0.0)
        market_reach = st.number_input("Market Reach", min_value=0.0)
        impressions = st.number_input("Impressions", min_value=0.0)

    with col2:
        click_through_rate = st.number_input("Click Through Rate", min_value=0.0)
        competition_index = st.number_input("Competition Index", min_value=0.0)
        seasonality_index = st.number_input("Seasonality Index", min_value=0.0)
        campaign_duration_days = st.number_input("Campaign Duration Days", min_value=0.0)
        customer_lifetime_value = st.number_input("Customer Lifetime Value", min_value=0.0)

    # ENCODING
    region_map = {
        "North": 0,
        "South": 1,
        "East": 2,
        "West": 3
    }

    channel_map = {
        "Email": 0,
        "Social Media": 1,
        "TV": 2,
        "Google Ads": 3
    }

    product_map = {
        "Electronics": 0,
        "Fashion": 1,
        "Home": 2,
        "Beauty": 3
    }

    segment_map = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    # PREDICT
    if st.button("🚀 Predict Revenue"):

        with st.spinner("AI Engine Processing Data..."):

            progress_bar = st.progress(0)

            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)

        input_data = np.array([[
            region_map[region],
            channel_map[channel],
            product_map[product_category],
            segment_map[customer_segment],
            ad_spend,
            price,
            discount_rate,
            market_reach,
            impressions,
            click_through_rate,
            competition_index,
            seasonality_index,
            campaign_duration_days,
            customer_lifetime_value
        ]])

        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)

        st.markdown(f"""
        <div class="prediction-box">
            💰 Predicted Revenue <br><br>
            ₹ {prediction[0]:,.2f}
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        # AI INSIGHTS
        st.markdown("## 🤖 AI Insights")

        if click_through_rate < 2:
            st.warning("⚠ Low CTR detected. Improve ad creatives and targeting.")

        elif click_through_rate < 5:
            st.info("ℹ Average CTR detected. Campaign performing moderately.")

        else:
            st.success("✅ Excellent CTR detected. Campaign engagement is strong.")

        if competition_index > 70:
            st.error("🔥 High market competition detected.")

        if customer_lifetime_value > 5000:
            st.success("💎 High-value customers identified.")

        st.write("")

        # KPI DASHBOARD
        k1, k2, k3 = st.columns(3)

        with k1:
            st.metric("📈 Market Reach", f"{market_reach:,.0f}")

        with k2:
            st.metric("🎯 CTR", f"{click_through_rate}%")

        with k3:
            st.metric("💰 Customer Value", f"₹ {customer_lifetime_value:,.0f}")

        st.write("")

        # BAR CHART
        chart_data = pd.DataFrame({
            "Metric": [
                "Ad Spend",
                "Reach",
                "CTR",
                "Customer Value"
            ],
            "Value": [
                ad_spend,
                market_reach,
                click_through_rate,
                customer_lifetime_value
            ]
        })

        fig = px.bar(
            chart_data,
            x="Metric",
            y="Value",
            title="📊 Campaign Performance Analytics"
        )

        st.plotly_chart(fig, use_container_width=True)

        # PIE CHART
        pie_fig = px.pie(
            chart_data,
            names="Metric",
            values="Value",
            title="📌 Campaign Distribution"
        )

        st.plotly_chart(pie_fig, use_container_width=True)

        # GAUGE CHART
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction[0],
            title={'text': "Revenue Score"},
            gauge={
                'axis': {'range': [0, max(prediction[0]*1.5, 1000)]},
                'bar': {'color': "green"}
            }
        ))

        st.plotly_chart(gauge, use_container_width=True)

        st.success("✅ AI Analysis Completed Successfully!")