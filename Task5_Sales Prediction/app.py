import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# PAGE CONFIG
st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# LOAD MODEL
def load_model():
    with open("sales_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# TITLE
st.title("📈 Sales Prediction Dashboard")
st.markdown(
    "Predict product sales based on advertising expenditure using a Random Forest Regressor."
)

# SIDEBAR
st.sidebar.header("Advertising Budget Inputs")

tv = st.sidebar.slider(
    "📺 TV Advertising Budget",
    min_value=0.0,
    max_value=300.0,
    value=100.0
)

radio = st.sidebar.slider(
    "📻 Radio Advertising Budget",
    min_value=0.0,
    max_value=50.0,
    value=20.0
)

newspaper = st.sidebar.slider(
    "📰 Newspaper Advertising Budget",
    min_value=0.0,
    max_value=120.0,
    value=30.0
)

predict_btn = st.sidebar.button("Predict Sales")
st.sidebar.markdown("---")

# MODEL PERFORMANCE
st.markdown("---")
st.subheader("📈 Model Performance")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("R² Score", "0.981")

with col2:
    st.metric("MAE", "0.620")

with col3:
    st.metric("RMSE", "0.768")

# PREDICTION SECTION
if predict_btn:

    input_data = [[tv, radio, newspaper]]
    prediction = model.predict(input_data)
    sales = prediction[0]

    st.markdown("---")
    st.subheader("📊 Prediction Result")
    st.metric(
        label="Predicted Sales",
        value=f"{sales:.2f} Thousand Units"
    )

    st.markdown("---")
    st.subheader("Input Summary")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("TV Budget", tv)

    with c2:
        st.metric("Radio Budget", radio)

    with c3:
        st.metric("Newspaper Budget", newspaper)

    st.markdown("---")
    st.subheader("📈 Business Insight")

    if sales > 20:
        st.success(
            "The current advertising strategy is highly effective and is expected to generate strong sales."
        )

    elif sales > 10:
        st.info(
            "The advertising strategy is expected to generate moderate sales."
        )

    else:
        st.warning(
            "Sales are expected to be relatively low."
        )

# FEATURE IMPORTANCE
st.markdown("---")
st.subheader("🎯 Feature Importance")
importance_df = pd.read_csv("feature_importance.csv")
fig, ax = plt.subplots(figsize=(4, 2.5))
ax.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)
ax.set_title("Feature Importance")
ax.set_xlabel("Features")
ax.set_ylabel("Importance")
plt.tight_layout()
st.pyplot(fig, use_container_width=False)
plt.close(fig)   

# MODEL INFORMATION
st.markdown("---")
st.subheader("Model Information")

st.write("""
### Selected Model
**Random Forest Regressor**

### Why was this model selected?
- Achieved the best predictive performance.
- Captures both linear and non-linear relationships effectively.
- Produces reliable sales predictions.

### Features Used
- TV Advertising Budget
- Radio Advertising Budget
- Newspaper Advertising Budget

### Target Variable
- Sales

### Objective
Predict future sales based on advertising expenditure across different media channels.
""")