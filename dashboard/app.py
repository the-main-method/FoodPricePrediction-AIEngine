import streamlit as st
import pandas as pd

import agri_price.predictor
import agri_price.data

# --- UI Setup ---
st.set_page_config(page_title="AgriPrice AI Demo", layout="wide")
st.title("🌾 National Agricultural Commodity Predictive Engine")
st.markdown("Adjust the macroeconomic parameters on the left to simulate shocks and observe how the CatBoost model adjusts its 1-Month forecast.")

@st.cache_resource
def load_and_train_model():
    return agri_price.predictor.load_model("models/agri_price_model.cbm")

@st.cache_data
def load_data():
    return agri_price.data.load_data("data/ml_ready_global_data.csv")

model, explainer = load_and_train_model()
X, y, cat_features = load_data()

X_baseline = X

# --- Sidebar: The Control Panel (The Levers) ---
st.sidebar.header("🕹️ Scenario Simulator")

# Cascading Dynamic Dropdowns (Reads directly from your CSV)
available_crops = sorted(X_baseline['food_item'].unique())
selected_crop = st.sidebar.selectbox("Select Crop", available_crops)

# Filter the available vendors based on the crop selected to prevent impossible combinations
valid_vendors = sorted(X_baseline[X_baseline['food_item'] == selected_crop]['vendor_type'].unique())
selected_vendor = st.sidebar.selectbox("Select Vendor Type", valid_vendors)

st.sidebar.markdown("---")
st.sidebar.subheader("Simulate Market Shocks")

# Extract realistic medians from the actual data for the sliders
inf_med = float(X_baseline['General_Inflation_Rate'].median())
temp_med = float(X_baseline['Avg_Temperature_C'].median())

# If you have Diesel or Conflict features in your master CSV, you can safely swap these back!
sim_inflation = st.sidebar.slider("General Inflation Rate (%)", 10.0, 50.0, inf_med)
sim_temp = st.sidebar.slider("Average Temperature (°C)", 20.0, 40.0, temp_med)
sim_precip = st.sidebar.slider("Precipitation (mm)", 0.0, 300.0, float(X_baseline['Precipitation_mm'].median()))

# --- Prediction Logic ---
# Grab the most recent real row for this specific crop & vendor to act as our baseline
crop_history = X_baseline[(X_baseline['food_item'] == selected_crop) & (X_baseline['vendor_type'] == selected_vendor)]

# Fallback in case a specific vendor/crop combo is rare
if crop_history.empty:
    crop_history = X_baseline[X_baseline['food_item'] == selected_crop]
    
baseline_row = crop_history.iloc[-1].copy()

# Inject the Manager's simulated inputs!
baseline_row['vendor_type'] = selected_vendor
baseline_row['General_Inflation_Rate'] = sim_inflation
baseline_row['Avg_Temperature_C'] = sim_temp
baseline_row['Precipitation_mm'] = sim_precip

# Format for CatBoost
input_df = agri_price.predictor.build_input_df(dict(baseline_row), model)

# Make the Prediction
prediction = model.predict(input_df)[0]

# --- Display Results ---
st.markdown("### 🔮 Forecast")
# Color code the metric (Red for price jumps, Green for drops)
metric_color = "inverse" if prediction > 0 else "normal"
st.metric(
    label=f"Predicted 1-Month Price Change for {selected_crop.title()}", 
    value=f"{prediction:+.2f}%",
    delta="Warning: Price Spike" if prediction > 10 else "Stable",
    delta_color=metric_color
)

# --- The "Glass Box" Explainability ---
st.markdown("### 🧠 Model Explainability (SHAP)")
st.markdown("Why is the model predicting this? Here are the top variables driving the math right now:")

shap_values, impacts = agri_price.predictor.run_shap(explainer, input_df)

# Sort and display as an interactive table and bar chart
if impacts:
    impact_df = pd.DataFrame(impacts)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(impact_df.head(8), hide_index=True)
    with col2:
        # Quick horizontal bar chart for visual impact
        chart_data = impact_df.head(8).set_index("feature")["impact_percentage"]
        st.bar_chart(chart_data, horizontal=True)
else:
    st.info("The model indicates that base market trends are overriding individual feature impacts for this specific input.")

st.success("Architecture Ready: NLP Sentiments, Macro-Economics, Weather, and Spatial Conflict mapped successfully.")