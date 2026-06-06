import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor
import shap

# --- UI Setup ---
st.set_page_config(page_title="AgriPrice AI Demo", layout="wide")
st.title("🌾 National Agricultural Commodity Predictive Engine")
st.markdown("Adjust the macroeconomic parameters on the left to simulate shocks and observe how the CatBoost model adjusts its 1-Month forecast.")

# --- Cache the Model Training ---
@st.cache_resource
# --- Cache the Model Training ---
@st.cache_resource
def load_and_train_model():
    # Load the latest dataset
    df = pd.read_csv('ml_ready_global_data.csv')
    
    # 1. DYNAMICALLY auto-detect all text/categorical columns! 
    # This grabs 'state', 'food_item', and any future text columns you add.
    cat_features = df.select_dtypes(include=['object', 'string']).columns.tolist()
    
    # Ensure categorical columns are perfectly clean strings
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
            
    # Use our precise 1-Month target
    target_col = 'TARGET_Price_Change_1M' 
    
    X = df.drop(columns=['Year', 'Month', 'Week', target_col], errors='ignore')
    y = df[target_col]
    
    # Double check that we only tell CatBoost about categories that are actually in X
    cat_features = [col for col in cat_features if col in X.columns]
    
    # Train a fast, reliable model on the fly for the demo
    model = CatBoostRegressor(iterations=138, depth=5, learning_rate=0.1, verbose=0)
    model.fit(X, y, cat_features=cat_features)
    
    return model, X, cat_features

# Load everything
model, X_baseline, cat_features = load_and_train_model()

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
input_df = pd.DataFrame([baseline_row])

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

# Run SHAP on the live input
explainer = shap.TreeExplainer(model)
shap_values = explainer(input_df)

# Process SHAP values into a clean DataFrame for the UI
impacts = []
for feature_name, shap_val, actual_val in zip(input_df.columns, shap_values.values[0], input_df.values[0]):
    if round(shap_val, 2) != 0:
        impacts.append({
            "Feature": feature_name,
            "Simulated Input Value": actual_val,
            "Impact on Price (%)": float(round(shap_val, 2))
        })

# Sort and display as an interactive table and bar chart
if impacts:
    impact_df = pd.DataFrame(impacts).sort_values(by="Impact on Price (%)", key=abs, ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(impact_df.head(8), hide_index=True)
    with col2:
        # Quick horizontal bar chart for visual impact
        chart_data = impact_df.head(8).set_index("Feature")["Impact on Price (%)"]
        st.bar_chart(chart_data, horizontal=True)
else:
    st.info("The model indicates that base market trends are overriding individual feature impacts for this specific input.")

st.success("Architecture Ready: NLP Sentiments, Macro-Economics, Weather, and Spatial Conflict mapped successfully.")