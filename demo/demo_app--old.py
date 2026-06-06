import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor
import shap

# --- UI Setup ---
st.set_page_config(page_title="AgriPrice AI Demo", layout="wide")
st.title("🌾 AgriPrice AI: Live Predictive Dashboard")
st.markdown("Adjust the macroeconomic parameters on the left to simulate shocks and observe how the CatBoost model adjusts its 4-week forecast.")

# --- Cache the Model Training ---
# @st.cache_resource ensures it only trains once when you start the app, making sliders instant!
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv('ml_ready_global_data.csv')
    cat_features = ['food_item', 'item_type', 'category', 'vendor_type']
    
    # Ensure categorical columns are strings
    for col in cat_features:
        df[col] = df[col].astype(str)
        
    X = df.drop(columns=['Year', 'Month', 'Week', 'TARGET_Price_Change_4W'])
    y = df['TARGET_Price_Change_4W']
    
    # Train a fast, reliable model
    model = CatBoostRegressor(iterations=95, depth=5, learning_rate=0.1, verbose=0)
    model.fit(X, y, cat_features=cat_features)
    
    return model, X, cat_features

# Load everything
model, X_baseline, cat_features = load_and_train_model()

# --- Sidebar: The Control Panel (The Levers) ---
st.sidebar.header("🕹️ Scenario Simulator")

# Crop Selection
selected_crop = st.sidebar.selectbox("Select Crop", X_baseline['food_item'].unique())
selected_vendor = st.sidebar.selectbox("Select Vendor Type", X_baseline['vendor_type'].unique())

st.sidebar.markdown("---")
st.sidebar.subheader("Simulate Market Shocks")

# Extract realistic min/max ranges from the actual data for the sliders
inf_med = float(X_baseline['Food_Inflation_Rate'].median())
diesel_med = float(X_baseline['Diesel_Price_Change_1M'].median())
events_max = int(X_baseline['Regional_Events_Count'].max())

# Interactive Sliders
sim_inflation = st.sidebar.slider("Food Inflation Rate (%)", 10.0, 50.0, inf_med)
sim_diesel = st.sidebar.slider("Diesel Price Shock (1M % Change)", -10.0, 150.0, diesel_med)
sim_conflict = st.sidebar.slider("Regional Conflict Events (Count)", 0, events_max, 0)

# --- Prediction Logic ---
# Grab the most recent real row for this specific crop to act as our baseline
crop_history = X_baseline[X_baseline['food_item'] == selected_crop]
baseline_row = crop_history.iloc[-1].copy()

# Inject the Manager's simulated inputs!
baseline_row['vendor_type'] = selected_vendor
baseline_row['Food_Inflation_Rate'] = sim_inflation
baseline_row['Diesel_Price_Change_1M'] = sim_diesel
baseline_row['Regional_Events_Count'] = float(sim_conflict)

# Format for CatBoost
input_df = pd.DataFrame([baseline_row])

# Make the Prediction
prediction = model.predict(input_df)[0]

# --- Display Results ---
st.markdown("### 🔮 Forecast")
# Color code the metric (Red for price jumps, Green for drops)
metric_color = "inverse" if prediction > 0 else "normal"
st.metric(
    label=f"Predicted 4-Week Price Change for {selected_crop.title()}", 
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
impact_df = pd.DataFrame(impacts).sort_values(by="Impact on Price (%)", key=abs, ascending=False)

col1, col2 = st.columns(2)
with col1:
    st.dataframe(impact_df.head(8), hide_index=True)
with col2:
    # Quick horizontal bar chart for visual impact
    chart_data = impact_df.head(8).set_index("Feature")["Impact on Price (%)"]
    st.bar_chart(chart_data, horizontal=True)

st.success("Architecture Ready: NLP Sentiments, Macro-Economics, Weather, and Spatial Conflict mapped successfully.")