# Data & Feature Manifest

This document defines the standardized feature names, their sources, and the processing pipeline used across the AI engine.

## Lag Intervals
Certain features include historical "lags" or "deltas" to capture trends. These are indicated by the following suffixes:
- `_1W`: 1-Week change/value
- `_1M`: 1-Month change/value
- `_3M`: 3-Month change/value
- `_6M`: 6-Month change/value
- `_1Y`: 1-Year change/value

---

## 1. Feature Definitions

### Categorical Features (User/Dashboard Inputs)
- `food_item`: Specific crop (e.g., "maize", "rice").
- `item_type`: Sub-type (e.g., "white", "local").
- `category`: Broad group (e.g., "cereals", "vegetables").
- `vendor_type`: Level of trade ("retail", "wholesale").
- `state`: Geographic region (mapped via reverse geocoding).

### Market & Price Features
- `price` (plus lag): The base unit price of the commodity.
- `target_price_change_1m`: The prediction target (% change in price 1 month into the future).

### Macroeconomic Features
- `general_inflation_rate` (plus lag): Headline inflation index.
- `food_inflation_rate` (plus lag): Specific food inflation index.
- `exchange_rate` (plus lag): NGN/USD weighted average rate.
- `diesel_price` (plus lag): Fuel/transportation cost index.
- `crude_oil_price` (plus lag): Global oil market price.

### Environmental & Narrative
- `avg_temperature_c`: Average weekly temperature.
- `precipitation_mm`: Total weekly rainfall.
- `solar_radiation_mj`: Sunlight exposure/solar radiation.
- `weekly_econ_sentiment_score`: Derived from news headlines via FinBERT.

### Conflict & Stability
- `regional_events_count` (plus lag): Number of insecurity events in the state (ACLED).
- `regional_fatalities_count` (plus lag): Total fatalities from insecurity events.

### Temporal & Contextual
- `year`: Calendar year.
- `month`: Calendar month.
- `week`: ISO week number (%W).
- `month_num`: 1–12 numeric representation.
- `seasonality_month`: Sine/Cosine or categorical indicator of harvest/seasonal cycles.

---

## 2. Data Sources & Pipeline

| Feature Group | Current Source | Raw Frequency | Additional Processing |
| :--- | :--- | :--- | :--- |
| **Market Prices** | Market Surveys (Excel) | Daily | Reverse Geocoding (Lat/Lon to State) |
| **News Sentiment** | Web/News API (Excel) | Daily | **FinBERT** Sentiment Classification |
| **Insecurity** | ACLED (Excel) | Monthly | Sum aggregation for events/fatalities |
| **Weather** | NASA POWER (API) | Daily | Mean/Sum aggregation by State |
| **Inflation** | National Bureau of Stats (Excel) | Monthly | Mean interpolation to weekly |
| **Diesel Price** | National Bureau of Stats (Excel) | Monthly | Mean interpolation by State |
| **Crude Oil** | Global Markets (Excel) | Daily | Mean aggregation |
| **Exchange Rate** | Central Bank (Excel) | Daily | Weighted Average Mean aggregation |

---

## 3. Standard Processing Steps
1. **Geospatial Mapping:** All point-data (Lat/Lon) is converted to State names using the `reverse_geocoder` library.
2. **Temporal Resampling:** Monthly data is converted to Weekly (`%W`) granularity using a day-weighted proportional distribution logic.
3. **Sentiment Scoring:** Raw news text is processed through the `ProsusAI/finbert` model. Scores are calculated as `(positive_confidence - negative_confidence)`.
