import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="centered"
)

st.markdown("""
<style>
    .stSelectbox label, .stSlider label { font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CARRIERS = [
    ("WN", "Southwest Airlines"),
    ("DL", "Delta Air Lines"),
    ("AA", "American Airlines"),
    ("UA", "United Airlines"),
    ("OO", "SkyWest Airlines"),
    ("MQ", "American Eagle Airlines"),
    ("US", "US Airways"),
    ("B6", "JetBlue Airways"),
    ("NK", "Spirit Airlines"),
    ("F9", "Frontier Airlines"),
]

AIRPORTS = [
    ("ATL", "Atlanta — Hartsfield-Jackson"),
    ("ORD", "Chicago — O'Hare"),
    ("DFW", "Dallas/Fort Worth"),
    ("DEN", "Denver International"),
    ("LAX", "Los Angeles International"),
    ("CLT", "Charlotte Douglas"),
    ("PHX", "Phoenix Sky Harbor"),
    ("IAH", "Houston — George Bush"),
    ("DTW", "Detroit Metro Wayne County"),
    ("EWR", "Newark Liberty"),
    ("SFO", "San Francisco International"),
    ("MSP", "Minneapolis-St Paul"),
    ("LAS", "Las Vegas — McCarran"),
    ("MCO", "Orlando International"),
    ("SEA", "Seattle/Tacoma"),
    ("LGA", "New York — LaGuardia"),
    ("SLC", "Salt Lake City"),
    ("BOS", "Boston — Logan"),
    ("BWI", "Baltimore/Washington"),
    ("JFK", "New York — JFK"),
]

MONTHS = [
    (1, "January"), (2, "February"), (3, "March"), (4, "April"),
    (5, "May"), (6, "June"), (7, "July"), (8, "August"),
    (9, "September"), (10, "October"), (11, "November"), (12, "December")
]

CATEGORICAL_FEATURES = ['carrier', 'airport', 'month']
NUMERIC_FEATURES     = ['year', 'arr_flights', 'cancel_rate', 'diversion_rate', 'delay_rate']
FEATURES             = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET               = 'avg_delay_minutes_per_flight'

# ── Model + data (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_data():
    df = pd.read_csv('Airline_Delay_Cause_cleaned_sample_60k.csv')
    upper = df[TARGET].quantile(0.999)
    df_clean = df[df[TARGET] <= upper].copy()

    X = df_clean[FEATURES]
    y = df_clean[TARGET]

    prep = ColumnTransformer([
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CATEGORICAL_FEATURES),
        ('num', 'passthrough', NUMERIC_FEATURES)
    ])
    model = Pipeline([
        ('prep', prep),
        ('model', RandomForestRegressor(
            n_estimators=300, max_depth=15, min_samples_leaf=5,
            max_features='sqrt', n_jobs=-1, random_state=42
        ))
    ])
    model.fit(X, y)

    # Precompute per carrier/airport/month averages for auto-filling numeric features
    historical_avg = df_clean.groupby(['carrier', 'airport', 'month'])[NUMERIC_FEATURES].mean()
    global_stats = df_clean[NUMERIC_FEATURES].mean()

    return model, df_clean, historical_avg, global_stats

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("✈️ Flight Delay Predictor")
st.caption("DSC 148 Final Project — predict average arrival delay in minutes")
st.divider()

model, df, historical_avg, global_stats = load_model_and_data()

col1, col2 = st.columns(2)

with col1:
    carrier_label = st.selectbox("Airline", options=[name for _, name in CARRIERS])
    carrier_code  = dict((name, code) for code, name in CARRIERS)[carrier_label]

    month_label = st.selectbox("Month", options=[name for _, name in MONTHS])
    month_num   = dict((name, num) for num, name in MONTHS)[month_label]

with col2:
    airport_label = st.selectbox("Airport", options=[name for _, name in AIRPORTS])
    airport_code  = dict((name, code) for code, name in AIRPORTS)[airport_label]

    year = st.slider("Year", min_value=2003, max_value=2025, value=2024)

# ── Auto-fill numeric features from historical averages ───────────────────────
key = (carrier_code, airport_code, month_num)
if key in historical_avg.index:
    stats = historical_avg.loc[key]
else:
    stats = global_stats

arr_flights    = stats['arr_flights']
cancel_rate    = stats['cancel_rate']
diversion_rate = stats['diversion_rate']
delay_rate     = stats['delay_rate']

st.info(
    f"📊 **Supporting features are filled automatically** from historical averages "
    f"for this airline/airport/month combination in the dataset. "
    f"Our ablation study found **delay rate** to be the single most important feature — "
    f"the historical average for this combination is **{delay_rate:.1%}** of flights delayed.",
    icon=None
)

# ── Prediction ─────────────────────────────────────────────────────────────────
if st.button("Predict delay", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        'carrier':        carrier_code,
        'airport':        airport_code,
        'month':          month_num,
        'year':           year,
        'arr_flights':    arr_flights,
        'cancel_rate':    cancel_rate,
        'diversion_rate': diversion_rate,
        'delay_rate':     delay_rate,
    }])

    prediction = max(0, model.predict(input_df)[0])

    if prediction < 5:
        verdict, color, bg, border = "🟢 Low delay",      "#064e3b", "#ecfdf5", "#10b981"
    elif prediction < 15:
        verdict, color, bg, border = "🟡 Moderate delay", "#78350f", "#fffbeb", "#f59e0b"
    else:
        verdict, color, bg, border = "🔴 High delay",     "#7f1d1d", "#fef2f2", "#ef4444"

    st.markdown(f"""
    <div style="background:{bg}; border-left:4px solid {border};
                border-radius:8px; padding:1.2rem 1.5rem; margin-top:1rem;">
        <p style="margin:0; font-size:0.85rem; color:{color}; font-weight:600;">{verdict}</p>
        <h2 style="margin:0.2rem 0 0; color:{color}; font-size:2.8rem; font-weight:700;">
            {prediction:.1f} min
        </h2>
        <p style="margin:0.4rem 0 0; color:#666; font-size:0.9rem;">
            Predicted average arrival delay for {carrier_label} flights
            arriving at {airport_label} in {month_label} {year}
        </p>
    </div>
    """, unsafe_allow_html=True)

    similar = df[
        (df['carrier'] == carrier_code) &
        (df['airport'] == airport_code) &
        (df['month']   == month_num)
    ]
    if len(similar) > 0:
        hist_avg = similar[TARGET].mean()
        st.caption(f"Historical average for this route: **{hist_avg:.1f} min** across {len(similar)} records in the dataset")
