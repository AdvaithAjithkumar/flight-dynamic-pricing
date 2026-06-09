import streamlit as st  # type: ignore[reportMissingImports]
import pandas as pd  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]
import joblib  # type: ignore[reportMissingImports]

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Flight Dynamic Pricing",
    page_icon="✈️",
    layout="centered"
)

# ── Load Model & Feature Columns ──────────────────────────
@st.cache_resource
def load_model():
    import importlib
    xgb = importlib.import_module("xgboost")
    booster = xgb.Booster()
    booster.load_model("models/xgboost_tuned.json")
    feature_cols = joblib.load("models/feature_columns.pkl")
    return booster, feature_cols, xgb

model, feature_cols, xgb = load_model()

# ── Header ────────────────────────────────────────────────
st.title("Flight Dynamic Pricing Predictor")
st.markdown(
    "Predicts optimal flight ticket price based on booking conditions. "
    "Built on 300K+ Indian flight records using XGBoost (R²=0.9858)."
)
st.divider()

# ── Input Form ────────────────────────────────────────────
st.subheader("Flight Details")

col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox("Airline", [
        "Air_India", "AirAsia", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"
    ])
    source_city = st.selectbox("Source City", [
        "Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"
    ])
    destination_city = st.selectbox("Destination City", [
        "Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"
    ])
    flight_class = st.radio("Class", ["Economy", "Business"])

with col2:
    days_left = st.slider("Days Before Departure", min_value=1, max_value=49, value=15)
    duration = st.slider("Flight Duration (hours)", min_value=1.0, max_value=30.0,
                         value=2.5, step=0.25)
    stops = st.selectbox("Number of Stops", ["zero", "one", "two_or_more"])
    departure_time = st.selectbox("Departure Time", [
        "Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"
    ])
    arrival_time = st.selectbox("Arrival Time", [
        "Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"
    ])

st.divider()

# ── Feature Engineering (mirror phase3 exactly) ──────────
def build_features(airline, source_city, destination_city, flight_class,
                   days_left, duration, stops, departure_time, arrival_time):

    # Base dict with all feature columns set to 0
    row = {col: 0 for col in feature_cols}

    # Numeric
    row["duration"]  = duration
    row["days_left"] = days_left

    # class_encoded
    row["class_encoded"] = 1 if flight_class == "Business" else 0

    # stops_encoded
    stops_map = {"zero": 0, "one": 1, "two_or_more": 2}
    row["stops_encoded"] = stops_map[stops]

    # departure_time_encoded
    dep_map = {
        "Late_Night": 0, "Afternoon": 1, "Morning": 2,
        "Evening": 3, "Early_Morning": 4, "Night": 5
    }
    row["departure_time_encoded"] = dep_map[departure_time]

    # arrival_time_encoded
    arr_map = {
        "Late_Night": 0, "Afternoon": 1, "Morning": 2,
        "Evening": 3, "Early_Morning": 4, "Night": 5
    }
    row["arrival_time_encoded"] = arr_map[arrival_time]

    # booking_urgency
    if days_left >= 50:
        row["booking_urgency"] = 0
    elif days_left >= 17:
        row["booking_urgency"] = 1
    else:
        row["booking_urgency"] = 2

    # airline one-hot (AirAsia is the dropped baseline)
    airline_cols = {
        "Air_India": "airline_Air_India",
        "GO_FIRST":  "airline_GO_FIRST",
        "Indigo":    "airline_Indigo",
        "SpiceJet":  "airline_SpiceJet",
        "Vistara":   "airline_Vistara"
    }
    if airline in airline_cols:
        row[airline_cols[airline]] = 1

    # source_city one-hot (Bangalore is dropped baseline)
    src_cols = {
        "Chennai":   "src_Chennai",
        "Delhi":     "src_Delhi",
        "Hyderabad": "src_Hyderabad",
        "Kolkata":   "src_Kolkata",
        "Mumbai":    "src_Mumbai"
    }
    if source_city in src_cols:
        row[src_cols[source_city]] = 1

    # destination_city one-hot (Bangalore is dropped baseline)
    dst_cols = {
        "Chennai":   "dst_Chennai",
        "Delhi":     "dst_Delhi",
        "Hyderabad": "dst_Hyderabad",
        "Kolkata":   "dst_Kolkata",
        "Mumbai":    "dst_Mumbai"
    }
    if destination_city in dst_cols:
        row[dst_cols[destination_city]] = 1

    # route one-hot
    route = f"{source_city}_to_{destination_city}"
    route_col = f"route_{route}"
    if route_col in row:
        row[route_col] = 1

    return pd.DataFrame([row])[feature_cols]


# ── Predict Button ────────────────────────────────────────
if st.button("Predict Price", type="primary", use_container_width=True):

    if source_city == destination_city:
        st.error("Source and destination cities cannot be the same.")
    else:
        X_input = build_features(
            airline, source_city, destination_city, flight_class,
            days_left, duration, stops, departure_time, arrival_time
        )

        dmatrix = xgb.DMatrix(X_input)
        pred_log = model.predict(dmatrix)[0]
        pred_price = int(np.expm1(pred_log))

        # Pricing tier
        if pred_price < 7000:
            tier = "Budget"
        elif pred_price < 30000:
            tier = "Standard"
        else:
            tier = "Premium"

        # Results
        st.divider()
        st.subheader("Predicted Price")

        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated Price", f"₹{pred_price:,}")
        c2.metric("Pricing Tier", tier)
        c3.metric("Days Left", days_left)

        # Booking urgency message
        if days_left < 17:
            st.warning(
                "Last-minute booking detected — prices are elevated. "
                "You're in the surge zone (under 17 days). "
                "Booking earlier could save significantly."
            )
        elif days_left >= 17 and days_left < 50:
            st.info("Medium booking window — prices are climbing. Book soon for better rates.")
        else:
            st.success("Early bird window — you're getting the best available rates.")

        # What's driving the price
        st.divider()
        st.subheader("What's Driving This Price")
        drivers = []
        if flight_class == "Business":
            drivers.append("Business class is the largest price driver (+premium)")
        if days_left < 17:
            drivers.append(f"Only {days_left} days left — last-minute surge pricing active")
        if airline == "Vistara":
            drivers.append("Vistara commands a premium over budget carriers")
        if airline in ["AirAsia", "Indigo", "SpiceJet", "GO_FIRST"]:
            drivers.append(f"{airline} is a budget carrier — lower base price")
        if stops == "one" and flight_class == "Business":
            drivers.append("One-stop Business flight — typically higher priced long-haul route")
        if duration > 10:
            drivers.append(f"Long flight duration ({duration}h) adds to price")

        for d in drivers:
            st.markdown(f"- {d}")

st.divider()
st.caption(
    "Model: XGBoost (Optuna-tuned) | Dataset: 300K+ Indian flight records | "
    "R²=0.9858 | RMSE=₹2,710 | Features: 51"
)