import streamlit as st
import joblib
import numpy as np

# --------------------------------
# Load model
# --------------------------------
model = joblib.load("energy_model.pkl")

try:
    scaler = joblib.load("scaler.pkl")
    use_scaler = True
except:
    use_scaler = False


# --------------------------------
# Page config
# --------------------------------
st.set_page_config(
    page_title="Energy Consumption Prediction",
    page_icon="⚡",
    layout="wide"
)

# --------------------------------
# Custom CSS
# --------------------------------
st.markdown("""
<style>

/* Hide Streamlit UI */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Background */
.stApp{
    background: linear-gradient(
        135deg,
        #eef2ff,
        #f8fafc
    );
}

/* Bigger title */
.title{
    font-size:55px;
    font-weight:800;
    color:#1e293b;
}

/* Bigger subtitle */
.subtitle{
    font-size:22px;
    color:#64748b;
    margin-bottom:30px;
}

/* Bigger labels */
label{
    font-size:18px !important;
    font-weight:600 !important;
}

/* Bigger input text */
input{
    font-size:18px !important;
    border-radius:12px !important;
}

/* Bigger dropdown */
div[data-baseweb="select"]{
    font-size:18px !important;
}

/* Bigger button */
div.stButton > button{
    width:100%;
    background: linear-gradient(
        90deg,
        #2563eb,
        #4f46e5
    );
    color:white;
    border:none;
    padding:16px;
    border-radius:15px;
    font-size:22px;
    font-weight:700;
}

div.stButton > button:hover{
    background:#1d4ed8;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# Popup dialog
# --------------------------------
@st.dialog("Prediction Result")
def show_prediction(pred):
    st.markdown(
        f"""
        <h1 style="
            text-align:center;
            color:#2563eb;
            font-size:50px;
        ">
            ⚡ {pred:.2f} kWh
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.write(
        "Predicted energy consumption generated "
        "using trained machine learning model."
    )

    if st.button("OK"):
        st.rerun()


# --------------------------------
# Header
# --------------------------------
st.markdown(
    '<div class="title">⚡ Energy Consumption Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict building energy consumption using machine learning models.</div>',
    unsafe_allow_html=True
)


# --------------------------------
# Layout
# --------------------------------
left, right = st.columns(2)

with left:
    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=1
    )

    hour = st.number_input(
        "Hour",
        min_value=0,
        max_value=23,
        value=0
    )

    day = st.selectbox(
        "Day of Week",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )

    holiday = st.selectbox(
        "Holiday",
        ["No", "Yes"]
    )

with right:
    temp = st.number_input("Temperature (°C)")
    humidity = st.number_input("Humidity (%)")
    square_footage = st.number_input("Square Footage")
    occupancy = st.number_input("Occupancy")
    hvac = st.number_input("HVAC Usage")
    lighting = st.number_input("Lighting Usage")
    renewable = st.number_input("Renewable Energy")


# --------------------------------
# Predict
# --------------------------------
if st.button("Predict Energy Consumption"):

    day_map = {
        "Monday":0,
        "Tuesday":1,
        "Wednesday":2,
        "Thursday":3,
        "Friday":4,
        "Saturday":5,
        "Sunday":6
    }

    holiday_map = {
        "No":0,
        "Yes":1
    }

    features = np.array([[
        month,
        hour,
        day_map[day],
        holiday_map[holiday],
        temp,
        humidity,
        square_footage,
        occupancy,
        hvac,
        lighting,
        renewable
    ]])

    if use_scaler:
        features = scaler.transform(features)

    prediction = model.predict(features)[0]

    # POPUP
    show_prediction(prediction)