import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Fetal Health Predictor", layout="wide")
st.title("Fetal Health Classification App")
st.write("Predict fetal health (Normal / Suspect / Pathological)")


@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    return df 


df = load_data()

menu = st.sidebar.selectbox(
    "Navigation",
    [ "Prediction"]
)



if menu == "Prediction":
    st.header("Predict Fetal Health")

    columns = [
        'baseline value',
        'accelerations',
        'uterine_contractions',
        'light_decelerations',
        'severe_decelerations',
        'prolongued_decelerations',
        'abnormal_short_term_variability',
        'mean_value_of_short_term_variability',
        'percentage_of_time_with_abnormal_long_term_variability',
        'mean_value_of_long_term_variability',
        'histogram_width',
        'histogram_min',
        'histogram_max',
        'histogram_number_of_peaks',
        'histogram_number_of_zeroes',
        'histogram_median',
        'histogram_variance',
        'histogram_tendency',
        'fetal_movement_binary'
    ]

    input_data = {}

    for col in columns:
        input_data[col] = st.number_input(f"Enter {col}", value=0.0)

    input_df = pd.DataFrame([input_data])

    model = joblib.load("rf.pkl")  

    if st.button("Predict"):
        prediction = model.predict(input_df)
        result = prediction[0]
        if result == 1:
            st.success("Normal")
        elif result == 2:
            st.warning("Suspect")
        elif result == 3:
            st.error("Pathological")
        else:
            st.info(f"Unknown prediction result: {result}") 