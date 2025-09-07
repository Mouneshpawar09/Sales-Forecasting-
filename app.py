import os
import streamlit as st
import pandas as pd
from retrain import retrain_model
from predict import predict_units_sold

st.set_page_config(page_title="Sales Forecasting", layout="centered")
st.title("🛒 Sales Forecasting & Inventory Planner")

os.makedirs("data", exist_ok=True)
st.header("📤 Upload New Sales Data (CSV)")
uploaded_file = st.file_uploader("Choose a CSV file with new sales data", type="csv")

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    new_data.to_csv("data/new_sales_data.csv", index=False)
    st.success("New data uploaded.")

    if st.button("🔁 Combine & Retrain"):
        msg = retrain_model("data/new_sales_data.csv", "data/sales_data.csv")
        st.success(msg)

st.divider()
st.header("🔮 Predict Single Entry")

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        price = st.number_input("Price", min_value=0.0, value=10.0)
        promotion = st.selectbox("Promotion?", [0, 1])
        holiday = st.selectbox("Holiday?", [0, 1])
    with col2:
        weather_index = st.number_input("Weather Index", value=0.0)
        day_of_week = st.selectbox("Day of Week (0=Mon)", list(range(7)))
        month = st.selectbox("Month", list(range(1, 13)))

    submitted = st.form_submit_button("📈 Predict Demand")
    if submitted:
        input_data = {
            "price": price,
            "promotion": promotion,
            "holiday": holiday,
            "weather_index": weather_index,
            "day_of_week": day_of_week,
            "month": month
        }
        prediction = predict_units_sold(input_data)
        st.success(f"📦 Predicted Units Sold: **{prediction}**")


