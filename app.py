import streamlit as st
import pandas as pd
import pickle
from huggingface_hub import hf_hub_download
st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒")
st.title("🛒 SuperKart — Sales Forecast")
st.write("Predict total product revenue in a store using product and store attributes.")
@st.cache_resource
def load_artifacts():
    model_path   = hf_hub_download(repo_id="dvrwar/superkart-model",
                                   filename="best_model.pkl")
    encoder_path = hf_hub_download(repo_id="dvrwar/superkart-model",
                                   filename="encoders.pkl")
    with open(model_path,   "rb") as f: model    = pickle.load(f)
    with open(encoder_path, "rb") as f: encoders = pickle.load(f)
    return model, encoders
model, encoders = load_artifacts()
st.subheader("Product Details")
col1, col2 = st.columns(2)
with col1:
    weight = st.number_input("Product Weight (kg)", 1.0, 25.0, 12.0)
    mrp    = st.number_input("Product MRP (₹)", 30.0, 300.0, 150.0)
    area   = st.slider("Allocated Display Area Ratio", 0.001, 0.35, 0.05, 0.001)
with col2:
    sugar   = st.selectbox("Sugar Content",  list(encoders["Product_Sugar_Content"].classes_))
    ptype   = st.selectbox("Product Type",   list(encoders["Product_Type"].classes_))
st.subheader("Store Details")
col3, col4 = st.columns(2)
with col3:
    size   = st.selectbox("Store Size", list(encoders["Store_Size"].classes_))
    city   = st.selectbox("City Tier",  list(encoders["Store_Location_City_Type"].classes_))
with col4:
    stype  = st.selectbox("Store Type", list(encoders["Store_Type"].classes_))
    age    = st.slider("Store Age (years)", 1, 50, 15)
if st.button("🔮 Predict Sales Revenue", use_container_width=True):
    inp = {
        "Product_Weight":           weight,
        "Product_Sugar_Content":    encoders["Product_Sugar_Content"].transform([sugar])[0],
        "Product_Allocated_Area":   area,
        "Product_Type":             encoders["Product_Type"].transform([ptype])[0],
        "Product_MRP":              mrp,
        "Store_Size":               encoders["Store_Size"].transform([size])[0],
        "Store_Location_City_Type": encoders["Store_Location_City_Type"].transform([city])[0],
        "Store_Type":               encoders["Store_Type"].transform([stype])[0],
        "Store_Age":                age
    }
    pred = model.predict(pd.DataFrame([inp]))[0]
    st.success(f"### 💰 Predicted Sales: ₹ {pred:,.2f}")
st.caption("SuperKart MLOps Project")
