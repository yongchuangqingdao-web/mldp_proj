""" Bukit Timah HDB Resale Price Estimator"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Resale Price Estimator", page_icon="🏠")

# --- Load the saved training model ---
# The .pkl file holds 4 things: the model, the scaler, the kmeans, and the list of column names
# The scaler and kmeans are needed to rebuild the cluster feature, and the column list keeps the colunms in the same order as training

try:
    bundle = joblib.load("house_predict_best_model.pkl")
except FileNotFoundError:
    st.error("Model file not found. Please make sure "
             "'house_predict_best_model.pkl' is in the same folder as this app.")

    st.stop()

model = bundle["model"]
scaler = bundle["scaler"]
kmeans = bundle["kmeans"]
columns = bundle["columns"]



#--- Page heading ---
st.title("🏠 Bukit Timah Resale Price Estimator")
st.write("Fill in the flat details below to get an estimated resale price. "
         "The estimate is based on 938 real transactions from 2000 to 2012")

#--- User inputs ---
col1, col2 = st.columns(2)

with col1:
    flat_type = st.selectbox("Flat type",
                             ["3 ROOM","4 ROOM","5 ROOM","EXECUTIVE"])
    storey_range = st.selectbox("Storey",
                                ["01 TO 03","04 TO 06","07 TO 09","10 TO 12","13 TO 15","16 TO 18","19 TO 21","22 TO 24"])

with col2:
    flat_model = st.selectbox("Flat model",
                              ["Adjoined flat","Apartment","Improved","Maisonette","Model A","Simplified","Standard"])

    floor_area = st.number_input("Floor area (sqm)",
                                 min_value=63, max_value=154, value=100)

# The model only learnt from sales up to 2012, so the slider stops there.
year = st.slider("Year of sale", 2000, 2012, value=2012)



#--- Input validation ---
# Warn the user when the combination is rare in the training data, so they know the estimate is less trustworthy instead of trusting a wrong number

if flat_type == "3 ROOM" and floor_area > 100:
    st.warning("A 3 ROOM flat larger than 100 sqm is rare in the data "
               "The estimate may be less accurate")
elif flat_type == "EXECUTIVE" and floor_area < 120:
    st.warning("An EXECUTIVE flat smaller than 120 sqm is rare in the data "
               "The estimate may be less accurate")

#--- Predict ---
if st.button("Estimate price", type="primary"):

    # step1: turn the floor value into middle floor
    low, high = storey_range.split("TO")
    storey_mid = (int(low) + int(high)) / 2

    # step2: rebuild the cluster feature the same way as in the notebook
    scaled = scaler.transform(pd.DataFrame([[storey_mid,floor_area]],
                                           columns=["storey_mid","floor_area_sqm"]))
    cluster = str(kmeans.predict(scaled)[0])

    # step3: put the input into one row
    new_flat = pd.DataFrame([{
        "floor_area_sqm": floor_area,
        "storey_mid": storey_mid,
        "storey_area_cluster": cluster,
        "year": year,
        "flat_type": flat_type,
        "flat_model": flat_model,
    }])

    # step4: OHE then line the columns up with the training columns
    # No drop_first here: one row has only one value per category, so dropping the first would remove the only colunm that is filled in.
    # reindex adds back every missing colunm as 0, which matches how training encoded them
    new_flat = pd.get_dummies(new_flat)
    new_flat = new_flat.reindex(columns=columns, fill_value=0)

    # step5: predict and show the result
    price = model.predict(new_flat)[0]
    st.metric("Estimated resale price", f"S${price:,.0f}")
    st.caption("The model is typically off by about $31,000, so use this as a guide price rather than a formal valuation")

    # step6: show how this estimate compares to real past sales
    df = pd.read_csv("dataset.csv")
    same_type = df[df["flat_type"] == flat_type]
    median_price= same_type["resale_price"].median()
    
    # Group past prices into 10 bins. value_conts(bins=...) returns Interval
    # objects, which do not dispaly readably, so relabel them as "142k-167k"
    bands = same_type["resale_price"].value_counts(bins=10).sort_index()
    bands.index = [f"{int(i.left/1000)}k-{int(i.right/1000)}k" for i in bands.index]

    st.write(f"How this compares to other {flat_type} flats in Bukit Timah: ")
    st.bar_chart(bands, horizontal=True)

    # Note: "$" starts LaTeX maths in Streamlit markdown, so it is escaped as "\$"
    st.caption(f"Your estimate: S\\${price:,.0f} | "
               f"Median {flat_type}: S\\${median_price:,.0f}")
