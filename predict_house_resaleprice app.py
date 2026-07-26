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
    df = pd.read_csv("dataset.csv")
    df["year"] = df["month"].str[:4].astype(int)
except FileNotFoundError as e:
    st.error(f"Required file not found: {e.filename}. Please make sure both 'house_predict_best_model.pkl' and 'dataset.csv' are in the same folder as this app.")
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

# The frontend always shows the newest year of house price
year = int(df["year"].max())



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
    # (df is already loaded and cached at the top, so it is not re-read here)
    same_type = df[df["flat_type"] == flat_type]

    # Prices in this dataset nearly doubled between 2000 and 2012, so comparing
    # the estimate against all years at once would measure the year of sale, not
    # the flat. The comparison is limited to sales around the selected year. The
    # window starts at +/- 2 years and widens only when that is too thin to plot,
    # because the most recent years are sparse (EXECUTIVE has no 2012 sales).
    MIN_SALES = 20
    span = int(df["year"].max()) - int(df["year"].min())
    half_width = 2
    window = same_type[same_type["year"].between(year - half_width, year + half_width)]
    while len(window) < MIN_SALES and half_width < span:
        half_width += 2
        window = same_type[same_type["year"].between(year - half_width, year + half_width)]

    low_year = max(int(df["year"].min()), year - half_width)
    high_year = min(int(df["year"].max()), year + half_width)
    period = f"{low_year}-{high_year}"

    if len(window) < 5:
        st.info(f"There are too few recorded {flat_type} sales to draw a comparison")
    else:
        # Group past prices into 10 bins. value_counts(bins=...) returns Interval
        # objects, which do not display readably, so relabel them as "142k-167k"
        bins = window["resale_price"].value_counts(bins=10).sort_index()
        labels = [f"{int(i.left/1000)}k-{int(i.right/1000)}k" for i in bins.index]

        # Split the counts into two columns so the band holding the estimate is
        # drawn in a different colour, making the user's own price visible on the
        # chart instead of only being mentioned in the text below it
        chart_data = pd.DataFrame({"Past sales": bins.values,
                                   "Your estimate": 0}, index=labels)

        matched = False
        for pos, interval in enumerate(bins.index):
            if interval.left < price <= interval.right:
                chart_data.iloc[pos, 1] = chart_data.iloc[pos, 0]  # move count across
                chart_data.iloc[pos, 0] = 0
                matched = True

        # Checked AFTER the loop, not inside it. Inside the loop this fires once
        # for every band before the matching one.
        if not matched:
            st.info("Your estimate falls outside the range of past sales for this "
                    "flat type, so it is not highlighted on the chart below")

        median_price = window["resale_price"].median()

        st.write(f"How this compares to other {flat_type} flats sold in "
                 f"Bukit Timah, {period}:")
        st.bar_chart(chart_data, horizontal=True)

        # Note: "$" starts LaTeX maths in Streamlit markdown, so it is escaped as "\$"
        st.caption(f"Your estimate: S\\${price:,.0f} | "
                   f"Median {flat_type} sold {period}: S\\${median_price:,.0f} "
                   f"(based on {len(window)} sales)")