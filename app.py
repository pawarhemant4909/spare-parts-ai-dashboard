import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.title("Spare Part Demand Predication System")

#load data
df = pd.read_csv("sales_data.csv")
df["Date"]= pd.to_datetime(df["Date"], format="%d-%m-%Y")
df["Month"] = df["Date"].dt.month

#profit calcu
df["Profit"] = (df["Selling_Price"] - df["Purchase_Price"]) * df["Quantity_Sold"]
st.subheader("Sale Overview")

#monthly sales
monthly_sales = df.groupby("Month")["Quantity_Sold"].sum()
st.bar_chart(monthly_sales)

st.subheader("Total Profit")
st.write("$",df["Profit"].sum())

#ml model 
df["Part_Code"] = df["Part_Name"].astype("category").cat.codes
X = df[["Month", "Part_Code"]]
y = df["Quantity_Sold"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor()
model.fit(X_train, y_train)

st.subheader("Predict Next Month Demand")
part_selected = st.selectbox("Select Part", df["Part_Name"].unique())
month_input = st.number_input("Enter Month (1-12)", min_value=1, max_value=12, value=3)

part_code = df[df["Part_Name"] == part_selected]["Part_Code"].iloc[0]
predication = model.predict([[month_input, part_code]])
st.write("predicted Demand:", round(predication[0]))

#stock alert
current_stock = df[df["Part_Name"] == part_selected]["Stock_Left"].iloc[-1]

if predication[0] > current_stock:
    st.error(":( LOW STOCK !, ORDER MORE.")
else:
    st.success("STOCK LEVEL OK :)")

import streamlit as st
import pandas as pd
#load sale data
df = pd.read_csv("sales_data.csv")
st.title("spare part stock editior")
edited_df = st.data_editor(df,num_rows="dynamic")
if st.button("Save Changes"):
    edited_df.to_csv("sales_data.csv", index=False)
    st.sucess("Stock Updated!")
st.subheader("Add Part")
new_part = st.text_input("Part name")
new_stock = st.number_input("Stock Available", min_value=0)
new_price = st.number_input("unit Price", min_value=0)
if st.button("Add Part"):
    if new_part:
        df.loc[len(df)] = [new_part, new_stock, new_price]
        df.to_csv("sales_data.csv", index=False)
        st.success(f"{new_part} added scuessfully!")
    else: 
        st.error("Enter a part name!")

