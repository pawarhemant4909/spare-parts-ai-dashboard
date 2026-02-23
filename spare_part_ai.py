import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("sales_data.csv")
df["Date"] = pd.to_datetime(df["Date"],format="%d-%m-%Y")
df["Month"] = df["Date"].dt.month
print("\ntop selling part:")
print(df.groupby("Part_Name")["Quantity_Sold"].sum().sort_values(ascending=False))
df["Profit"] = (df["Selling_Price"] - df["Purchase_Price"]) * df["Quantity_Sold"]
print("\ntotal Profit:",df["Profit"].sum())
df["Part_Code"] = df["Part_Name"].astype("category").cat.codes

X = df[["Month", "Part_Code"]]
y = df["Quantity_Sold"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

print("\nPredicted Demand for March:")
unique_parts = df["Part_Name"].unique()
for part in unique_parts:
    part_code = df[df["Part_Name"]==part]["Part_Code"].iloc[0]
    predication = model.predict([[3,part_code]])
    print(part, ":", round(predication[0]))

#smart stock alert
print("\nStock Alerts:")
for paret in unique_parts:
    part_data = df[df["Part_Name"] == part]
    part_code = part_data["Part_Code"].iloc[0]
    current_stock = part_data["Stock_Left"].iloc[-1]

    predicted = model.predict([[3,part_code]])[0]
    if predicted > current_stock:
        print("Order More", part)