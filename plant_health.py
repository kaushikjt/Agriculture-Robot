import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

file_path = "plant_health_data.xlsx"
df = pd.read_excel(file_path)


encoder = LabelEncoder()
df["Leaf_Color"] = encoder.fit_transform(df["Leaf_Color"])


healthy = df[df["Plant_Health"] == 1]
not_healthy = df[df["Plant_Health"] == 0]
balanced_df = pd.concat([healthy.sample(len(not_healthy), replace=True), not_healthy])


X = balanced_df[["Leaf_Color", "Soil_Moisture", "Humidity", "Air_Quality"]]
y = balanced_df["Plant_Health"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

leaf_color_input = input("Enter Leaf Color (Green, Yellow, Brown): ").strip().capitalize()
soil_moisture_input = float(input("Enter Soil Moisture (%): "))
humidity_input = float(input("Enter Humidity (%): "))
air_quality_input = float(input("Enter Air Quality Index (AQI): "))


leaf_color_encoded = encoder.transform([leaf_color_input])[0]


input_data = np.array([[leaf_color_encoded, soil_moisture_input, humidity_input, air_quality_input]])


prediction = model.predict(input_data)[0]
predicted_health = "Healthy" if prediction == 1 else "Not Healthy"


print("\nPlant Health Prediction")
print(f"Leaf Color: {leaf_color_input}")
print(f"Soil Moisture: {soil_moisture_input}%")
print(f"Humidity: {humidity_input}%")
print(f"Air Quality Index: {air_quality_input}")
print(f"Prediction: {predicted_health}")
