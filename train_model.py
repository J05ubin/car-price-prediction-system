import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("cardekho_dataset.csv")

# Drop unwanted columns
df.drop(["Unnamed: 0", "car_name"], axis=1, inplace=True)

# Define X and y
X = df.drop("selling_price", axis=1)
y = np.log(df["selling_price"])  # log transform improves performance

# Categorical and numerical columns
categorical_cols = ["brand", "fuel_type", "transmission_type", "seller_type"]
numerical_cols = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# Model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Save model
with open("car_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")
