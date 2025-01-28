import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

column_names = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration",
                "num_of_doors", "body_style", "drive_wheels", "engine_location",
                "wheel_base", "length", "width", "height", "curb_weight", "engine_type",
                "num_of_cylinders", "engine_size", "fuel_system", "bore", "stroke",
                "compression_ratio", "horsepower", "peak_rpm", "city_mpg", "highway_mpg", "price"]

df = pd.read_csv(url, names=column_names)

# Step 2: Preprocess the data
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Convert columns to numeric
df['price'] = df['price'].astype(int)
df['horsepower'] = df['horsepower'].astype(int)
df['engine_size'] = df['engine_size'].astype(int)
df['curb_weight'] = df['curb_weight'].astype(int)

# Remove extreme outliers
df = df[df['price'] < 50000]  # Remove luxury cars

# Convert categorical variables
df = pd.get_dummies(df, columns=['make'], drop_first=True)

# Step 3: Select features and target variable
X = df[['horsepower', 'engine_size', 'curb_weight', 'city_mpg', 'highway_mpg']]
y = df['price']

# Step 4: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")