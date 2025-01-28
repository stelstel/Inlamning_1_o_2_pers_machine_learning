# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import os

# Clearing the Screen
os.system('cls')

# Step 2: Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'

column_names = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration",
                "num_of_doors", "body_style", "drive_wheels", "engine_location",
                "wheel_base", "length", "width", "height", "curb_weight", "engine_type",
                "num_of_cylinders", "engine_size", "fuel_system", "bore", "stroke",
                "compression_ratio", "horsepower", "peak_rpm", "city_mpg", "highway_mpg", "price"]

df = pd.read_csv(url, names=column_names)

df.to_csv('df_temp.csv') # //////////////////////////////////////

print(df.head()) #//////////////////////////////////

# Step 3: Preprocess the data

# Replace '?' with NaN and drop missing values'
df.replace('?', np.nan, inplace=True)

newdf = df.dropna() # Drop rows with missing values

newdf['price'] = newdf['price'].astype(int)

newdf.to_csv('df_temp_replaced.csv') # //////////////////////////////////////

print(newdf.head()) #//////////////////////////////////

# Convert categorical variables to numeric
newdf = pd.get_dummies(newdf, columns=[
    'make', 
    'num_of_cylinders',
    'fuel_type',
    'aspiration',
    'num_of_doors',
    'body_style',
    'drive_wheels',
    'engine_location',
    'engine_type',
    'fuel_system'
], drop_first=True)

# Step 4: Select features and target variable
X = newdf[['horsepower', 'engine_size', 'curb_weight', 'city_mpg', 'highway_mpg']]
y = newdf['price']

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features to improve performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)  # Use y_test (actual) and y_pred (predicted)
print(f"Mean Squared Error: {mse}")
