# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import kagglehub
import os

# Clearing the Screen
os.system('cls')

# Download latest version
path = kagglehub.dataset_download("mohansacharya/graduate-admissions")

# Load the dataset
df = pd.read_csv(f"{path}/Admission_Predict.csv")

# Preprocess the data
df['Chance of Admit '] = (df['Chance of Admit '] >= 0.75).astype(int)  # Convert to binary classification. true/false -> 1/0
X = df[['GRE Score', 'TOEFL Score', 'CGPA']]
y = df['Chance of Admit ']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) 
print(f"Accuracy: {accuracy}")

print("********** End **********")
