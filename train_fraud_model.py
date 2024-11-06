import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the data
df = pd.read_csv('advanced_synthetic_transactions.csv')

# Data Preprocessing
# Drop irrelevant columns
df = df.drop(['TransactionID', 'UserID', 'CardID', 'DeviceID', 'IPAddress','Time','DayOfWeek','Hour','Location'], axis=1)

# Convert target variable to binary
df['Fraudulent'] = df['Fraudulent'].map({'Yes': 1, 'No': 0})

# Handle categorical variables using Label Encoding
categorical_features = ['Location_City', 'Location State','Merchant', 'MerchantCategory', 'TransactionType', 'IsRecurring', 'IsInternational']
le = LabelEncoder()
for col in categorical_features:
    df[col] = le.fit_transform(df[col])

# Feature and target separation
X = df.drop('Fraudulent', axis=1)
y = df['Fraudulent']

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model
y_pred = rf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model and scaler
joblib.dump(rf, 'fraud_model.joblib')
joblib.dump(le, 'label_encoder.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Model and scaler saved successfully.")
