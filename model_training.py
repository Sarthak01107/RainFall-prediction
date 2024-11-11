# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load data
data = pd.read_csv('data/rainfall_data.csv')

# Preprocess data
# Encode categorical columns and handle missing values
le = LabelEncoder()
for col in ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']:
    data[col] = le.fit_transform(data[col].astype(str))

# Select only numeric columns to calculate the mean for missing value imputation
df_numeric = data.select_dtypes(include=[float, int])  # Select only numeric columns
mean_values = df_numeric.mean()  # Calculate mean for numeric columns only

# Fill missing values in the original data using the calculated mean values
data = data.fillna(mean_values)

# Split data
X = data.drop(columns=['RainTomorrow', 'Date'])
y = data['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, 'scaler.pkl')  # Save the scaler for later use

# Model training with multiple algorithms
models = {
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'SVC': SVC()
}

best_model = None
best_score = 0

for model_name, model in models.items():
    model.fit(X_train, y_train)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    print(f"{model_name} Accuracy: {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_model = model

# Save the best model
joblib.dump(best_model, 'best_model.pkl')
print(f"Best Model: {best_model} with accuracy {best_score:.4f}")
