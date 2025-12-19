
import pandas as pd
import numpy as np
import sys
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 1. Load Raw Data
# We load RAW data so we can build a pipeline that handles raw inputs (essential for the App)
DATA_PATH = r"E:\Microsoft\Dataset\WA_Fn-UseC_-Telco-Customer-Churn.csv"
print(f"Loading raw data from {DATA_PATH}...")

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: {DATA_PATH} not found.")
    sys.exit(1)

# 2. Preprocessing & Feature Engineering (Manual Step)
# This logic duplicates preprocessing_real_data.py to ensure the model is trained on the same engineered features.
# The App will typically need to replicate these 'Feature Engineering' steps before passing data to the Pipeline.

# Cleanup
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

# Feature Eng 1: Avg Monthly Usage
df['Avg_Monthly_Usage'] = np.where(df['tenure'] > 0, df['TotalCharges'] / df['tenure'], 0)

# Feature Eng 2: Tenure Groups
df['Tenure_Group'] = pd.cut(
    df['tenure'], 
    bins=[-1, 12, 24, 48, 999], 
    labels=['0-1 Year', '1-2 Years', '2-4 Years', '4+ Years']
)

# Feature Eng 3: Payment Type Simplified
df['Payment_Desc'] = df['PaymentMethod'].astype(str).str.lower()
df['Payment_Type_Simplified'] = np.where(df['Payment_Desc'].str.contains('automatic'), 'Automatic', 'Manual')
df = df.drop('Payment_Desc', axis=1)

print("Feature Engineering complete.")

# 3. Setup X and y
target_col = 'Churn'
# Encode Target manually for y
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

X = df.drop(target_col, axis=1)
y = df[target_col]

# Define Feature Groups for Pipeline
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numeric Features: {numeric_features}")
print(f"Categorical Features: {categorical_features}")

# 4. Define Pipeline
# Preprocessor handles Scaling and OneHotEncoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

# Pipeline
# We train RandomForest as the final model
final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Train-Test Split & Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training final pipeline (Random Forest)...")
final_model.fit(X_train, y_train)

# 6. Evaluation
y_pred = final_model.predict(X_test)
y_prob = final_model.predict_proba(X_test)[:, 1]

print("\nFinal Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# 7. Save Model
if not os.path.exists('models'):
    os.makedirs('models')

save_path = 'models/churn_model.pkl'
joblib.dump(final_model, save_path)
print(f"\nSaved Final Pipeline to {save_path}")
print("This pipeline accepts a DataFrame with raw features (including engineered ones).")
