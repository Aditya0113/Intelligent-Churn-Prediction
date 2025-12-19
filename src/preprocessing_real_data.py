
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import sys

# Path provided by user
DATA_PATH = r"E:\Microsoft\Dataset\WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_PATH = r"E:\Microsoft\data\processed_telco_churn.csv"

print(f"Loading data from {DATA_PATH}...")
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: File not found at {DATA_PATH}")
    sys.exit(1)

# 1. Initial Cleanup
# 'TotalCharges' is often read as object due to empty strings " "
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# 2. Handle Missing Values
print("Handling missing values...")
# 'TotalCharges' has missing values after coercion (small number usually)
# Strategy: Fill with median for numerical
num_imputer = SimpleImputer(strategy='median')
df['TotalCharges'] = num_imputer.fit_transform(df[['TotalCharges']])

# Feature Engineering (New Request)
print("Creating new features...")

# 1. Average Monthly Usage
# Explanation: Helps identify if a customer is a heavy user relative to their tenure.
# Logic: TotalCharges / tenure. Handle division by zero.
df['Avg_Monthly_Usage'] = np.where(df['tenure'] > 0, df['TotalCharges'] / df['tenure'], 0)

# 2. Tenure Groups
# Explanation: Churn behavior often varies significantly by customer lifecycle stage.
# New customers (0-12 months) are often much riskier than established ones.
df['Tenure_Group'] = pd.cut(
    df['tenure'], 
    bins=[-1, 12, 24, 48, 999], 
    labels=['0-1 Year', '1-2 Years', '2-4 Years', '4+ Years']
)

# 3. Payment Type Simplification
# Explanation: Automatic payments often correlate with lower churn (stickier).
# Logic: Group 'Bank transfer (automatic)' and 'Credit card (automatic)' vs others.
df['Payment_Desc'] = df['PaymentMethod'].astype(str).str.lower()
df['Payment_Type_Simplified'] = np.where(
    df['Payment_Desc'].str.contains('automatic'), 
    'Automatic', 
    'Manual'
)
df = df.drop('Payment_Desc', axis=1) # Cleanup temporary column

print("Features created: Avg_Monthly_Usage, Tenure_Group, Payment_Type_Simplified")

# Drop customerID as it's not a feature
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

# 3. Define Features
target_col = 'Churn'
y = None
X = df

if target_col in df.columns:
    # Encode Target (Yes/No -> 1/0)
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])
    y = df[target_col]
    X = df.drop(target_col, axis=1)
else:
    print("Warning: 'Churn' column not found. Processing as inference data.")

# Identify columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numeric features: {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")

# 4. Encoding and Scaling
print("Encoding and Scaling...")

# Numerical: Scaling
# Categorical: OneHotEncoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ],
    verbose_feature_names_out=False
)

# Apply transformations
X_processed = preprocessor.fit_transform(X)

# Get feature names back
if hasattr(preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    feature_names = numeric_cols + list(cat_feature_names)
else:
    feature_names = numeric_cols + [f"cat_{i}" for i in range(X_processed.shape[1] - len(numeric_cols))]

# Convert back to DataFrame
X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

if y is not None:
    X_processed_df['Churn'] = y.values

print("Preprocessing complete.")
print(f"Processed Data Shape: {X_processed_df.shape}")

print("\nFirst 5 rows of processed data:")
print(X_processed_df.head())

# Save parsed data
X_processed_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved processed data to {OUTPUT_PATH}")
