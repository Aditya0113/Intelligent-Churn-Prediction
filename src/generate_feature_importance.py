import joblib
import pandas as pd
import os
import numpy as np

# Load Model
MODEL_PATH = 'models/churn_model.pkl'
if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}")
    exit()

model = joblib.load(MODEL_PATH)

# Extract Feature Names and Importances
try:
    # 1. Get Feature Names from Preprocessor
    preprocessor = model.named_steps['preprocessor']
    feature_names = preprocessor.get_feature_names_out()
    
    # 2. Get Importances from Classifier
    classifier = model.named_steps['classifier']
    importances = classifier.feature_importances_
    
    # 3. Create DataFrame
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # 4. Clean Feature Names (remove 'num__' and 'cat__' prefixes)
    df_imp['Feature'] = df_imp['Feature'].str.replace('num__', '').str.replace('cat__', '')
    
    # 5. Sort
    df_imp = df_imp.sort_values(by='Importance', ascending=False)
    
    # 6. Save
    csv_path = 'models/feature_importance.csv'
    df_imp.to_csv(csv_path, index=False)
    print(f"Feature importance saved to {csv_path}")
    print(df_imp.head(10))

except Exception as e:
    print(f"Error extracting feature importance: {e}")
