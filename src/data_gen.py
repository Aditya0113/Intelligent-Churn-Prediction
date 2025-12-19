
import pandas as pd
import numpy as np
import os

print("Generating synthetic data...")
np.random.seed(42)
n_rows = 2000

# Customer IDs
customer_ids = [f'C{str(i).zfill(5)}' for i in range(1, n_rows + 1)]

# Feature 1: Tenure (months) - mixture of new and loyal customers
tenure = np.random.randint(1, 72, n_rows)

# Feature 2: Monthly Bill ($)
monthly_bill = np.random.normal(70, 30, n_rows)
monthly_bill = np.clip(monthly_bill, 20, 150)

# Feature 3: Data Usage GB
usage_gb = monthly_bill * 5 + np.random.normal(0, 50, n_rows) # Correlated with bill
usage_gb = np.clip(usage_gb, 10, 1000)

# Feature 4: Customer Support Calls (Higher calls -> Higher churn risk)
support_calls = np.random.poisson(2, n_rows)

# Feature 5: Subscription Type
sub_types = ['Basic', 'Standard', 'Premium']
subscription = np.random.choice(sub_types, n_rows, p=[0.4, 0.4, 0.2])

# Feature 6: Contract Type
contract_types = ['Month-to-month', 'One year', 'Two year']
contract = np.random.choice(contract_types, n_rows)

# Target: Churn (Probabilistic based on features)
churn_prob = np.zeros(n_rows)

# Logic for churn probability
churn_prob += 0.4 * (contract == 'Month-to-month') # Month-to-month risky
churn_prob -= 0.2 * (contract == 'Two year')       # Long contract safe
churn_prob += 0.1 * (support_calls > 3)            # Many calls risky
churn_prob += 0.05 * (monthly_bill > 100)          # High bill slightly risky
churn_prob -= 0.3 * (tenure > 24)                  # Loyal customers safe

# Add noise
churn_prob += np.random.normal(0, 0.1, n_rows)
churn_prob = np.clip(churn_prob, 0.1, 0.9) # Bound probabilities

churn = (np.random.rand(n_rows) < churn_prob).astype(int)

cols = {
    'CustomerID': customer_ids,
    'Tenure_Months': tenure,
    'Monthly_Bill': monthly_bill.round(2),
    'Total_Usage_GB': usage_gb.round(2),
    'Support_Calls': support_calls,
    'Subscription_Type': subscription,
    'Contract_Term': contract,
    'Churn': churn
}

df = pd.DataFrame(cols)

output_path = os.path.join('data', 'customer_churn_data.csv')
if not os.path.exists('data'):
    os.makedirs('data')
    
df.to_csv(output_path, index=False)
print(f"Data saved to {output_path}")
print(df.head())
