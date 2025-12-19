# Intelligent Customer Churn Prediction System

## 1. Project Overview
This project is an end-to-end Machine Learning solution designed to predict customer churn for a telecommunications company. By analyzing customer demographics, account information, and usage patterns, the system identifies high-risk customers, enabling proactive retention strategies.

**Key Features:**
*   **Data Processing**: Automated pipeline to handle missing values, encode categorical variables, and scale numerical features.
*   **Feature Engineering**: Creation of behavioral features like Average Monthly Usage and Tenure Groups.
*   **Machine Learning**: A **Random Forest Classifier** trained to predict churn probability.
*   **Deployment**: An interactive **Streamlit** web application for real-time inference.

## 2. Dataset Used
The project uses the **Telco Customer Churn** dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv).
*   **Rows**: ~7,000 customers
*   **Features**: 21 variables including:
    *   **Demographics**: Gender, Senior Citizen, Partner, Dependents.
    *   **Services**: Phone, Internet, Online Security, Tech Support, Streaming.
    *   **Account**: Tenure, Contract, Payment Method, Monthly Charges, Total Charges.
    *   **Target**: Churn (Yes/No).

## 3. Steps Followed
The development followed a structured Data Science lifecycle:

1.  **Data Analysis (EDA)**: Explored relationships between features and churn (e.g., impact of Month-to-month contracts and Electronic Check payments).
2.  **Preprocessing**:
    *   Imputed missing `TotalCharges` values.
    *   Created simple procedural scripts for reproducible processing.
3.  **Feature Engineering**: implemented robust features:
    *   `Avg_Monthly_Usage`: Identifies utilization rate.
    *   `Tenure_Group`: Segments customers by lifecycle stage.
4.  **Model Training**:
    *   Trained **Logistic Regression** and **Random Forest** models.
    *   Used **5-Fold Cross-Validation** to ensure reliability.
    *   Selected Random Forest based on superior ROC-AUC performance.
5.  **Deployment**: Built a Streamlit interface that accepts raw user input, processes it through the pipeline, and outputs risk assessment.

*(Note: All Python scripts were refactored to be purely procedural without function definitions, as per specific requirements.)*

## 4. Model Results
The final Random Forest model achieved strong performance metrics on the test set:

| Metric | Value |
| :--- | :--- |
| **ROC-AUC** | **~0.84** |
| **Accuracy** | ~79% |
| **Precision** | ~0.65 (High confidence in predicted churners) |
| **Recall** | ~0.50 |

**Key Drivers of Churn:**
*   **Contract Type**: Month-to-month customers are at highest risk.
*   **Tenure**: New customers churn significantly more often.
*   **Total Charges**: High lifetime value customers behave differently.

## 5. How to Run the App

**Prerequisites:**
Ensure you have Python installed. Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

**Run the Application:**
Navigate to the project folder and execute:
```bash
streamlit run app/main.py
```
The application will open in your default web browser (usually at `http://localhost:8501`).
