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

## 5. Streamlit Dashboard Features
The interactive web application (`app/main.py`) has been upgraded to a **Business-Class Dashboard**:

1.  **KPI Cards**: Real-time overview of Churn Risk, Probability %, and Tenure.
2.  **Risk Meter**: Visual progress bar with strict color-coded alerts (Red/Yellow/Green).
3.  **Business Actions**: Context-aware recommendations (e.g., "Retention Protocol" vs "Engagement").
4.  **What-If Analysis**: Simulate strategy changes (e.g., switching Contract type) to see immediate risk impact.
5.  **Explainability**: Visualizing the top 10 factors driving the model's decision.
6.  **Batch Prediction**: Upload a CSV to generate churn predictions for hundreds of customers at once. (Use `sample_upload.csv` to test!).

### 💎 Business Value
*   **Proactive Retention**: Identifies at-risk customers *before* they leave, allowing for targeted intervention.
*   **Data-Driven Decisions**: Replaces intuition with probability-based risk assessments.
*   **Revenue Protection**: Focuses resources on high-value customers with high churn risk.
*   **Efficiency**: Automates the analysis of thousands of customer profiles in seconds.

## 6. Dashboard Screenshots
*(Add your screenshots here)*
*   **Main Dashboard**: [placeholder for dashboard.png]
*   **Batch Portal**: [placeholder for batch.png]

**Run the Application:**
Navigate to the project folder and execute:
```bash
streamlit run app/main.py
```
The application will open in your default web browser (usually at `http://localhost:8501`).
