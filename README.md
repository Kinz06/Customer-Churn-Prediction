# Customer Churn Prediction

## Overview

This project predicts whether a customer will churn using Machine Learning models and a Rule-Based system. Two algorithms are trained and compared, and a simple rule engine is implemented using domain knowledge derived from exploratory data analysis.

## Dataset

File: customer_churn_dataset.csv
Rows: 64,374
Columns: 12
Churn Rate: 47.4%

Source: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset

| Column | Type | Description |
|--------|------|-------------|
| CustomerID | ID | Dropped during preprocessing |
| Age | Numeric | Customer age in years |
| Gender | Categorical | Female or Male |
| Tenure | Numeric | Number of months as a customer |
| Usage Frequency | Numeric | Product usage count |
| Support Calls | Numeric | Number of support calls made |
| Payment Delay | Numeric | Days payment was delayed |
| Subscription Type | Categorical | Basic, Standard, or Premium |
| Contract Length | Categorical | Monthly, Quarterly, or Annual |
| Total Spend | Numeric | Total amount spent by the customer |
| Last Interaction | Numeric | Days since last interaction |
| Churn | Target | 1 = Churned, 0 = Stayed |

## Project Files

| File | Description |
|------|-------------|
| churn_prediction.ipynb | Main Jupyter Notebook containing all 8 tasks |
| customer_churn_dataset.csv | Dataset used for training and evaluation |
| model_comparison_summary.txt | Comparison of model results and metrics |
| README.md | Project documentation |
| eda_plots.png | Exploratory data analysis visualizations |
| model_comparison.png | Metrics bar chart and confusion matrices |
| roc_curves.png | ROC curve comparison for both models |
| feature_importance.png | Random Forest feature importance chart |

## How to Run

Install the required libraries:

    pip install pandas numpy scikit-learn matplotlib seaborn jupyter

Launch the notebook:

    jupyter notebook churn_prediction.ipynb

The dataset file must be in the same folder as the notebook.

## Preprocessing Steps

- Dropped CustomerID column
- Checked for and confirmed no missing values
- Checked for and confirmed no duplicate rows
- Encoded Gender as binary: Female = 0, Male = 1
- Encoded Subscription Type as ordinal: Basic = 1, Standard = 2, Premium = 3
- Encoded Contract Length as ordinal: Monthly = 1, Quarterly = 2, Annual = 3
- Applied StandardScaler to all numeric features
- Split data into 80% training and 20% testing with stratification

## Models

Logistic Regression
A linear classifier that estimates the probability of churn using a logistic function. It is fast, interpretable, and serves as the baseline model.

Random Forest
An ensemble of 100 decision trees. Each tree votes on the prediction, and the majority result is taken. It captures non-linear relationships between features and produces feature importance scores.

## Results

| Model | Accuracy | ROC-AUC | F1-Score | Precision | Recall |
|-------|----------|---------|---------|-----------|--------|
| Logistic Regression | 82.55% | 0.9023 | 0.8171 | 0.8114 | 0.8228 |
| Random Forest | 99.84% | 1.0000 | 0.9983 | 0.9992 | 0.9974 |

## Rule-Based Logic

Nine rules are applied to each customer. Each matching condition adds or subtracts a risk score. If the total score is 5 or above, the customer is predicted to churn.

| Rule | Condition | Score |
|------|-----------|-------|
| R1 | Support Calls >= 5 | +4 |
| R2 | Payment Delay >= 20 days | +3 |
| R3 | Contract Length is Monthly | +3 |
| R4 | Subscription Type is Basic | +2 |
| R5 | Tenure <= 6 months | +2 |
| R6 | Usage Frequency <= 5 | +2 |
| R7 | Contract Length is Annual | -3 |
| R8 | Subscription Type is Premium | -2 |
| R9 | Support Calls = 0 and Tenure > 24 months | -2 |

## Best Model

Random Forest is the better model for this dataset.

- Accuracy is 17 percentage points higher than Logistic Regression
- ROC-AUC of 1.0000 versus 0.9023
- F1-Score of 0.9983 versus 0.8171
- Captures non-linear interactions between features such as Support Calls, Payment Delay, and Contract Length combined
- Ensemble of 100 trees reduces variance and avoids overfitting
- Provides feature importance scores that confirm Support Calls and Payment Delay are the strongest churn predictors

Logistic Regression is preferred when a fully explainable per-prediction output is required, such as for compliance or audit purposes.

The rule-based system is useful for real-time alerting without any model inference, such as triggering an action when a customer misses a payment or downgrades their subscription.

---
