# Customer Churn Prediction with Random Forest

A machine learning project predicting customer churn using a Random Forest classifier with hyperparameter tuning.

## Overview

This project analyzes customer data to identify patterns that indicate whether a customer is likely to churn (leave the service). Using ensemble methods and cross-validation, the model achieves strong predictive performance to help businesses proactively retain at-risk customers.

## Dataset

[Telco Customer Churn Dataset](https://www.kaggle.com/code/amycorona/predicting-churn-with-a-random-forest-classifier) from Kaggle, containing customer information including:
- Demographics (gender, senior citizen status, dependents)
- Account information (tenure, contract type, payment method)
- Service subscriptions (phone, internet, streaming services)
- Billing details (monthly charges, total charges)

## Methodology

1. **Data Exploration & Cleaning**: Analyzed feature distributions, handled missing values, and encoded categorical variables
2. **Train-Test Split**: Divided data into training and testing sets for model validation
3. **Model Training**: Built Random Forest classifier with ensemble learning
4. **Hyperparameter Tuning**: Used GridSearchCV with 5-fold cross-validation to optimize forest parameters including:
   - Number of estimators
   - Maximum tree depth
   - Minimum samples for splitting
   - Split criterion (Gini vs. Entropy)
5. **Model Evaluation**: Assessed performance using multiple metrics

## Results

- **AUC Score**: 0.99 - indicating excellent discrimination between churners and non-churners
- Successfully identified optimal Random Forest configuration through systematic parameter search

## Technologies Used

- **Python 3.x**
- **pandas** - data manipulation and analysis
- **scikit-learn** - machine learning models and evaluation
- **matplotlib/seaborn** - data visualization
- **Jupyter Notebook** - interactive development environment

## Key Skills Demonstrated

- Binary classification modeling
- Ensemble learning methods (Random Forest)
- Cross-validation techniques
- Hyperparameter optimization with GridSearchCV
- Model evaluation and interpretation

## Usage
```bash
# Clone the repository
git clone [your-repo-url]

# Open the notebook
jupyter notebook predicting_churn.ipynb
```

## Future Enhancements

- Feature importance analysis to identify key churn drivers
- Compare performance with other algorithms (XGBoost, Gradient Boosting)
- Implement cost-sensitive learning to account for business costs of false positives/negatives
- Deploy model as a web application for real-time churn prediction
