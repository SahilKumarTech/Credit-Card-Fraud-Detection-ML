#  Credit Card Fraud Detection - Machine Learning Project

##  Project Overview
This project focuses on detecting fraudulent credit card transactions using machine learning. The dataset contains over 1.2 million transactions with only 0.57% fraud cases, presenting a classic imbalanced classification problem.

##  Project Goals
- Perform comprehensive Exploratory Data Analysis (EDA)
- Handle severe class imbalance (0.57% fraud rate)
- Build and evaluate multiple machine learning models
- Compare model performance on highly imbalanced data
- Deploy the best-performing model for fraud detection

##  Dataset Information
- **Total Transactions**: 1,273,687
- **Fraudulent Transactions**: 7,325 (0.57%)
- **Legitimate Transactions**: 1,266,360 (99.43%)
- **Features**: 23 original features, reduced to 14 after preprocessing
- **Time Period**: 2019 transactions

##  Technical Stack
- Python 3.x
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Imbalanced-learn (for handling class imbalance)
- XGBoost/LightGBM

##  Key Steps

### 1. Exploratory Data Analysis (Completed)
- Data quality assessment and cleaning
- Feature engineering (time-based features)
- Visualization of fraud patterns
- Correlation analysis
- Handling missing values and duplicates

### 2. Data Preprocessing
- Encoding categorical variables
- Feature scaling and normalization
- Time-based feature extraction
- Geolocation feature engineering

### 3. Model Development (Next Steps)
- Implement resampling techniques (SMOTE, ADASYN)
- Train multiple classifiers:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Neural Networks
- Handle class imbalance using appropriate metrics

### 4. Model Evaluation
- Primary Metrics: Precision, Recall, F1-Score, AUC-ROC
- Confusion matrix analysis
- Feature importance analysis
- Cross-validation results

# Credit Card Fraud Detection using Machine Learning

A comprehensive machine learning project for detecting fraudulent credit card transactions. This project implements multiple classification algorithms to identify fraud patterns in highly imbalanced financial data.

##  Highlights
- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, SVM
- **Imbalanced Data Handling**: SMOTE oversampling technique
- **Comprehensive Evaluation**: Accuracy, F1 Score, Confusion Matrix
- **Best Model**: XGBoost with 86.44% F1 Score

##  Key Features
- Data preprocessing and feature scaling
- Class imbalance handling with SMOTE
- Model training and performance comparison
- Detailed evaluation metrics
- Visualization of results

## 🛠️ Technologies Used
- Python, Pandas, NumPy
- Scikit-learn, XGBoost, Imbalanced-learn
- Matplotlib, Seaborn
- Jupyter Notebook

##  Results
XGBoost emerged as the best model with:
- **Accuracy**: 99.87%
- **F1 Score**: 86.44%

