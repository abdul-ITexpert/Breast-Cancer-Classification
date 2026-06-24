# Breast Cancer Classification Using Machine Learning

## Project Overview

This project aims to classify breast cancer tumors as **Benign** or **Malignant** using machine learning techniques. The workflow includes data preprocessing, outlier detection, feature selection, feature scaling, model training, and performance evaluation. Multiple classification algorithms were tested and compared to identify the best-performing model.

## Dataset Preprocessing

- Loaded and explored the Breast Cancer dataset.
- Checked for data quality and consistency.
- Applied **Label Encoding** to the target variable.
- Prepared the dataset for machine learning modeling.

## Outlier Detection and Removal

To improve data quality and reduce noise, outliers were identified and removed using:

- **Interquartile Range (IQR) Method**
- **Z-Score Method**

This preprocessing step helped improve the reliability of model predictions.

## Initial Model Performance

The following machine learning models were trained and evaluated on the cleaned dataset:

| Model | Accuracy |
|---------|---------|
| Logistic Regression | 96% |
| K-Nearest Neighbors (KNN) | 95% |
| Support Vector Machine (SVM) | 96% |
| Decision Tree | 95% |

## Data Visualization

- Generated **Regression Plots (Regplots)** for all numerical features to analyze feature relationships and trends.
- Visualized patterns within the dataset to better understand feature behavior.

## Feature Selection

Feature selection techniques were applied to identify the most relevant features and reduce dimensionality.

After feature selection:

| Model | Accuracy |
|---------|---------|
| Logistic Regression | 96% |
| K-Nearest Neighbors (KNN) | 95% |

The results remained consistent while using a reduced set of important features.

## Feature Scaling

After feature selection, **StandardScaler** was applied to normalize feature values.

Model performance after scaling:

| Model | Accuracy |
|---------|---------|
| Logistic Regression | 98% |
| K-Nearest Neighbors (KNN) | 96% |

Feature scaling improved the overall performance, with Logistic Regression achieving the highest accuracy.

## Model Evaluation

Models were evaluated using:

- Accuracy Score
- Confusion Matrix
- Heatmap Visualization

A confusion matrix heatmap was created to provide a clear visual representation of classification performance and prediction accuracy.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Results

The best-performing model was **Logistic Regression with StandardScaler**, achieving an accuracy of **98%** after feature selection and feature scaling.

## Key Learning Outcomes

- Data Cleaning and Preprocessing
- Outlier Detection using IQR and Z-Score
- Feature Selection Techniques
- Feature Scaling with StandardScaler
- Machine Learning Model Comparison
- Performance Evaluation using Confusion Matrix and Heatmaps
- Data Visualization with Seaborn and Matplotlib

## Future Improvements

- Hyperparameter Tuning using GridSearchCV
- Cross-Validation
- Ensemble Learning Methods (Random Forest, XGBoost)
- Model Deployment using Flask or Streamlit
- Feature Importance Analysis
