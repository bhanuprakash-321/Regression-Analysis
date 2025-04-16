# Medical Insurance Charges Prediction

## Overview

This project focuses on predicting medical insurance charges based on key demographic and health-related factors. By leveraging machine learning techniques, the goal is to assist insurance companies in making data-driven pricing decisions and identifying factors that contribute to higher insurance costs.

---

## Business Objective

In the highly competitive insurance industry, accurately predicting medical charges is critical for pricing insurance premiums and assessing risk. This project aims to:

- Predict the **medical insurance charges** for individuals based on attributes such as age, BMI, smoking status, and region.
- **Identify key risk factors** driving higher premiums, such as smoking, obesity, and age.
- Provide insurance companies with insights that can help in **strategic decision-making** regarding customer pricing, claims forecasting, and personalized policy offerings.

The project is intended to streamline pricing models and enhance risk management strategies for insurance companies.

---

## Dataset Overview

The dataset used for this project contains the following features:

- **Source**: [Kaggle - Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Target Variable**: `charges` (Medical expenses)
- **Total Records**: 1,300+
- **Features**:
  - **Age**: Age of the individual
  - **Sex**: Gender of the individual
  - **BMI**: Body Mass Index
  - **Children**: Number of children/dependents covered
  - **Smoker**: Whether the individual smokes (yes/no)
  - **Region**: Geographical region of the individual
  - **Charges**: Medical insurance charges (target variable)

---

## Key Insights & EDA

- **Smokers** were found to have significantly higher medical charges than non-smokers.
- **BMI** was identified as another strong predictor of higher insurance costs.
- Exploratory Data Analysis (EDA) revealed a significant correlation between **smoking** status and **medical charges**.
- Outliers were identified in `charges` and `BMI`, which were kept in the analysis due to their relevance in real-world scenarios.

---

## Preprocessing & Techniques

- **One-Hot Encoding** was applied to handle categorical variables (`sex`, `smoker`, and `region`).
- The dataset was **split into training and testing sets** to evaluate model performance.

---

## Models & Performance

Various regression models were applied, and their performances were compared. The key findings are:

| Model                | R² Score | Mean Absolute Error (MAE) |
|----------------------|----------|---------------------------|
| **Random Forest**     | **0.87** | **2797**                  |
| **XGBoost**           | 0.85     | 3117                      |
| **Linear Regression** | 0.81     | 4171                      |
| **Decision Tree**     | 0.74     | 3579                      |
| **K-Nearest Neighbors**| 0.13    | 8390                      |
| **Support Vector**    | -0.13    | 9253                      |

### Key Insights:
- **Random Forest Regressor** was the best performing model, achieving an **R² score of 87%** and an **MAE of 2797**.
- **Support Vector Machine** performed poorly, with an **R² score of -0.13**, indicating that this model was not suitable for this dataset.
- **Linear Regression** performed fairly well with an **R² score of 81%**, but was outperformed by Random Forest and XGBoost.
- **K-Nearest Neighbors** also showed poor performance with a **R² score of 0.13**, indicating that it wasn't well-suited for this dataset.

---

## Visualizations

Several insightful visualizations were created to understand the relationships between features and predictions:

- **Correlation Heatmap**: Analyzing the correlation between features.
- **BMI vs Charges**: Scatter plot showing the relationship between BMI and charges, with differentiation based on smoking status.
- **Actual vs Predicted**: Scatter plot comparing true values vs predicted values, highlighting model accuracy.

---

## Conclusion

- **Random Forest Regressor** achieved the highest performance with an **R² score of 87%** and an **MAE of 2797**, making it the model of choice for medical charges prediction.
- **Smoking** and **BMI** were identified as the most influential features driving higher medical charges.
- **Support Vector Machine** performed poorly, with an **R² score of -0.13**, and should likely be avoided for this type of regression problem.
- **Linear Regression** and **XGBoost** also performed well, with **R² scores of 81% and 85%**, respectively, offering promising alternatives for future optimization.

---

## Future Work

- **Hyperparameter Tuning**: Implement **GridSearchCV** or **RandomizedSearchCV** for further optimization of the Random Forest and XGBoost models.
- **Model Deployment**: Build an interactive web application using **Streamlit** or **Flask** for real-time predictions.
- **Model Interpretability**: Use **SHAP** or **LIME** for enhanced model explainability and transparency.

---


