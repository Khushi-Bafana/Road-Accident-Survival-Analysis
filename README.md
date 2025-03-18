# Road Accident Survival Analysis
![Survival Probability Across Ages and Speeds](https://github.com/Khushi-Bafana/Road-Accident-Survival-Analysis/blob/main/Survival%20Probability%20Across%20Ages%20and%20Speeds.png)
## Objective
The objective of this project is to analyze various factors influencing survival in road accidents and develop a predictive model that estimates the probability of survival. By leveraging demographic, behavioral, and situational data, the analysis aims to identify key contributors to survival rates and provide insights for improving road safety measures.

## Purpose
- To understand the impact of various factors (age, gender, speed of impact, safety device usage, road conditions, weather, etc.) on survival outcomes.
- To develop a machine learning model that predicts survival probability based on accident-related data.
- To generate actionable insights that can help policymakers, emergency responders, and vehicle manufacturers enhance road safety strategies.

## Project Steps

### 1. Data Collection
- The dataset contains information on various factors affecting survival in road accidents, such as:
  - Demographics: Age, gender, health conditions
  - Vehicle-related: Type of vehicle, airbag deployment, seatbelt usage
  - Environmental: Weather conditions, road type, lighting conditions
  - Accident-specific: Speed of impact, collision type, presence of alcohol or drugs
- Data was sourced from publicly available road accident databases and cleaned for consistency.

### 2. Data Preprocessing
- Checked for missing values and handled them appropriately using imputation techniques.
- Converted categorical variables into numerical representations using one-hot encoding and label encoding.
- Performed feature scaling (Standardization/Normalization) where necessary to normalize the dataset.
- Removed duplicates and handled outliers using statistical methods (IQR, Z-score analysis).
- Performed feature engineering by creating new meaningful features, such as risk score and impact severity index.

### 3. Exploratory Data Analysis (EDA)
- Visualized data distributions using histograms, density plots, and box plots to understand feature behavior.
- Used correlation heatmaps to identify relationships between different variables.
- Conducted bivariate and multivariate analysis to assess interactions between variables.
- Performed statistical hypothesis testing to validate the significance of specific features.
- Identified key factors affecting survival probabilities and their impact through graphical representations.

### 4. Model Building
- Split the dataset into training (80%) and testing (20%) sets.
- Applied multiple machine learning models for binary classification:
  - **Logistic Regression**: Baseline model for comparison.
  - **Decision Trees**: To capture non-linear relationships.
  - **Random Forest**: To improve generalization and reduce overfitting.
  - **Gradient Boosting (XGBoost, LightGBM)**: For enhanced predictive performance.
  - **Neural Networks**: For deep learning-based predictions.
- Used GridSearchCV and RandomizedSearchCV for hyperparameter tuning to optimize model performance.

### 5. Model Evaluation
- Evaluated model performance using various metrics:
  - **Accuracy**: Measures overall correctness of predictions.
  - **Precision, Recall, and F1-score**: Assesses the balance between false positives and false negatives.
  - **ROC-AUC Score**: Evaluates model discrimination between survival and non-survival cases.
  - **Confusion Matrix**: Provides a breakdown of prediction results.
  - **Feature Importance Analysis**: Determines which factors contribute most to survival probability.
- Compared different models and selected the best-performing one based on evaluation metrics.

### 6. Insights and Interpretations
- Identified the most influential features impacting survival outcomes, such as:
  - Speed of impact
  - Use of safety devices (seatbelts, airbags)
  - Alcohol/drug involvement
  - Weather and road conditions
- Provided recommendations based on findings to improve road safety, such as:
  - Increasing awareness about seatbelt usage.
  - Enhancing road lighting in accident-prone areas.
  - Enforcing stricter speed limits in high-risk zones.
- Visualized important factors affecting survival using SHAP values, feature importance plots, and decision boundaries.

## Conclusion
This project successfully analyzed accident survival factors and developed a predictive model with meaningful insights. The findings provide actionable recommendations for improving road safety. Future enhancements can include integrating real-time accident data, using advanced deep learning models, and refining feature selection for better accuracy.

## Future Work
- **Expand Dataset**: Incorporate additional data sources for improved generalization.
- **Real-time Prediction**: Develop a mobile or web application for real-time survival prediction.
- **Explainable AI**: Utilize interpretability techniques to enhance trust in predictions.
- **Integration with Emergency Services**: Provide real-time alerts and predictions to assist emergency responders.

