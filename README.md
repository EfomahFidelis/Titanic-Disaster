# Titanic - Machine Learning from Disaster
This project aims to build a predictive model to answer the question: **“What sorts of people were more likely to survive the sinking of the Titanic?”** The Titanic is one of the most infamous shipwrecks in history, and this project leverages machine learning techniques to analyze survival factors.
<br>

## **Introduction:**
This project employs advanced machine learning techniques to predict the survival likelihood of Titanic passengers based on demographic, social, and economic features. Using a dataset containing passenger information such as age, gender, ticket class, and more, the project explores relationships between features and survival rates, applies feature engineering, and builds models using **XGBoost** and **LightGBM** to produce predictions. 

The primary objective is to identify key survival factors while demonstrating practical experience in feature engineering, data preprocessing, hyperparameter tuning, and model evaluation.

---

## **Methodology:**
### **1. Exploratory Data Analysis (EDA):**
- Visualized survival distributions and key variables such as gender, age, and class to uncover patterns.
- Correlation analysis was performed to evaluate the relationships between features and survival outcomes.

### **2. Feature Engineering:**
- Derived meaningful features, including:
  - **Family size**: Total number of family members onboard.
  - **Isolation status**: Whether a passenger traveled alone.
  - **Title extraction**: Extracted social titles (e.g., Mr., Mrs., Miss) from passenger names to capture social standing.
  - **Cabin presence**: Indicated whether cabin information was available for a passenger.
- Imputed missing data for variables such as age, fare, and embarkation location.

### **3. Data Preprocessing:**
- **No one-hot encoding was applied**; instead, categorical variables (e.g., gender, embarkation location, and titles) were transformed into numeric representations using label encoding.
- Ensured consistency of feature sets between training and test datasets by dropping unnecessary columns such as `Ticket` and `Cabin`.

### **4. Model Training:**
- Two gradient-boosting algorithms were employed:
  - **XGBoost (Extreme Gradient Boosting):** A robust, efficient tree-based model.
  - **LightGBM (Light Gradient Boosting Machine):** Known for its speed and efficiency on large datasets.
- Both models were trained using hyperparameter tuning via `RandomizedSearchCV` to optimize their performance.

### **5. Evaluation:**
- Model performance was compared based on accuracy on the validation set.
- Feature importance was analyzed for both models to identify key predictors of survival.
- Visualizations were generated to highlight correlations and feature importance.

---

## **Technology:**
- **Programming Languages:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, LightGBM

---

## Model Training and Evaluation

### Hyperparameter Tuning
- **XGBoost:**
  - Best Parameters: 
    ```json
    {
      "subsample": 1.0,
      "reg_lambda": 1,
      "reg_alpha": 0,
      "n_estimators": 500,
      "max_depth": 3,
      "learning_rate": 0.2,
      "gamma": 1,
      "colsample_bytree": 1.0
    }
    ```
  - Cross-validation score: **0.823**

- **LightGBM:**
  - Best Parameters:
    ```json
    {
      "subsample": 1.0,
      "reg_lambda": 1.5,
      "reg_alpha": 1,
      "num_leaves": 31,
      "n_estimators": 100,
      "min_child_samples": 20,
      "max_depth": 3,
      "learning_rate": 0.05,
      "colsample_bytree": 0.8
    }
    ```
  - Cross-validation score: **0.824**

### Model Performance
- **XGBoost Validation Accuracy:** **0.80**
- **LightGBM Validation Accuracy:** **0.82**


## **Results:**
1. **Model Performance:**
   - Both XGBoost and LightGBM demonstrated strong predictive capabilities. A direct comparison of their accuracy scores was visualized to determine the best-performing model.

    - I. **Best Performing Model:** LightGBM achieved a slightly higher cross-validation score (**0.824**) and validation accuracy (**0.82**) compared to XGBoost.
    - II. **Feature Importance Visualization:** The most important features influencing survival were identified and visualized.
    - III. **Predictions and Submission:**
                - XGBoost predictions saved to: `submission_xgb.csv`.
                - LightGBM predictions saved to: `submission_lgbm.csv`.
   
2. **Feature Importance:**
   - Gender emerged as the most influential predictor, with females showing a significantly higher likelihood of survival.
   - Other key features included **fare**, **class (ticket category)**, and **family size**.

3. **Insights:**
   - Social class and gender played pivotal roles in determining survival probabilities, reflecting historical patterns from the disaster.
   - Practical knowledge of data preprocessing, feature engineering, and implementing advanced machine learning models was showcased.

---

## **Key Highlights:**
- Leveraged the strengths of **XGBoost** and **LightGBM**, demonstrating the effectiveness of gradient boosting algorithms in predictive modeling.
- Refrained from using one-hot encoding, opting for more efficient label encoding techniques to handle categorical variables.
- Advanced hyperparameter tuning and evaluation methodologies ensured optimal model performance.

---

### **Visualizations:**
- **Feature Correlations with Survival:** A bar chart and heatmap revealed the strongest correlations between features and survival probabilities.
- **Feature Importance Analysis:** Visualized the relative importance of features based on trained models.
- **Model Comparison:** A bar chart was used to compare the validation accuracy of XGBoost and LightGBM models.

This project is a testament to the power of machine learning for uncovering insights from historical data while refining technical skills in data analysis and modeling.

---

## Conclusion
Both XGBoost and LightGBM models performed well in predicting Titanic survival, with LightGBM slightly outperforming XGBoost. The results demonstrate the significance of features such as `Fare` and `Pclass` in determining survival probabilities.
