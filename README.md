# 🚗 Auto MPG Regression – Machine Learning Project

This project focuses on predicting the **fuel efficiency (MPG - Miles Per Gallon)** of automobiles using their technical specifications. It applies various data preprocessing techniques, regression algorithms, and ensemble methods to develop accurate and robust predictive models.

---

## 📂 Dataset

- **Source:** [UCI Machine Learning Repository – Auto MPG Dataset](https://archive.ics.uci.edu/ml/datasets/auto+mpg)
- **Total Instances:** 398
- **Features Used:**
  - Cylinders
  - Displacement
  - Horsepower
  - Weight
  - Acceleration
  - Model Year
  - Origin
- **Target Variable:** MPG (Miles Per Gallon)

---

## 📊 Workflow Overview

1. **Data Cleaning**
   - Handled 12 missing values in the `Horsepower` column using group-wise median imputation (based on Cylinder count).
2. **Exploratory Data Analysis (EDA)**
   - Correlation matrix, clustermap, skewness analysis, and distribution plots.
3. **Outlier Removal**
   - Used IQR method on `Horsepower` and `Acceleration`.
4. **Feature Engineering**
   - Log transformation applied to target (`MPG`) due to skewness.
   - One-hot encoding on `Cylinders` and `Origin`.
5. **Data Splitting & Scaling**
   - 10% training / 90% testing split to challenge the models.
   - Applied `RobustScaler` to minimize the influence of outliers.
6. **Modeling**
   - Applied and tuned:
     - Linear Regression
     - Ridge Regression (L2)
     - Lasso Regression (L1)
     - ElasticNet (L1 + L2)
     - XGBoost Regressor
     - Averaging Ensemble (Lasso + XGBoost)
7. **Performance Evaluation**
   - Mean Squared Error (MSE) used as the evaluation metric.

---

## 🧠 Model Comparison (MSE on Test Set)

| Model                  | MSE         | Notes                                       |
|------------------------|-------------|---------------------------------------------|
| Linear Regression      | 0.01363     | Baseline model                              |
| Ridge Regression       | 0.01340     | Slightly improved with L2 regularization    |
| Lasso Regression       | 0.01331 ✅  | Feature selection + low error               |
| ElasticNet             | 0.01330 ✅✅ | Best performance, balanced model            |
| XGBoost                | 0.01810 ❌   | Overcomplicated for this small dataset      |
| Averaged (Lasso + XGB) | 0.01365     | Balanced but not outperforming Lasso/ENet   |

---

## 🧪 Key Insights

- **ElasticNet and Lasso** were the most effective models for this dataset.
- **RobustScaler** proved useful in stabilizing model performance.
- **XGBoost**, while powerful, underperformed due to dataset simplicity.
- **Averaging models** yielded consistent but not superior results.

---

## 📦 Technologies Used

- Python 3.10+
- NumPy, Pandas
- Seaborn, Matplotlib
- Scikit-learn
- XGBoost
- Jupyter Notebook / VSCode

---

## 📁 Project Structure

