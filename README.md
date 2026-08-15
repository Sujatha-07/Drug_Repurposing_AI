# Drug Repurposing AI

## 1. Project Overview

Drug Repurposing AI is a machine-learning based application that predicts potential repurposed uses of existing drugs.

The system accepts a drug name as input and displays:

- Drug Name
- Current Use
- Repurposed Use
- Predicted Category
- Prediction Confidence

The project also compares the performance of four machine-learning algorithms:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

---

## 2. Project Objective

The main objective of the project is to use existing drug-related information and machine-learning techniques to identify possible alternative therapeutic uses of drugs.

The system provides a simple web interface where a user can enter a drug name and receive the prediction results.

---

## 3. Technologies Used

### Frontend

- React
- Vite
- JavaScript
- CSS

### Backend

- Python
- FastAPI
- Uvicorn

### Machine Learning

- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Joblib

### Dataset

- Excel dataset containing drug-related properties, current use, targets, drug class, and repurposed use information.

---

## 4. Machine Learning Algorithms

### Logistic Regression

Used for multiclass classification of drug repurposing categories.

### Decision Tree

Uses feature-based decision rules to classify drugs into repurposing categories.

### Random Forest

Uses an ensemble of multiple decision trees to improve classification performance.

### XGBoost

Uses gradient-boosted decision trees for classification and is used as the prediction model integrated with the backend.

---

## 5. Model Comparison

The current model comparison used in the application is:

| Algorithm | Accuracy |
|---|---:|
| Logistic Regression | 55.44% |
| Decision Tree | 49.74% |
| Random Forest | 53.37% |
| XGBoost | 57.51% |

### Best Performing Model

**XGBoost — 57.51% accuracy**

The application displays this comparison below the individual drug prediction.

> Note: Model accuracy and prediction confidence are different metrics. Accuracy represents model performance on the evaluation set, while confidence represents the probability associated with an individual prediction.

---

## 6. Prediction Output

The user enters a drug name.

Example:

```text
Input:
Abacavir