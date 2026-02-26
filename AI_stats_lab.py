"""
Linear & Logistic Regression Lab
Proper Complete Solution
"""

import numpy as np

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# QUESTION 1 – Linear Regression Pipeline (Diabetes)
# =========================================================

def diabetes_linear_pipeline():

    # STEP 1: Load dataset
    data = load_diabetes()
    X = data.data
    y = data.target

    # STEP 2: Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3: Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # Fit only on train
    X_test_scaled = scaler.transform(X_test)         # Transform test

    # STEP 4: Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # STEP 5: Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # STEP 6: Top 3 features (largest absolute coefficients)
    coef_abs = np.abs(model.coef_)
    top_3_feature_indices = np.argsort(coef_abs)[-3:].tolist()

    # -------------------------------------------------------
    # Overfitting occurs if train R² >> test R².
    # If both are similar → model generalizes well.
    # Feature scaling is important because:
    # - Features may have different magnitudes.
    # - Scaling ensures fair coefficient comparison.
    # - Improves numerical stability.
    # -------------------------------------------------------

    return train_mse, test_mse, train_r2, test_r2, top_3_feature_indices


# =========================================================
# QUESTION 2 – Cross-Validation (Linear Regression)
# =========================================================

def diabetes_cross_validation():

    # Load dataset
    data = load_diabetes()
    X = data.data
    y = data.target

    # Standardize entire dataset for CV
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5-fold CV
    model = LinearRegression()
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')

    mean_r2 = np.mean(scores)
    std_r2 = np.std(scores)

    # -------------------------------------------------------
    # Standard deviation represents variability of performance
    # across folds.
    # Cross-validation reduces variance risk by:
    # - Using multiple train/test splits
    # - Giving more reliable estimate of generalization
    # -------------------------------------------------------

    return mean_r2, std_r2


# =========================================================
# QUESTION 3 – Logistic Regression Pipeline (Cancer)
# =========================================================

def cancer_logistic_pipeline():

    # STEP 1: Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # STEP 2: Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3: Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # STEP 4: Train model
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # STEP 5: Metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    cm = confusion_matrix(y_test, y_test_pred)

    # -------------------------------------------------------
    # False Negative (FN) in medical context:
    # A patient actually has cancer,
    # but the model predicts "no cancer".
    # This is dangerous because:
    # - Disease remains untreated
    # - Condition may worsen
    # -------------------------------------------------------

    return train_accuracy, test_accuracy, precision, recall, f1


# =========================================================
# QUESTION 4 – Logistic Regularization Path
# =========================================================

def cancer_logistic_regularization():

    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    for C_value in [0.01, 0.1, 1, 10, 100]:
        model = LogisticRegression(max_iter=5000, C=C_value)
        model.fit(X_train_scaled, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, model.predict(X_test_scaled))

        results[C_value] = (train_acc, test_acc)

    # -------------------------------------------------------
    # Very small C:
    # - Strong regularization
    # - Simpler model
    # - Risk of underfitting
    # Very large C:
    # - Weak regularization
    # - Complex model
    # - Risk of overfitting
    # Overfitting usually happens when C is very large.
    # -------------------------------------------------------

    return results


# =========================================================
# QUESTION 5 – Cross-Validation (Logistic Regression)
# =========================================================

def cancer_cross_validation():

    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Scale entire dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=1, max_iter=5000)

    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

    mean_accuracy = np.mean(scores)
    std_accuracy = np.std(scores)

    # -------------------------------------------------------
    # Cross-validation is critical in medical diagnosis because:
    # - Medical decisions affect human lives.
    # - We need reliable and stable performance estimates.
    # - It reduces risk of relying on one lucky split.
    # -------------------------------------------------------

    return mean_accuracy, std_accuracy
