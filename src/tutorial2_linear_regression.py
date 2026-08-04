"""
=============================================================================
TUTORIAL 2 (T2): LINEAR REGRESSION IN PYTHON
Project: Precision AgTech Crop & Climate Prediction
=============================================================================
This program demonstrates Linear Regression algorithms in Python:
1. Simple & Multiple Linear Regression
2. Ridge Regression (L2 Regularization)
3. Lasso Regression (L1 Regularization)
4. Model Evaluation Metrics: MSE, RMSE, MAE, R-squared (R2 Score)
=============================================================================
"""

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib


def load_regression_data(file_path: str):
    """
    Load dataset and prepare features and continuous target.
    For Linear Regression (predicting continuous numerical value),
    we predict 'rainfall' (mm) from soil & atmospheric variables:
    N, P, K, temperature, humidity, ph.
    """
    print("=" * 70)
    print("STEP 1: PREPARING REGRESSION DATASET")
    print("=" * 70)
    
    df = pd.read_csv(file_path)
    print(f"[OK] Loaded dataset shape: {df.shape}")
    
    # Feature matrix X (soil & climate metrics) and target vector y (rainfall in mm)
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph']
    target_col = 'rainfall'
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    print(f"[*] Input Features (X): {feature_cols}")
    print(f"[*] Target Variable (y): '{target_col}' (Continuous Numerical Variable in mm)")
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    # Scale features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler


def train_evaluate_linear_regression(X_train, X_test, y_train, y_test, feature_cols):
    """
    Train Ordinary Least Squares (OLS) Linear Regression, Ridge, and Lasso.
    Evaluate performance using MSE, RMSE, MAE, and R2 Score.
    """
    print("\n" + "=" * 70)
    print("STEP 2: TRAINING LINEAR REGRESSION MODELS")
    print("=" * 70)
    
    models = {
        "Multiple Linear Regression (OLS)": LinearRegression(),
        "Ridge Regression (Alpha=1.0)": Ridge(alpha=1.0),
        "Lasso Regression (Alpha=0.1)": Lasso(alpha=0.1)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2_Score": r2
        }
        
        print(f"\n--- {name} ---")
        print(f"   Mean Squared Error (MSE):       {mse:10.4f}")
        print(f"   Root Mean Squared Error (RMSE): {rmse:10.4f} mm")
        print(f"   Mean Absolute Error (MAE):      {mae:10.4f} mm")
        print(f"   R-squared (R2 Score):          {r2:10.4f}")
        
        # Display Feature Coefficients
        print("   Feature Coefficients (Slopes / Weights):")
        for col, coef in zip(feature_cols, model.coef_):
            print(f"     - {col:12s}: {coef:8.4f}")
        print(f"     - Intercept (b): {model.intercept_:8.4f}")
    
    return results


def main():
    print("\n--- AGTECH PRECISION AGRICULTURE - TUTORIAL 2: LINEAR REGRESSION ---\n")
    dataset_path = os.path.join("data", "Crop_recommendation.csv")
    
    X_train, X_test, y_train, y_test, feature_cols, scaler = load_regression_data(dataset_path)
    results = train_evaluate_linear_regression(X_train, X_test, y_train, y_test, feature_cols)
    
    # Save best regression model
    os.makedirs("models", exist_ok=True)
    joblib.dump(results["Multiple Linear Regression (OLS)"]["model"], "models/linear_regression_model.pkl")
    print("\n[SAVE] Saved linear regression model to: models/linear_regression_model.pkl")
    
    print("\n" + "=" * 70)
    print(">>> TUTORIAL 2 (LINEAR REGRESSION) COMPLETED SUCCESSFULLY! <<<")
    print("=" * 70)


if __name__ == "__main__":
    main()
