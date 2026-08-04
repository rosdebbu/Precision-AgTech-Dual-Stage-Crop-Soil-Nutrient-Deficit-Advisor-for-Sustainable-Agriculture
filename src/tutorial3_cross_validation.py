"""
=============================================================================
TUTORIAL 3 (T3): CROSS VALIDATION IN PYTHON
Project: Precision AgTech Model Evaluation & Validation
=============================================================================
This program demonstrates Cross-Validation techniques in Python:
1. K-Fold Cross Validation (K=5, K=10)
2. Stratified K-Fold Cross Validation (preserves class distributions)
3. Model evaluation and stability comparison across folds (Mean & Std Dev)
4. Evaluation of multiple classification algorithms (Random Forest, Decision Tree,
   Support Vector Machine, K-Nearest Neighbors, Naive Bayes, Logistic Regression)
=============================================================================
"""

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import (
    cross_val_score,
    cross_validate,
    KFold,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


def load_and_preprocess(file_path: str):
    """Load dataset, scale features, and encode targets."""
    df = pd.read_csv(file_path)
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X = df[feature_cols].copy()
    y_raw = df['label'].copy()
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, feature_cols, label_encoder.classes_


def run_kfold_cross_validation(X, y, num_folds=5):
    """Demonstrate Standard K-Fold Cross Validation."""
    print("=" * 70)
    print(f"STEP 1: K-FOLD CROSS VALIDATION (K={num_folds} Folds)")
    print("=" * 70)
    print(f"Why Cross Validation?")
    print(" - A single train/test split might give an unusually lucky or unlucky accuracy.")
    print(f" - K-Fold divides dataset into {num_folds} equal parts (folds).")
    print(f" - It trains on {num_folds-1} folds and tests on 1 fold, repeating {num_folds} times.")
    print(" - This guarantees every sample is tested exactly once!")
    
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    scores = cross_val_score(rf_model, X, y, cv=kf, scoring='accuracy')
    
    print(f"\n--- Random Forest Classifier ({num_folds}-Fold Scores) ---")
    for fold_idx, score in enumerate(scores, 1):
        print(f"   Fold {fold_idx}: Accuracy = {score * 100:.2f}%")
    
    mean_acc = scores.mean() * 100
    std_acc = scores.std() * 100
    print(f"\n[*] Overall Mean Accuracy: {mean_acc:.2f}% (+/- {std_acc:.2f}%)")
    return scores


def run_stratified_kfold_cross_validation(X, y, num_folds=5):
    """Demonstrate Stratified K-Fold Cross Validation across multiple algorithms."""
    print("\n" + "=" * 70)
    print(f"STEP 2: STRATIFIED K-FOLD CROSS VALIDATION & MULTI-MODEL BENCHMARK")
    print("=" * 70)
    print("Why Stratified K-Fold?")
    print(" - Ensures that each of the K folds has the exact same proportion of crop classes.")
    
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Support Vector Classifier (SVM)": SVC(kernel='rbf', C=1.0, random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Gaussian Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    print(f"\n{'Algorithm':32s} | {'Mean Accuracy':15s} | {'Std Dev':10s} | {'Min Fold':10s} | {'Max Fold':10s}")
    print("-" * 88)
    
    for name, clf in classifiers.items():
        cv_results = cross_validate(
            clf, X, y, cv=skf, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        )
        
        acc_scores = cv_results['test_accuracy'] * 100
        mean_acc = acc_scores.mean()
        std_acc = acc_scores.std()
        min_acc = acc_scores.min()
        max_acc = acc_scores.max()
        
        results[name] = {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "scores": acc_scores,
            "f1_score": cv_results['test_f1_macro'].mean() * 100
        }
        
        print(f"{name:32s} | {mean_acc:13.2f}% | {std_acc:8.2f}% | {min_acc:8.2f}% | {max_acc:8.2f}%")
        
    return results


def main():
    print("\n--- AGTECH PRECISION AGRICULTURE - TUTORIAL 3: CROSS VALIDATION ---\n")
    dataset_path = os.path.join("data", "Crop_recommendation.csv")
    
    X, y, feature_cols, classes = load_and_preprocess(dataset_path)
    run_kfold_cross_validation(X, y, num_folds=5)
    run_stratified_kfold_cross_validation(X, y, num_folds=5)
    
    print("\n" + "=" * 70)
    print(">>> TUTORIAL 3 (CROSS VALIDATION) COMPLETED SUCCESSFULLY! <<<")
    print("=" * 70)


if __name__ == "__main__":
    main()
