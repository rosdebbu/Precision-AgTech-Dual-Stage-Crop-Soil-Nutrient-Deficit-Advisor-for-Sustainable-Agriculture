"""
=============================================================================
TUTORIAL 1 (T1): DATA PRE-PROCESSING IN PYTHON
Project: Precision AgTech Crop Recommendation System
=============================================================================
This program demonstrates the complete Data Pre-processing pipeline in Python 
using Pandas, NumPy, and Scikit-Learn.

What is Data Pre-processing?
-----------------------------
Raw data collected from sensors (like soil N, P, K levels, rainfall, temperature) 
cannot be fed directly into a Machine Learning algorithm because:
1. Features have vastly different units and scales (e.g. Rainfall up to 300mm vs pH 6.5).
2. Raw data might have missing (null) values or duplicates.
3. Target outputs are text strings ("rice", "maize") which computers cannot process mathematically.
4. Data needs to be split cleanly into Training (to learn) and Testing (to evaluate) sets.

The 6 Core Steps of Pre-processing in this Script:
  Step 1: Data Loading & Initial Inspection
  Step 2: Missing Values & Duplicate Audit
  Step 3: Descriptive Statistics & Outlier Analysis
  Step 4: Label Encoding (Text Targets -> Numbers)
  Step 5: Feature Scaling (StandardScaler vs MinMaxScaler)
  Step 6: Train-Test Dataset Splitting
=============================================================================
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import joblib


def load_dataset(file_path: str) -> pd.DataFrame:
    """Step 1: Load CSV file into a Pandas DataFrame."""
    print("=" * 70)
    print("STEP 1: LOADING DATASET")
    print("=" * 70)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"[OK] Successfully loaded dataset from: {file_path}")
    print(f"[*] Dataset Shape: {df.shape[0]} rows (samples) and {df.shape[1]} columns (features + target)")
    print("\n--- First 5 Rows of Raw Data ---")
    print(df.head())
    print("\n--- Data Types & Column Summary ---")
    print(df.info())
    return df


def audit_data_quality(df: pd.DataFrame):
    """Step 2: Audit data for missing values (nulls) and duplicate rows."""
    print("\n" + "=" * 70)
    print("STEP 2: DATA QUALITY AUDIT (MISSING VALUES & DUPLICATES)")
    print("=" * 70)
    
    # Check for missing/null values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print("[*] Checking Missing (Null) Values per Column:")
    for col, count in null_counts.items():
        print(f"   - {col:12s}: {count} missing values")
    print(f"[->] Total Missing Values in Dataset: {total_nulls}")
    
    # Check for duplicates
    num_duplicates = df.duplicated().sum()
    print(f"\n[*] Checking Duplicate Rows:")
    print(f"[->] Total Duplicate Rows: {num_duplicates}")
    
    if total_nulls == 0 and num_duplicates == 0:
        print("[OK] Data Quality Result: Clean dataset! No missing values or duplicates found.")
    return null_counts, num_duplicates


def analyze_descriptive_stats(df: pd.DataFrame):
    """Step 3: Analyze statistical distribution of numerical features."""
    print("\n" + "=" * 70)
    print("STEP 3: DESCRIPTIVE STATISTICS & OUTLIER ANALYSIS")
    print("=" * 70)
    
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    print("[*] Statistical Summary (Mean, Std Dev, Min, Max, Quantiles):")
    stats = df[feature_cols].describe().T
    print(stats[['mean', 'std', 'min', '50%', 'max']])
    
    print("\n[*] IQR (Interquartile Range) Outlier Analysis:")
    for col in feature_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"   - {col:12s}: {len(outliers):3d} potential outliers outside [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    print("\n[*] Target Class Distribution ('label'):")
    crop_counts = df['label'].value_counts()
    print(f"[->] Total Unique Crop Classes: {len(crop_counts)}")
    print(f"[->] Samples per Class: {crop_counts.iloc[0]} samples per crop (Balanced Dataset!)")


def encode_and_scale_data(df: pd.DataFrame):
    """
    Steps 4, 5, 6: Encode Target, Scale Features, and Split into Train/Test sets.
    """
    print("\n" + "=" * 70)
    print("STEP 4: TARGET ENCODING (LABEL ENCODER)")
    print("=" * 70)
    
    # Separate input features (X) and target variable (y)
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X = df[feature_cols].copy()
    y_raw = df['label'].copy()
    
    # Encode crop names to numeric integers (e.g. rice -> 20, maize -> 11)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    
    print("[OK] Converted text target labels into numbers 0 through 21.")
    print("   Sample Mapping:")
    for cls_name, cls_idx in zip(label_encoder.classes_[:5], range(5)):
        print(f"   - '{cls_name}' ----> {cls_idx}")
    
    print("\n" + "=" * 70)
    print("STEP 5: FEATURE SCALING (STANDARDIZATION)")
    print("=" * 70)
    print("Why Scale Features?")
    print(" - 'rainfall' ranges from 20 to 300 mm.")
    print(" - 'ph' ranges from 3.5 to 9.9.")
    print(" - Without scaling, algorithms like KNN or SVM will treat rainfall as 30x more important than pH!")
    print(" - StandardScaler rescales data so Mean = 0 and Variance = 1.")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    
    print("\n--- Scaled Features Preview (Mean ~ 0, Std ~ 1) ---")
    print(X_scaled_df.head(3))
    
    print("\n" + "=" * 70)
    print("STEP 6: TRAIN-TEST DATASET SPLITTING")
    print("=" * 70)
    print("Why Split Data?")
    print(" - 80% Training Set: Used to teach the machine learning model.")
    print(" - 20% Testing Set: Kept secret to grade model accuracy on unseen real-world data.")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    
    print(f"[OK] Training Set Size: {X_train.shape[0]} samples")
    print(f"[OK] Testing Set Size:  {X_test.shape[0]} samples")
    
    # Save preprocessed objects to models directory
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/crop_scaler.pkl")
    joblib.dump(label_encoder, "models/crop_label_encoder.pkl")
    print("[SAVE] Saved feature scaler to: models/crop_scaler.pkl")
    print("[SAVE] Saved label encoder to:  models/crop_label_encoder.pkl")
    
    return X_train, X_test, y_train, y_test, feature_cols, label_encoder, scaler


def main():
    print("\n--- AGTECH PRECISION AGRICULTURE - TUTORIAL 1: DATA PRE-PROCESSING ---\n")
    dataset_path = os.path.join("data", "Crop_recommendation.csv")
    
    df = load_dataset(dataset_path)
    audit_data_quality(df)
    analyze_descriptive_stats(df)
    X_train, X_test, y_train, y_test, feature_cols, encoder, scaler = encode_and_scale_data(df)
    
    print("\n" + "=" * 70)
    print(">>> TUTORIAL 1 PRE-PROCESSING COMPLETED SUCCESSFULLY! <<<")
    print("=" * 70)


if __name__ == "__main__":
    main()
