"""
=============================================================================
CROP RECOMMENDATION SYSTEM - INFERENCE & PREDICTION DEMO
Project: Precision AgTech Crop Recommendation System
=============================================================================
This program loads the trained Machine Learning pre-processing artifacts 
(Scaler & LabelEncoder) and trained Random Forest Classifier model to predict 
the best crop to cultivate for any custom soil & weather parameters.
=============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_and_save_crop_model(data_path: str):
    """Ensure a trained Random Forest model is saved in models/."""
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    df = pd.read_csv(data_path)
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    X = df[feature_cols]
    y_raw = df['label']
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_scaled, y)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model, "models/crop_model.pkl")
    joblib.dump(scaler, "models/crop_scaler.pkl")
    joblib.dump(label_encoder, "models/crop_label_encoder.pkl")
    return rf_model, scaler, label_encoder


def predict_crop_for_soil(n, p, k, temp, humidity, ph, rainfall):
    """
    Takes 7 soil and climate inputs and predicts optimal crop.
    """
    model_path = os.path.join("models", "crop_model.pkl")
    scaler_path = os.path.join("models", "crop_scaler.pkl")
    encoder_path = os.path.join("models", "crop_label_encoder.pkl")
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path)):
        data_path = os.path.join("data", "Crop_recommendation.csv")
        rf_model, scaler, label_encoder = train_and_save_crop_model(data_path)
    else:
        rf_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(encoder_path)
    
    # Step 1: Format user inputs into a 2D array
    raw_input = np.array([[n, p, k, temp, humidity, ph, rainfall]])
    
    # Step 2: Scale user inputs using trained StandardScaler
    scaled_input = scaler.transform(raw_input)
    
    # Step 3: Predict crop class probabilities
    probabilities = rf_model.predict_proba(scaled_input)[0]
    best_class_idx = np.argmax(probabilities)
    confidence = probabilities[best_class_idx] * 100
    
    # Step 4: Map class integer back to crop name using LabelEncoder
    recommended_crop = label_encoder.inverse_transform([best_class_idx])[0]
    
    return recommended_crop, confidence, probabilities, label_encoder.classes_


def main():
    print("=" * 70)
    print("🌾 PRECISION AGTECH CROP RECOMMENDATION INFERENCE ENGINE 🌾")
    print("=" * 70)
    
    # Sample Test Scenario 1: Soil parameters for Rice
    print("\n--- TEST SCENARIO 1: High Nitrogen & Heavy Rainfall ---")
    n, p, k, temp, hum, ph, rain = 90, 42, 43, 20.8, 82.0, 6.5, 202.9
    print(f"Inputs -> N:{n}, P:{p}, K:{k}, Temp:{temp}°C, Humidity:{hum}%, pH:{ph}, Rainfall:{rain}mm")
    
    crop, conf, proba, classes = predict_crop_for_soil(n, p, k, temp, hum, ph, rain)
    print(f"👉 Recommended Crop: *** {crop.upper()} ***")
    print(f"👉 Model Confidence: {conf:.2f}%")
    
    # Sample Test Scenario 2: Soil parameters for Coffee
    print("\n--- TEST SCENARIO 2: Cool Mountain Temperature & Moderate Rainfall ---")
    n, p, k, temp, hum, ph, rain = 100, 20, 30, 24.0, 60.0, 6.8, 160.0
    print(f"Inputs -> N:{n}, P:{p}, K:{k}, Temp:{temp}°C, Humidity:{hum}%, pH:{ph}, Rainfall:{rain}mm")
    
    crop2, conf2, _, _ = predict_crop_for_soil(n, p, k, temp, hum, ph, rain)
    print(f"👉 Recommended Crop: *** {crop2.upper()} ***")
    print(f"👉 Model Confidence: {conf2:.2f}%")
    
    print("\n" + "=" * 70)
    print(">>> CROP RECOMMENDATION SYSTEM PREDICTION COMPLETE! <<<")
    print("=" * 70)


if __name__ == "__main__":
    main()
