import json
import os

os.makedirs('notebooks', exist_ok=True)

notebook = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# Tutorial 1 (T1): Data Pre-processing in Python for Crop Recommendation System\n',
                '**Course / Lab Project**: Precision AgTech Dual-Stage Advisor\n',
                '**Dataset**: `Crop_recommendation.csv` (2,200 rows, 7 features, 1 target label)\n\n',
                '---\n',
                '## Objectives of Tutorial 1\n',
                '1. **Data Loading & Inspection**: Read raw CSV data, inspect column types and shapes.\n',
                '2. **Data Cleaning**: Audit for missing (null) values and duplicate records.\n',
                '3. **Exploratory Data Analysis (EDA)**: Compute summary statistics (mean, std, IQR outlier analysis).\n',
                '4. **Target Encoding**: Convert text crop names (`rice`, `maize`, `apple`) into numeric integers using `LabelEncoder`.\n',
                '5. **Feature Scaling**: Apply `StandardScaler` to standardize features ($Mean=0, Std=1$).\n',
                '6. **Train-Test Splitting**: Divide data into 80% Training and 20% Testing sets using stratified sampling.\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# Step 1: Import required Python libraries\n',
                'import pandas as pd\n',
                'import numpy as np\n',
                'import matplotlib.pyplot as plt\n',
                'import seaborn as sns\n',
                'from sklearn.model_selection import train_test_split\n',
                'from sklearn.preprocessing import StandardScaler, LabelEncoder\n',
                'import joblib\n',
                'import os\n\n',
                'print("Libraries imported successfully!")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Step 1: Load and Inspect Dataset'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'file_path = os.path.join("data", "Crop_recommendation.csv")\n',
                'if not os.path.exists(file_path):\n',
                '    file_path = os.path.join("..", "data", "Crop_recommendation.csv")\n\n',
                'df = pd.read_csv(file_path)\n',
                'print(f"Dataset Shape: {df.shape[0]} rows and {df.shape[1]} columns")\n',
                'df.head()'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Step 2: Data Quality Audit (Missing Values & Duplicates)'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'print("Missing Values per Column:")\n',
                'print(df.isnull().sum())\n\n',
                'print(f"Total Duplicate Rows: {df.duplicated().sum()}")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Step 3: Statistical Summary & Outlier Analysis'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]\n',
                'df[feature_cols].describe().T'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Step 4: Target Encoding (LabelEncoder)'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'X = df[feature_cols].copy()\n',
                'y_raw = df["label"].copy()\n\n',
                'label_encoder = LabelEncoder()\n',
                'y_encoded = label_encoder.fit_transform(y_raw)\n\n',
                'print("Unique Crops:", label_encoder.classes_)\n',
                'print("Total Classes:", len(label_encoder.classes_))'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Step 5: Feature Scaling (StandardScaler)'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'scaler = StandardScaler()\n',
                'X_scaled = scaler.fit_transform(X)\n\n',
                'X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)\n',
                'X_scaled_df.head()'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Step 6: Train-Test Dataset Splitting'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'X_train, X_test, y_train, y_test = train_test_split(\n',
                '    X_scaled, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded\n',
                ')\n\n',
                'print(f"Training set shape: {X_train.shape}")\n',
                'print(f"Testing set shape:  {X_test.shape}")\n\n',
                '# Save preprocessed objects\n',
                'os.makedirs("models", exist_ok=True)\n',
                'joblib.dump(scaler, "models/crop_scaler.pkl")\n',
                'joblib.dump(label_encoder, "models/crop_label_encoder.pkl")\n',
                'print("Saved crop_scaler.pkl and crop_label_encoder.pkl successfully!")'
            ]
        }
    ],
    'metadata': {
        'language_info': {'name': 'python'}
    },
    'nbformat': 4,
    'nbformat_minor': 2
}

with open('notebooks/01_tutorial1_crop_recommendation.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Notebook notebooks/01_tutorial1_crop_recommendation.ipynb generated successfully!")
