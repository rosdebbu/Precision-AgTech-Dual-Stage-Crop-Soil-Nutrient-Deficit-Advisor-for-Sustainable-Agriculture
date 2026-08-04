import json
import os

os.makedirs('notebooks', exist_ok=True)

# ---------------------------------------------------------
# TUTORIAL 2 NOTEBOOK: LINEAR REGRESSION
# ---------------------------------------------------------
t2_notebook = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# Tutorial 2 (T2): Linear Regression in Python\n',
                '**Course / Lab Project**: Precision AgTech Climate & Soil Regression Analysis\n',
                '**Goal**: Predict continuous environmental variables (`rainfall` in mm) from soil and atmospheric metrics ($N, P, K$, Temperature, Humidity, pH).\n\n',
                '---\n',
                '## Objectives of Tutorial 2\n',
                '1. **Understanding Linear Regression**: Modeling relationships between continuous features ($X$) and target ($y$).\n',
                '2. **Ordinary Least Squares (OLS) Multiple Linear Regression**: Finding optimal feature coefficients.\n',
                '3. **Regularized Linear Regression**: Implementing Ridge ($L_2$) and Lasso ($L_1$) regression.\n',
                '4. **Evaluation Metrics**: Computing Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and $R^2$ Score.\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# Step 1: Import regression libraries\n',
                'import pandas as pd\n',
                'import numpy as np\n',
                'import matplotlib.pyplot as plt\n',
                'import seaborn as sns\n',
                'from sklearn.model_selection import train_test_split\n',
                'from sklearn.linear_model import LinearRegression, Ridge, Lasso\n',
                'from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n',
                'from sklearn.preprocessing import StandardScaler\n',
                'import joblib\n',
                'import os\n\n',
                'print("Linear Regression libraries imported!")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Step 1: Load Dataset & Define Features/Target'
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
                'feature_cols = ["N", "P", "K", "temperature", "humidity", "ph"]\n',
                'target_col = "rainfall"\n\n',
                'X = df[feature_cols]\n',
                'y = df[target_col]\n',
                'df[feature_cols + [target_col]].head()'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Step 2: Train-Test Split and Feature Scaling'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'X_train, X_test, y_train, y_test = train_test_split(\n',
                '    X, y, test_size=0.20, random_state=42\n',
                ')\n\n',
                'scaler = StandardScaler()\n',
                'X_train_scaled = scaler.fit_transform(X_train)\n',
                'X_test_scaled = scaler.transform(X_test)\n',
                'print(f"Training shapes: X_train={X_train_scaled.shape}, y_train={y_train.shape}")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Step 3: Train Multiple Linear Regression (OLS), Ridge, and Lasso'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'models = {\n',
                '    "OLS Linear Regression": LinearRegression(),\n',
                '    "Ridge Regression (L2)": Ridge(alpha=1.0),\n',
                '    "Lasso Regression (L1)": Lasso(alpha=0.1)\n',
                '}\n\n',
                'for name, model in models.items():\n',
                '    model.fit(X_train_scaled, y_train)\n',
                '    y_pred = model.predict(X_test_scaled)\n',
                '    mse = mean_squared_error(y_test, y_pred)\n',
                '    rmse = np.sqrt(mse)\n',
                '    mae = mean_absolute_error(y_test, y_pred)\n',
                '    r2 = r2_score(y_test, y_pred)\n',
                '    print(f"=== {name} ===")\n',
                '    print(f"MSE:  {mse:.4f}")\n',
                '    print(f"RMSE: {rmse:.4f} mm")\n',
                '    print(f"MAE:  {mae:.4f} mm")\n',
                '    print(f"R2 Score: {r2:.4f}\\n")'
            ]
        }
    ],
    'metadata': {'language_info': {'name': 'python'}},
    'nbformat': 4,
    'nbformat_minor': 2
}

with open('notebooks/02_tutorial2_linear_regression.ipynb', 'w') as f:
    json.dump(t2_notebook, f, indent=2)

# ---------------------------------------------------------
# TUTORIAL 3 NOTEBOOK: CROSS VALIDATION
# ---------------------------------------------------------
t3_notebook = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# Tutorial 3 (T3): Cross-Validation in Python\n',
                '**Course / Lab Project**: Precision AgTech Model Evaluation & Validation\n',
                '**Goal**: Evaluate model performance and stability using K-Fold and Stratified K-Fold Cross Validation.\n\n',
                '---\n',
                '## Objectives of Tutorial 3\n',
                '1. **Understanding K-Fold Cross Validation**: Dividing dataset into $K$ equal folds to prevent train-test split bias.\n',
                '2. **Stratified K-Fold CV**: Ensuring class balance across folds for multi-class crop prediction.\n',
                '3. **Multi-Algorithm Benchmark**: Comparing Random Forest, Decision Tree, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and Logistic Regression.\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# Step 1: Import Cross Validation modules\n',
                'import pandas as pd\n',
                'import numpy as np\n',
                'import matplotlib.pyplot as plt\n',
                'import seaborn as sns\n',
                'from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate\n',
                'from sklearn.preprocessing import StandardScaler, LabelEncoder\n',
                'from sklearn.ensemble import RandomForestClassifier\n',
                'from sklearn.tree import DecisionTreeClassifier\n',
                'from sklearn.svm import SVC\n',
                'from sklearn.neighbors import KNeighborsClassifier\n',
                'from sklearn.naive_bayes import GaussianNB\n',
                'from sklearn.linear_model import LogisticRegression\n',
                'import os\n\n',
                'print("Cross-Validation libraries imported!")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Step 1: Load and Preprocess Data'
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
                'feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]\n',
                'X = df[feature_cols]\n',
                'y_raw = df["label"]\n\n',
                'encoder = LabelEncoder()\n',
                'y = encoder.fit_transform(y_raw)\n',
                'scaler = StandardScaler()\n',
                'X_scaled = scaler.fit_transform(X)\n',
                'print(f"Data ready: X={X_scaled.shape}, y={y.shape}")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Step 2: 5-Fold Stratified Cross-Validation Benchmark'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n',
                'classifiers = {\n',
                '    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),\n',
                '    "Decision Tree": DecisionTreeClassifier(random_state=42),\n',
                '    "SVM (RBF Kernel)": SVC(kernel="rbf", random_state=42),\n',
                '    "KNN (K=5)": KNeighborsClassifier(n_neighbors=5),\n',
                '    "Gaussian Naive Bayes": GaussianNB(),\n',
                '    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)\n',
                '}\n\n',
                'results = []\n',
                'for name, clf in classifiers.items():\n',
                '    scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring="accuracy") * 100\n',
                '    results.append({\n',
                '        "Algorithm": name,\n',
                '        "Mean Accuracy (%)": scores.mean(),\n',
                '        "Std Dev (%)": scores.std(),\n',
                '        "Min Fold (%)": scores.min(),\n',
                '        "Max Fold (%)": scores.max()\n',
                '    })\n\n',
                'results_df = pd.DataFrame(results)\n',
                'results_df.sort_values(by="Mean Accuracy (%)", ascending=False)'
            ]
        }
    ],
    'metadata': {'language_info': {'name': 'python'}},
    'nbformat': 4,
    'nbformat_minor': 2
}

with open('notebooks/03_tutorial3_cross_validation.ipynb', 'w') as f:
    json.dump(t3_notebook, f, indent=2)

print("Tutorial 2 and Tutorial 3 Notebooks created successfully!")
