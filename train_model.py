import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

import joblib

# =====================================
# STEP 1: LOAD DATASET
# =====================================

file_path = Path("data/crx.data")

df = pd.read_csv(
    file_path,
    header=None,
    na_values="?"
)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# =====================================
# STEP 2: SEPARATE FEATURES AND TARGET
# =====================================

X = df.iloc[:, :-1].copy()
y = df.iloc[:, -1].copy()

# Encode the target labels:
# + = approved (positive class)
# - = not approved (negative class)
y = y.map({"+": 1, "-": 0})

if y.isnull().any():
    raise ValueError("Unexpected or missing target labels found.")

# =====================================
# STEP 3: IDENTIFY COLUMN TYPES
# =====================================

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

# =====================================
# STEP 4: CREATE PREPROCESSING PIPELINES
# =====================================

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# =====================================
# STEP 5: SPLIT DATASET
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# =====================================
# STEP 6: DEFINE MODELS
# =====================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
}

# =====================================
# STEP 7: TRAIN AND EVALUATE
# =====================================

results = []
trained_models = {}

for name, classifier in models.items():

    print(f"\nTraining {name}...")

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })

    trained_models[name] = model

    print(f"\n{name} Results")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-Score:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

# =====================================
# STEP 8: COMPARE MODELS
# =====================================

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(
    results_df.sort_values(
        by="F1-Score",
        ascending=False
    ).to_string(index=False)
)

# =====================================
# STEP 9: SAVE THE BEST MODEL
# =====================================

# Select the model with the highest test F1-score.
best_model_name = results_df.loc[
    results_df["F1-Score"].idxmax(),
    "Model"
]

best_model = trained_models[best_model_name]

Path("models").mkdir(exist_ok=True)

joblib.dump(
    best_model,
    "models/creditworthiness_model.pkl"
)

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("\nBest model:", best_model_name)
print("Saved model: models/creditworthiness_model.pkl")
print("Saved comparison: models/model_comparison.csv")
print("\nModel training completed successfully!")