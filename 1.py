
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ============================================================
# 1. Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "churn_predictor.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# This is a chosen operating threshold, not a guaranteed optimum.
CHURN_THRESHOLD = 0.40


# ============================================================
# 2. Load and validate the dataset
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

required_columns = {
    "customerID",
    "Churn",
    "TotalCharges",
    "tenure",
    "MonthlyCharges",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Dataset is missing required columns: {sorted(missing_columns)}"
    )

if df.empty:
    raise ValueError("The dataset is empty.")

# Remove the identifier because it is not a useful predictive feature.
df = df.drop(columns=["customerID"])

# Normalize target labels and explicitly map them to 0 and 1.
df["Churn"] = df["Churn"].astype("string").str.strip()

target_mapping = {"No": 0, "Yes": 1}

if df["Churn"].isna().any():
    raise ValueError("The target column contains missing labels.")

invalid_targets = set(df["Churn"].unique()) - set(target_mapping)

if invalid_targets:
    raise ValueError(f"Unexpected target labels: {invalid_targets}")

y = df["Churn"].map(target_mapping).astype(int)
X = df.drop(columns=["Churn"])

# Convert TotalCharges safely because the Telco dataset may contain blanks.
X["TotalCharges"] = pd.to_numeric(
    X["TotalCharges"].astype("string").str.strip(),
    errors="coerce",
)

# Only interpret missing TotalCharges as zero when tenure is zero.
# Other missing values will be imputed within the ML pipeline.
zero_tenure = X["tenure"].eq(0)
X.loc[zero_tenure & X["TotalCharges"].isna(), "TotalCharges"] = 0

if y.nunique() != 2:
    raise ValueError("Both churn classes must be present.")

if y.value_counts().min() < 5:
    raise ValueError(
        "At least five examples of each target class are required "
        "for five-fold stratified cross-validation."
    )

print("\nDataset shape:", df.shape)
print("\nTarget distribution:")
print(y.value_counts().rename(index={0: "No", 1: "Yes"}))
print("\nMissing values:")
print(X.isna().sum())


# ============================================================
# 3. Identify feature types
# ============================================================

cat_cols = X.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

num_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nCategorical features:", cat_cols)
print("Numerical features:", num_cols)


# ============================================================
# 4. Create leakage-safe preprocessing
# ============================================================

numeric_pipeline = Pipeline([
    # Fit the imputer using training data only.
    ("imputer", SimpleImputer(strategy="median")),

    # Scaling is useful for Logistic Regression, KNN and SVC.
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),

    # Ignore categories that were not seen during fitting.
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("categorical", categorical_pipeline, cat_cols),
    ("numerical", numeric_pipeline, num_cols),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(random_state=RANDOM_STATE)),
])


# ============================================================
# 5. Split data before fitting or tuning
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

cv_strategy = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ============================================================
# 6. Define candidate models and hyperparameters
# ============================================================

param_grid = [
    {
        "model": [LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        )],
        "model__C": [0.1, 1, 10],
    },
    {
        "model": [KNeighborsClassifier()],
        "model__n_neighbors": [5, 9],
        "model__weights": ["uniform", "distance"],
    },
    {
        # CHANGE: probability=True enables predict_proba().
        "model": [SVC(
            probability=True,
            random_state=RANDOM_STATE,
        )],
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear", "rbf"],
    },
    {
        "model": [RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1,
        )],
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10],
    },
    {
        "model": [XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=1,
        )],
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 6],
    },
    {
        "model": [LGBMClassifier(
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=1,
        )],
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [-1, 6],
    },
]


# ============================================================
# 7. Run hyperparameter search
# ============================================================

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv_strategy,
    n_jobs=-1,
    refit=True,
    error_score="raise",
    return_train_score=True,
)

print("\nTraining and tuning models...")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# GridSearchCV selects the model using training-fold CV results.
# The held-out test set is used only for final evaluation.
print("\nBest parameters:")
print(grid.best_params_)

print("\nBest cross-validation F1:", round(grid.best_score_, 4))


# ============================================================
# 8. Evaluate the selected model
# ============================================================

# CHANGE: Locate the positive class by its actual label.
classes = list(best_model.classes_)

if 1 not in classes:
    raise ValueError("The fitted model does not contain churn class 1.")

churn_class_index = classes.index(1)

test_probabilities = best_model.predict_proba(X_test)
churn_probabilities = test_probabilities[:, churn_class_index]

# Apply the explicit threshold without changing the test data.
y_pred = (churn_probabilities >= CHURN_THRESHOLD).astype(int)

print("\nFinal held-out test metrics")
print("Threshold:", CHURN_THRESHOLD)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(
    precision_score(y_test, y_pred, zero_division=0), 4
))
print("Recall:", round(
    recall_score(y_test, y_pred, zero_division=0), 4
))
print("F1:", round(
    f1_score(y_test, y_pred, zero_division=0), 4
))

if len(np.unique(y_test)) == 2:
    print("ROC-AUC:", round(
        roc_auc_score(y_test, churn_probabilities), 4
    ))
    print("PR-AUC (average precision):", round(
        average_precision_score(y_test, churn_probabilities), 4
    ))
else:
    warnings.warn(
        "ROC-AUC and PR-AUC cannot be reliably computed "
        "because the test set contains only one class."
    )

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["No Churn", "Churn"],
    zero_division=0,
))


# ============================================================
# 9. Save the full fitted pipeline and metadata
# ============================================================

# CHANGE: Save one structured artifact for reliable API loading.
artifact = {
    "model": best_model,
    "threshold": CHURN_THRESHOLD,
    "target_mapping": target_mapping,
    "churn_class_index": churn_class_index,
    "feature_columns": X.columns.tolist(),
    "categorical_columns": cat_cols,
    "numerical_columns": num_cols,
    "model_name": type(best_model.named_steps["model"]).__name__,
    "cv_f1": float(grid.best_score_),
    "test_metrics": {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    },
}

joblib.dump(artifact, MODEL_PATH)

print(f"\nModel artifact saved to: {MODEL_PATH}")
print("Training pipeline completed successfully.")
