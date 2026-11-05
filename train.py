from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "churn_predictor.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD = 0.5

NUMERIC_COLUMNS = ["CustomerAge", "TenureMonths", "MonthlyCharges", "SupportTickets", "UsageHours"]
CATEGORICAL_COLUMNS = ["ContractType"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

df = pd.read_csv(DATA_PATH)

missing_columns = [column for column in FEATURE_COLUMNS + ["Churn"] if column not in df.columns]
if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")

y = df["Churn"].str.strip().map({"No": 0, "Yes": 1})
if y.isna().any():
    raise ValueError("Churn must contain only 'Yes' or 'No'.")
if y.nunique() != 2 or y.value_counts().min() < 5:
    raise ValueError("Both churn classes need at least five examples.")

X = df[FEATURE_COLUMNS]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

pipeline = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_COLUMNS),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
    ])),
    ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
])

search = GridSearchCV(
    estimator=pipeline,
    param_grid={"model__C": [0.1, 1, 10]},
    scoring="f1",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
)
search.fit(X_train, y_train)
best_model = search.best_estimator_

churn_class_index = list(best_model.classes_).index(1)
test_probability = best_model.predict_proba(X_test)[:, churn_class_index]
test_prediction = (test_probability >= THRESHOLD).astype(int)

test_metrics = {
    "accuracy": float(accuracy_score(y_test, test_prediction)),
    "precision": float(precision_score(y_test, test_prediction, zero_division=0)),
    "recall": float(recall_score(y_test, test_prediction, zero_division=0)),
    "f1": float(f1_score(y_test, test_prediction, zero_division=0)),
}

artifact = {
    "model": best_model,
    "model_name": type(best_model.named_steps["model"]).__name__,
    "threshold": THRESHOLD,
    "feature_columns": FEATURE_COLUMNS,
    "numeric_columns": NUMERIC_COLUMNS,
    "categorical_columns": CATEGORICAL_COLUMNS,
    "churn_class_index": churn_class_index,
    "best_params": search.best_params_,
    "cv_f1": float(search.best_score_),
    "test_metrics": test_metrics,
    "training_rows": len(X_train),
    "test_rows": len(X_test),
}

joblib.dump(artifact, MODEL_PATH)

print("Best parameters:", search.best_params_)
print("Cross-validation F1:", round(search.best_score_, 4))
print("Held-out test metrics:", {key: round(value, 4) for key, value in test_metrics.items()})
print(f"Model artifact saved to: {MODEL_PATH}")
