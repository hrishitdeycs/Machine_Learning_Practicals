import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, make_scorer,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
data, meta = arff.loadarff("dataset_31_credit-g.arff")
df = pd.DataFrame(data)

for col in df.select_dtypes("object"):
    df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

df = df.dropna()

# ---------------------------------------------------------
# 2. Features and target
# ---------------------------------------------------------
target = df.columns[-1]
X = pd.get_dummies(df.drop(columns=target), drop_first=True).astype(float)
y = df[target].map({"good": 0, "bad": 1}).astype(int)

print("Number of features after encoding:", X.shape[1])
print("\nClass labels:")
print("0 = good credit")
print("1 = bad credit (positive class)")
print("\nClass counts:\n", y.value_counts().to_string())

# ---------------------------------------------------------
# 3. Split data: 80% training, 20% testing
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ---------------------------------------------------------
# 4. Pipeline: scaling -> forward stepwise selection -> ANN
# ---------------------------------------------------------
n_features_to_select = 15   # arbitrary choice; change if needed


def make_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("select", SequentialFeatureSelector(
            LogisticRegression(max_iter=1000),   # fast model used only to score subsets
            n_features_to_select=n_features_to_select,
            direction="forward",
            scoring="f1",
            cv=3,
            n_jobs=-1,
        )),
        ("ann", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            random_state=42,
        )),
    ])


# ---------------------------------------------------------
# 5. Train on training set (selection fit on train only)
# ---------------------------------------------------------
model = make_pipeline()
model.fit(X_train, y_train)

selected_cols = X.columns[model["select"].get_support()]
print(f"\nSelected features (Forward Selection, F1) [{len(selected_cols)}]:")
print(list(selected_cols))

# ---------------------------------------------------------
# 6. Predictions and confusion matrix
# ---------------------------------------------------------
threshold = 0.50
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print("\nConfusion Matrix:")
print(pd.DataFrame(
    cm,
    index=["Actual Good", "Actual Bad"],
    columns=["Predicted Good", "Predicted Bad"],
))
TN, FP, FN, TP = cm.ravel()

# ---------------------------------------------------------
# 7. Performance metrics (80/20 test set)
# ---------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
error = 1 - accuracy
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
specificity = TN / (TN + FP) if (TN + FP) != 0 else 0.0
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

print("\n========== ANN RESULTS ==========")
print(f"Threshold   : {threshold}")
print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"TP          : {TP}")
print(f"TN          : {TN}")
print(f"FP          : {FP}")
print(f"FN          : {FN}")
print(f"Error       : {error:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1-score    : {f1:.4f}")
print(f"AUC         : {auc:.4f}")

# ---------------------------------------------------------
# 8. Stratified K-Fold CV (scaling + selection inside each fold)
# ---------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def _cm(y_t, y_p):
    return confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()  # tn, fp, fn, tp


def specificity_score(y_t, y_p):
    tn, fp, _, _ = _cm(y_t, y_p)
    return tn / (tn + fp) if (tn + fp) != 0 else 0.0


scoring = {
    "accuracy": "accuracy",
    "error": make_scorer(lambda a, b: 1 - accuracy_score(a, b)),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "specificity": make_scorer(specificity_score),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "TP": make_scorer(lambda a, b: _cm(a, b)[3]),
    "TN": make_scorer(lambda a, b: _cm(a, b)[0]),
    "FP": make_scorer(lambda a, b: _cm(a, b)[1]),
    "FN": make_scorer(lambda a, b: _cm(a, b)[2]),
}

cv_scores = cross_validate(make_pipeline(), X, y, cv=cv, scoring=scoring)

print("\n========== 5-FOLD CROSS-VALIDATION ==========")
for metric, vals in cv_scores.items():
    if metric.startswith("test_"):
        name = metric.replace("test_", "")
        print(f"{name:11s}: mean={vals.mean():.4f}, std={vals.std():.4f}")