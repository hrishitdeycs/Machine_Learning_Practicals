import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, make_scorer,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
df = pd.read_csv(
    "data_banknote_authentication.txt",
    header=None,
    names=["variance", "skewness", "curtosis", "entropy", "class"],
)
df = df.dropna()

# ---------------------------------------------------------
# 2. Features and target
# ---------------------------------------------------------
X = df.drop("class", axis=1)
y = df["class"].astype(int)

print("Features:", list(X.columns))
print("\nClass labels:")
print("0 = genuine banknote")
print("1 = forged banknote (positive class)")
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
# 4. Pipeline: scaling -> forward stepwise selection -> linear SVM
# ---------------------------------------------------------
n_features_to_select = 3   # 4 -> 3 features (arbitrary choice)


def make_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("select", SequentialFeatureSelector(
            SVC(kernel="linear", random_state=42),
            n_features_to_select=n_features_to_select,
            direction="forward",
            scoring="f1",
            cv=3,
            n_jobs=-1,
        )),
        ("svm", SVC(kernel="linear", random_state=42)),
    ])


# ---------------------------------------------------------
# 5. Train on training set (scaling and selection fit on train only)
# ---------------------------------------------------------
model = make_pipeline()
model.fit(X_train, y_train)

selected_cols = X.columns[model["select"].get_support()]
print(f"\nSelected features (Forward Selection, F1) [{len(selected_cols)}]:")
print(list(selected_cols))

# ---------------------------------------------------------
# 6. Predictions and confusion matrix
# ---------------------------------------------------------
y_pred = model.predict(X_test)
y_score = model.decision_function(X_test)   # used for AUC

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print("\nConfusion Matrix:")
print(pd.DataFrame(
    cm,
    index=["Actual 0 (genuine)", "Actual 1 (forged)"],
    columns=["Predicted 0", "Predicted 1"],
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
auc = roc_auc_score(y_test, y_score)

print("\n========== LINEAR SVM RESULTS ==========")
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