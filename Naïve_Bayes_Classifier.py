import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
file_path = "11_mushroom_edibility.csv"
df = pd.read_csv(file_path)

# ---------------------------------------------------------
# 2. Define features and target
# ---------------------------------------------------------
X = df.drop(columns=["SampleID", "Class"], errors="ignore")
y = df["Class"]

data = pd.concat([X, y], axis=1).dropna()
X = data.drop(columns=["Class"])
y = data["Class"]

# ---------------------------------------------------------
# 3. Encode categorical variables
# ---------------------------------------------------------
# Scaling is omitted here because features are nominal categories
# and CategoricalNB expects discrete integer category indices.
X_encoded = X.copy()
feature_encoders = {}

for column in X_encoded.columns:
  encoder = LabelEncoder()
  X_encoded[column] = encoder.fit_transform(X_encoded[column].astype(str))
  feature_encoders[column] = encoder

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y.astype(str))

print("\nClass labels:")
for i, label in enumerate(target_encoder.classes_):
  print(f"{i} = {label}")

# ---------------------------------------------------------
# 4. Split data: 80% training, 20% testing
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ---------------------------------------------------------
# 5. Forward Stepwise Feature Selection (fit on train only)
# ---------------------------------------------------------
n_features_to_select = min(8, X_train.shape[1])

sfs_forward = SequentialFeatureSelector(
    estimator=CategoricalNB(),
    n_features_to_select=n_features_to_select,
    direction="forward",
    scoring="accuracy",
    cv=3,
    n_jobs=-1,
)
sfs_forward.fit(X_train, y_train)

selected_cols = X_train.columns[sfs_forward.get_support()]
print("\nSelected features (Forward Selection):", list(selected_cols))

X_train_sel = X_train[selected_cols]
X_test_sel = X_test[selected_cols]

# ---------------------------------------------------------
# 6. Train Naive Bayes model
# ---------------------------------------------------------
model = CategoricalNB()
model.fit(X_train_sel, y_train)

# ---------------------------------------------------------
# 7. Predictions & Confusion Matrix
# ---------------------------------------------------------
y_pred = model.predict(X_test_sel)
y_prob = model.predict_proba(X_test_sel)[:, 1]

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

if cm.shape != (2, 2):
  raise ValueError("This code expects exactly two classes in the Class column.")

TN, FP, FN, TP = cm.ravel()

# ---------------------------------------------------------
# 8. Performance metrics (80/20 test set)
# ---------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
error = 1 - accuracy
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

print("\n========== NAIVE BAYES RESULTS ==========")
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
# 9. K-Fold Cross-Validation (feature selection inside each fold)
# ---------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def specificity_score(y_true, y_pred):
  tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
  return tn / (tn + fp) if (tn + fp) != 0 else 0.0


cv_pipeline = Pipeline([
    (
        "select",
        SequentialFeatureSelector(
            estimator=CategoricalNB(),
            n_features_to_select=n_features_to_select,
            direction="forward",
            scoring="accuracy",
            cv=3,
            n_jobs=-1,
        ),
    ),
    ("clf", CategoricalNB()),
])

scoring = {
    "accuracy": "accuracy",
    "error": make_scorer(lambda y_t, y_p: 1 - accuracy_score(y_t, y_p)),
    "precision": "precision",
    "recall": "recall",
    "specificity": make_scorer(specificity_score),
    "f1": "f1",
    "roc_auc": "roc_auc",
}

cv_scores = cross_validate(
    cv_pipeline,
    X_encoded,
    y_encoded,
    cv=cv,
    scoring=scoring,
)

print("\n========== 5-FOLD CROSS-VALIDATION ==========")
for metric, vals in cv_scores.items():
  if metric.startswith("test_"):
    clean_name = metric.replace("test_", "")
    print(f"{clean_name:11s}: mean={vals.mean():.4f}, std={vals.std():.4f}")