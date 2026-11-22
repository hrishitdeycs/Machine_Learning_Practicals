import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Settings
DIRECTION = "forward"      # "forward" or "backward"
N_FEATURES = 8             # number of features to keep

# 1. Load dataset
data = load_diabetes(as_frame=True)
X, y = data.data, data.target
print("Shape:", X.shape)
print("Features:", list(X.columns))

# 2. Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 3. Pipeline: z-score -> stepwise selection -> regression
def make_pipeline(name):
    if name == "Ridge":
        reg = RidgeCV(alphas=np.logspace(-3, 3, 50))
    elif name == "Lasso":
        reg = LassoCV(alphas=np.logspace(-3, 1, 50), cv=5, max_iter=10000)
    else:
        reg = LinearRegression()
    return Pipeline([
        ("scale", StandardScaler()),
        ("select", SequentialFeatureSelector(
            LinearRegression(),
            n_features_to_select=N_FEATURES,
            direction=DIRECTION,
            scoring="r2",
            cv=5)),
        ("reg", reg),
    ])

cv = KFold(n_splits=5, shuffle=True, random_state=42)
summary = []
fitted = {}

for name in ["Linear (baseline)", "Ridge", "Lasso"]:
    key = name.split()[0]
    pipe = make_pipeline(key).fit(X_train, y_train)
    fitted[key] = pipe

    sel_cols = X.columns[pipe["select"].get_support()]
    reg = pipe["reg"]

    print(f"\n==================== {name} ====================")
    print(f"Selected features ({DIRECTION} selection):", list(sel_cols))
    print("Dropped features:", [c for c in X.columns if c not in sel_cols])
    if hasattr(reg, "alpha_"):
        print(f"Best alpha: {reg.alpha_:.5f}")
    print("Coefficients:")
    for c, w in zip(sel_cols, reg.coef_):
        print(f"  {c:4s}: {w:9.4f}")
    if key == "Lasso":
        print("Features shrunk to exactly 0:",
              [c for c, w in zip(sel_cols, reg.coef_) if w == 0])

    # Hold-out evaluation
    y_pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("\n--- Hold-out (80/20) ---")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {np.sqrt(mse):.4f}")
    print(f"MAE : {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"R2  : {r2_score(y_test, y_pred):.4f}")

    # K-fold CV (scaling, selection and alpha tuning redone inside each fold)
    scores = cross_validate(
        make_pipeline(key), X, y, cv=cv,
        scoring={"MSE": "neg_mean_squared_error",
                 "RMSE": "neg_root_mean_squared_error",
                 "MAE": "neg_mean_absolute_error",
                 "R2": "r2"})
    print("\n--- 5-Fold CV ---")
    row = {"Model": key,
           "Test RMSE": np.sqrt(mse),
           "Test R2": r2_score(y_test, y_pred)}
    for k in ["MSE", "RMSE", "MAE"]:
        v = -scores[f"test_{k}"]
        print(f"{k:5s}: mean={v.mean():.4f}, std={v.std():.4f}")
        row[f"CV {k}"] = v.mean()
    v = scores["test_R2"]
    print(f"R2   : mean={v.mean():.4f}, std={v.std():.4f}")
    row["CV R2"] = v.mean()
    summary.append(row)

# 4. Comparison table
print("\n==================== COMPARISON ====================")
print(pd.DataFrame(summary).set_index("Model").round(4))

# 5. Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sel_cols = X.columns[fitted["Ridge"]["select"].get_support()]
idx = np.arange(len(sel_cols))
axes[0].bar(idx - 0.2, fitted["Ridge"]["reg"].coef_, 0.4, label="Ridge")
axes[0].bar(idx + 0.2, fitted["Lasso"]["reg"].coef_, 0.4, label="Lasso")
axes[0].set_xticks(idx)
axes[0].set_xticklabels(sel_cols)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_title("Coefficients (standardised features)")
axes[0].legend()

for ax, key in zip(axes[1:], ["Ridge", "Lasso"]):
    pred = fitted[key].predict(X_test)
    ax.scatter(y_test, pred, alpha=0.7)
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(lims, lims, color="red", linewidth=2)
    ax.set_title(f"{key}: Actual vs Predicted")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True)

plt.tight_layout()
plt.show()