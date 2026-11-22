import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# 1. Load the three files
yield_df = pd.read_csv("FAOSTAT_data_en_8-4-2026.csv")
nit_df = pd.read_csv("nitrogen.csv")
land_df = pd.read_csv("land_use.csv")

# Check what is inside (helps if the filters below don't match)
for name, d in [("nitrogen", nit_df), ("land_use", land_df)]:
    print(name, "columns:", list(d.columns))
    if "Item" in d.columns:
        print(name, "items:", d["Item"].unique()[:10])

# 2. Keep the required rows, one value per Year
nit_df = nit_df[nit_df["Item"].str.contains("Nutrient nitrogen N", case=False, na=False)]
land_df = land_df[land_df["Item"].str.contains("Arable land", case=False, na=False)]

yield_y = yield_df.groupby("Year")["Value"].mean().rename("Yield")
nit_y = nit_df.groupby("Year")["Value"].mean().rename("Nitrogen")
land_y = land_df.groupby("Year")["Value"].mean().rename("ArableLand")

# 3. Merge on Year
data = pd.concat([yield_y, nit_y, land_y], axis=1, join="inner").dropna().reset_index()
print("\nMerged data shape:", data.shape)
print(data.head())
print("\nCorrelation:\n", data[["Nitrogen", "ArableLand", "Yield"]].corr().round(3))

X = data[["Nitrogen", "ArableLand"]]
y = data["Yield"]

# 4. Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 5. Pipeline: Min-Max scaling -> feature selection -> multiple linear regression
def make_pipeline():
    return Pipeline([
        ("scale", MinMaxScaler()),
        ("select", SelectKBest(score_func=f_regression, k=2)),
        ("reg", LinearRegression()),
    ])

pipe = make_pipeline().fit(X_train, y_train)

sel = pipe["select"]
print("\nFeature selection (f_regression):")
for name, f, p in zip(X.columns, sel.scores_, sel.pvalues_):
    print(f"  {name:11s} F={f:.3f}, p={p:.6f}")
print("Selected features:", list(X.columns[sel.get_support()]))
print("Coefficients (scaled):", dict(zip(X.columns[sel.get_support()], pipe["reg"].coef_.round(4))))
print("Intercept:", round(pipe["reg"].intercept_, 4))

# 6. Hold-out evaluation
y_pred = pipe.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("\n--- Hold-out (80/20) ---")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {np.sqrt(mse):.4f}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R2  : {r2_score(y_test, y_pred):.4f}")

# 7. K-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(
    make_pipeline(), X, y, cv=cv,
    scoring={"MSE": "neg_mean_squared_error",
             "RMSE": "neg_root_mean_squared_error",
             "MAE": "neg_mean_absolute_error",
             "R2": "r2"})

print("\n--- 5-Fold CV ---")
for k in ["MSE", "RMSE", "MAE"]:
    v = -scores[f"test_{k}"]
    print(f"{k:5s}: mean={v.mean():.4f}, std={v.std():.4f}")
v = scores["test_R2"]
print(f"R2   : mean={v.mean():.4f}, std={v.std():.4f}")

# 8. Plot: actual vs predicted (a 2-feature model can't be drawn as one line)
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.8, label="Test samples")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, color="red", linewidth=2, label="Perfect prediction")
plt.title("Multiple Linear Regression: Actual vs Predicted Yield")
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.legend()
plt.grid(True)
plt.show()