import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

# 1. Load: only Year and Value
df = pd.read_csv("FAOSTAT_data_en_8-4-2026.csv")
df = df[["Year", "Value"]].dropna().sort_values("Year")
X = df[["Year"]]
y = df["Value"]

# 2. Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 3. Pipeline: Min-Max scaling -> degree-2 terms -> feature selection -> regression
def make_pipeline():
    return Pipeline([
        ("scale", MinMaxScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),  # Year, Year^2
        ("select", SelectKBest(score_func=f_regression, k="all")),
        ("reg", LinearRegression()),
    ])

pipe = make_pipeline().fit(X_train, y_train)

# Feature selection scores
names = pipe["poly"].get_feature_names_out(["Year"])
sel = pipe["select"]
print("Feature selection (f_regression):")
for n, f, p in zip(names, sel.scores_, sel.pvalues_):
    print(f"  {n:7s} F={f:.3f}, p={p:.6f}")
print("Coefficients (scaled):", dict(zip(names, pipe["reg"].coef_.round(4))))
print("Intercept:", round(pipe["reg"].intercept_, 4))

# 4. Hold-out evaluation
y_pred = pipe.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("\n--- Hold-out (80/20) ---")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {np.sqrt(mse):.4f}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R2  : {r2_score(y_test, y_pred):.4f}")

# 5. K-fold cross-validation
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

# 6. Future predictions
future = pd.DataFrame({"Year": [2027, 2028, 2029, 2030]})
future_pred = pipe.predict(future)
print("\nPredicted yield:")
for yr, p in zip(future["Year"], future_pred):
    print(f"{yr}: {p:.4f}")

# 7. Plot
curve_years = pd.DataFrame({"Year": np.arange(X["Year"].min(), 2031)})
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color="blue", alpha=0.6, label="Training data")
plt.scatter(X_test, y_test, color="orange", alpha=0.8, label="Test data")
plt.plot(curve_years["Year"], pipe.predict(curve_years), color="red",
         linewidth=2, label="Polynomial fit (degree 2)")
plt.scatter(future["Year"], future_pred, color="green", marker="*", s=180,
            label="Predicted (2027-2030)")
plt.title("Wheat Yield Prediction using Polynomial Regression (Degree 2)")
plt.xlabel("Year")
plt.ylabel("Yield (Value)")
plt.legend()
plt.grid(True)
plt.show()