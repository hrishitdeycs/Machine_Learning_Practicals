import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load: only Annual Income and Spending Score
df = pd.read_csv("Mall_Customers.csv")
cols = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[cols].dropna()

# 2. Scale (z-score)
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# 3. Elbow method: inertia (WCSS) for K = 1..10
print("K   Inertia(WCSS)")
ks = range(1, 11)
inertias = []
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"{k:<3d} {km.inertia_:12.4f}")

# 4. Final model: K = 5
# KMeans minimises inertia = sum of squared Euclidean distances to centroids
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X_scaled)
labels = kmeans.labels_

print("\n===== K-MEANS (K=5) =====")
print(f"Within-cluster SS (inertia): {kmeans.inertia_:.4f}")
print("Cluster sizes:", np.bincount(labels))

# Manual check of the objective: sum of squared Euclidean distances
manual_wcss = sum(
    np.sum((X_scaled[labels == c] - kmeans.cluster_centers_[c]) ** 2)
    for c in range(5)
)
print(f"Manual WCSS check          : {manual_wcss:.4f}")

# Centres back in original units (easier to interpret)
centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=cols)
print("\nCluster centres (original units):\n", centers.round(2))

# 5. Plots
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(list(ks), inertias, marker="o")
axes[0].axvline(5, color="red", linestyle="--")
axes[0].set(title="Elbow method", xlabel="K", ylabel="Inertia (WCSS)")

axes[1].scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap="viridis", s=50)
axes[1].scatter(centers.iloc[:, 0], centers.iloc[:, 1], c="red",
                marker="X", s=200, label="Centroids")
axes[1].set(title="K-Means (K=5)", xlabel=cols[0], ylabel=cols[1])
axes[1].legend()

plt.tight_layout()
plt.show()