import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Settings
LINKAGES = ["single", "complete", "average"]
N_CLUSTERS = 5

# 1. Load: only Annual Income and Spending Score
df = pd.read_csv("Mall_Customers.csv")
cols = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[cols].dropna()

# 2. Scale (z-score)
X_scaled = StandardScaler().fit_transform(X)

# 3. Bottom-up (agglomerative) clustering with sklearn, Euclidean distance
def wcss_of(labels):
    return sum(
        np.sum((X_scaled[labels == c] - X_scaled[labels == c].mean(axis=0)) ** 2)
        for c in np.unique(labels))

results = {}
for method in LINKAGES:
    labels = AgglomerativeClustering(
        n_clusters=N_CLUSTERS, linkage=method, metric="euclidean"
    ).fit_predict(X_scaled)
    results[method] = (labels, wcss_of(labels))

# 4. Comparison table
print("===== COMPARISON (K=%d, Euclidean) =====" % N_CLUSTERS)
print("Linkage     WCSS      Cluster sizes")
for m in LINKAGES:
    print(f"{m:10s} {results[m][1]:8.4f}  {np.bincount(results[m][0])}")
print("(K-means K=5 WCSS for reference: 65.5684)")

# 5. Best linkage = lowest WCSS
best = min(LINKAGES, key=lambda m: results[m][1])
best_labels, best_wcss = results[best]

print(f"\n===== BEST LINKAGE: {best.upper()} =====")
print(f"WCSS: {best_wcss:.4f}")
print("Cluster sizes:", np.bincount(best_labels))
means = df.loc[X.index, cols].assign(Cluster=best_labels).groupby("Cluster").mean()
print("Cluster means (original units):\n", means.round(2))

# 6. Dendrograms (scipy is used only for drawing the tree)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, method in zip(axes, LINKAGES):
    Z = linkage(X_scaled, method=method, metric="euclidean")
    dendrogram(Z, no_labels=True, ax=ax)
    h = Z[:, 2]
    cut = (h[-N_CLUSTERS] + h[-(N_CLUSTERS - 1)]) / 2
    ax.axhline(cut, color="red", linestyle="--", label=f"Cut for {N_CLUSTERS} clusters")
    title = f"{method} linkage" + ("  (BEST)" if method == best else "")
    ax.set(title=title, xlabel="Customers", ylabel="Merge distance")
    ax.legend()
plt.tight_layout()
plt.show()

# 7. Cluster scatter plots for all linkages
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, method in zip(axes, LINKAGES):
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=results[method][0], cmap="viridis", s=40)
    title = f"{method} linkage (WCSS={results[method][1]:.2f})"
    ax.set(title=title + ("  (BEST)" if method == best else ""),
           xlabel=cols[0], ylabel=cols[1])
plt.tight_layout()
plt.show()