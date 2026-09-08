import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120

df = pd.read_csv('penguins_cleaned.csv')
feature_cols = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
X = df[feature_cols].copy()

print("=== FEATURES USED ===")
print(X.describe().round(2))

# Standardize - essential for KMeans since Body Mass (g) has a much
# larger numeric range than the mm-scale measurements, which would
# dominate the Euclidean distance calculation otherwise
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\n=== SCALED FEATURE MEANS/STD (should be ~0 / ~1) ===")
print(pd.DataFrame(X_scaled, columns=feature_cols).describe().round(2).loc[['mean','std']])

# Elbow method + silhouette score to choose k
inertias = []
sil_scores = []
k_range = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

print("\n=== INERTIA AND SILHOUETTE BY K ===")
for k, inertia, sil in zip(k_range, inertias, sil_scores):
    print(f"k={k}: inertia={inertia:.1f}, silhouette={sil:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(list(k_range), inertias, marker='o', color='steelblue')
axes[0].set_xlabel('Number of clusters (k)')
axes[0].set_ylabel('Inertia (within-cluster SSE)')
axes[0].set_title('Elbow Method', fontweight='bold')
axes[0].axvline(3, color='red', linestyle='--', alpha=0.6, label='k=3')
axes[0].legend()

axes[1].plot(list(k_range), sil_scores, marker='o', color='seagreen')
axes[1].set_xlabel('Number of clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score by k', fontweight='bold')
axes[1].axvline(3, color='red', linestyle='--', alpha=0.6, label='k=3')
axes[1].legend()
plt.tight_layout()
plt.savefig('fig1_elbow_silhouette.png')
plt.close()

np.save('X_scaled.npy', X_scaled)
df.to_csv('penguins_for_clustering.csv', index=False)
print("\nSaved scaled features and figure.")
