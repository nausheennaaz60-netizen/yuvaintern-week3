import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120

df = pd.read_csv('penguins_for_clustering.csv')
X_scaled = np.load('X_scaled.npy')
feature_cols = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']

# PCA for 2D visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("=== PCA EXPLAINED VARIANCE ===")
print(pca.explained_variance_ratio_, "total:", pca.explained_variance_ratio_.sum())

# --- KMeans k=2 ---
km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_k2 = km2.fit_predict(X_scaled)
df['cluster_k2'] = labels_k2

# --- KMeans k=3 ---
km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_k3 = km3.fit_predict(X_scaled)
df['cluster_k3'] = labels_k3

# Compare against true species (ground truth known, but NOT used in clustering)
ari_k2 = adjusted_rand_score(df['Species'], labels_k2)
ari_k3 = adjusted_rand_score(df['Species'], labels_k3)
print(f"\nAdjusted Rand Index vs true species -- k=2: {ari_k2:.3f}, k=3: {ari_k3:.3f}")

print("\n=== CROSSTAB: k=2 clusters vs true species ===")
print(pd.crosstab(df['cluster_k2'], df['Species']))
print("\n=== CROSSTAB: k=3 clusters vs true species ===")
print(pd.crosstab(df['cluster_k3'], df['Species']))

# --- Visualize k=2 and k=3 on PCA plane, plus true species for comparison ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sc0 = axes[0].scatter(X_pca[:,0], X_pca[:,1], c=labels_k2, cmap='Set1', alpha=0.7, s=35)
axes[0].set_title(f'K-Means (k=2)\nSilhouette={silhouette_score(X_scaled, labels_k2):.3f}', fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')

sc1 = axes[1].scatter(X_pca[:,0], X_pca[:,1], c=labels_k3, cmap='Set1', alpha=0.7, s=35)
axes[1].set_title(f'K-Means (k=3)\nSilhouette={silhouette_score(X_scaled, labels_k3):.3f}', fontweight='bold')
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')

species_codes = df['Species'].astype('category').cat.codes
sc2 = axes[2].scatter(X_pca[:,0], X_pca[:,1], c=species_codes, cmap='Set1', alpha=0.7, s=35)
axes[2].set_title('True Species Labels\n(ground truth, not used in clustering)', fontweight='bold')
axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')

plt.tight_layout()
plt.savefig('fig2_pca_comparison.png')
plt.close()

df.to_csv('penguins_clustered.csv', index=False)
print("\nSaved clustered dataset and PCA comparison figure.")
