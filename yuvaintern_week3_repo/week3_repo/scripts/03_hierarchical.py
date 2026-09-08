import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120

df = pd.read_csv('penguins_clustered.csv')
X_scaled = np.load('X_scaled.npy')

# Ward linkage minimizes within-cluster variance, analogous to KMeans' objective,
# which makes it a fair comparison method
Z = linkage(X_scaled, method='ward')

plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90, leaf_font_size=8, color_threshold=None)
plt.axhline(y=Z[-3, 2], color='red', linestyle='--', label='Cut for 3 clusters')
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)', fontweight='bold')
plt.xlabel('Sample clusters (truncated)')
plt.ylabel('Ward distance')
plt.legend()
plt.tight_layout()
plt.savefig('fig3_dendrogram.png')
plt.close()

# Cut the tree at 3 clusters and compare to KMeans k=3 and true species
hc_labels = fcluster(Z, t=3, criterion='maxclust')
df['cluster_hc'] = hc_labels

ari_hc = adjusted_rand_score(df['Species'], hc_labels)
ari_vs_kmeans = adjusted_rand_score(df['cluster_k3'], hc_labels)
sil_hc = silhouette_score(X_scaled, hc_labels)

print(f"Hierarchical (3 clusters) -- Silhouette: {sil_hc:.3f}, ARI vs true species: {ari_hc:.3f}")
print(f"Agreement between Hierarchical and K-Means (k=3) labelings: ARI={ari_vs_kmeans:.3f}")

print("\n=== CROSSTAB: Hierarchical clusters vs true species ===")
print(pd.crosstab(df['cluster_hc'], df['Species']).to_string())

df.to_csv('penguins_clustered.csv', index=False)
