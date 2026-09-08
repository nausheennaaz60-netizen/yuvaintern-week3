import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120

df = pd.read_csv('penguins_clustered.csv')
feature_cols = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']

# Use hierarchical clustering result (best ARI) as the final model for profiling
profile = df.groupby('cluster_hc')[feature_cols].mean().round(1)
sizes = df['cluster_hc'].value_counts().sort_index()
profile['n_penguins'] = sizes
# majority species per cluster, for labeling
majority_species = df.groupby('cluster_hc')['Species'].agg(lambda x: x.value_counts().idxmax())
profile['dominant_species'] = majority_species
print("=== FINAL CLUSTER PROFILES (Hierarchical, k=3) ===")
print(profile)

# Radar-style comparison: normalized cluster centers as a bar chart
centers_norm = (profile[feature_cols] - profile[feature_cols].mean()) / profile[feature_cols].std()

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(feature_cols))
width = 0.25
colors = ['#3D5A80', '#E07A5F', '#4C956C']
for i, (cluster_id, row) in enumerate(centers_norm.iterrows()):
    ax.bar(x + i*width, row.values, width, label=f'Cluster {cluster_id} (n={sizes[cluster_id]})', color=colors[i])
ax.set_xticks(x + width)
ax.set_xticklabels(feature_cols, rotation=10, ha='right', fontsize=9)
ax.set_ylabel('Standardized mean (z-score across clusters)')
ax.set_title('Cluster Feature Profiles (Hierarchical Clustering, k=3)', fontweight='bold')
ax.axhline(0, color='black', linewidth=0.8)
ax.legend()
plt.tight_layout()
plt.savefig('fig4_cluster_profiles.png')
plt.close()

print("\nSaved cluster profile figure.")
