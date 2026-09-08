# Week 3 — Unsupervised Learning & Clustering Analysis

Task submitted for the **YuvaIntern Virtual Data Science with Python Trainee** internship (Week 3).

## Objective
Apply unsupervised clustering techniques to segment a public dataset into meaningful groups
and analyze each cluster's characteristics, using Scikit-learn.

## Dataset
Continues from [Week 1](../yuvaintern-week1-data-cleaning) and
[Week 2](../yuvaintern-week2-eda) — the Palmer Archipelago Penguins dataset. Chosen specifically
because its true species labels can be withheld from clustering and used afterward to validate
the results — something not usually possible in real unsupervised learning problems.

## Repo structure
```
├── Week3_Clustering_Report.docx     # Full report: methodology, results, interpretation
├── data/
│   └── penguins_clustered.csv       # Final dataset with all cluster assignments
├── scripts/
│   ├── 01_preprocess_and_optimalk.py  # Feature scaling, elbow method, silhouette scores
│   ├── 02_kmeans_clustering.py        # K-Means (k=2, k=3), PCA visualization, ARI validation
│   ├── 03_hierarchical.py             # Ward-linkage hierarchical clustering + dendrogram
│   └── 04_cluster_profiles.py         # Final cluster characterization
└── figures/                          # All generated chart images
```

## Key findings
- **Two clustering algorithms compared**: K-Means and Agglomerative (Ward-linkage) Hierarchical
  Clustering, both applied to standardized physical measurements only (species labels withheld).
- **Internal metric vs. ground truth disagreement**: silhouette score preferred k=2, but the
  Adjusted Rand Index against true species showed k=3 was substantially more accurate
  (0.793 vs 0.655) — a reminder that internal clustering metrics measure geometric separation,
  not real-world meaning.
- **Hierarchical clustering outperformed K-Means**: at k=3, Ward-linkage hierarchical clustering
  achieved ARI = 0.916 vs K-Means' 0.793, recovering the species structure almost perfectly
  (Gentoo and Chinstrap clusters were 100% pure; the only confusion was 11 Chinstrap penguins
  grouped with Adelie).
- Full cluster profiling and research implications are in the report.

## How to run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
cd scripts
python 01_preprocess_and_optimalk.py
python 02_kmeans_clustering.py
python 03_hierarchical.py
python 04_cluster_profiles.py
```

## Full report
See [`Week3_Clustering_Report.docx`](./Week3_Clustering_Report.docx) for the complete write-up.
