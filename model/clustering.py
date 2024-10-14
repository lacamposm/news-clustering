import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.manifold import TSNE
from utils.llms import get_embed_dataframe

df_embed_summary = get_embed_dataframe("mxbai-embed-large:335m")

# ============================================= T-SNE ========================================================= #
normalized_embeddings = normalize(df_embed_summary)
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(normalized_embeddings)
plt.figure(figsize=(10, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=5, alpha=0.7)
plt.title("t-SNE visualization of normalized embeddings")
plt.xlabel("t-SNE component 1")
plt.ylabel("t-SNE component 2")
plt.show()

# ============================================= k-means ========================================================== #
df_embed_summary = get_embed_dataframe("mxbai-embed-large:335m")
scaled_data = StandardScaler().fit_transform(df_embed_summary)
pca = PCA(n_components=0.90)
pca.fit(scaled_data)
pd.DataFrame(pca.explained_variance_ratio_).sum()
kmeans = KMeans(n_clusters=5, random_state=42).fit(pca.transform(scaled_data))
df_embed_summary["cluster"] = kmeans.labels_
print(df_embed_summary["cluster"].value_counts())
df_embed_summary.head()


# ============================================= HDBSCAN ========================================================== #
df_embed_summary = get_embed_dataframe("mxbai-embed-large:335m")
normalized_embeddings = normalize(df_embed_summary)
cluster_hdbscan = HDBSCAN(min_cluster_size=10, metric="cosine")
cluster_hdbscan.fit(normalized_embeddings)
df_embed_summary["cluster"] = cluster_hdbscan.labels_
print(df_embed_summary["cluster"].value_counts())
df_embed_summary.head()
