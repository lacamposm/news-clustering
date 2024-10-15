import umap
import pandas as pd
from sklearn.cluster import KMeans
from utils.llms import get_embed_dataframe
from utils.plots import (
    elbow_plot,
    silhouette_plot,
    gap_statistic,
    calinski_harabasz_plot,
    davies_bouldin_plot,
    tsne_plot_2d,
    umap_plot_2d
)


def test_umap_kmeans(model_embed_name: str = "nomic-embed-text"):
    """
    :param model_embed_name:
    :return:
    """
    news_embeddings = get_embed_dataframe(model_embed_name)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=10, random_state=42)
    reduced_news_embeddings = reducer.fit_transform(news_embeddings)
    df_reduced = pd.DataFrame(reduced_news_embeddings)

    # ====================================== Finding numbers of clusters ============================================ #
    elbow_plot(df_reduced, n=30, scaled_data=True)
    silhouette_plot(df_reduced, n=30, scaled_data=True)
    gap_statistic(df_reduced, max_clusters=30, scaled_data=True)
    calinski_harabasz_plot(df_reduced, n=30, scaled_data=True)
    davies_bouldin_plot(df_reduced, n=30, scaled_data=True)

    # ==================================================== 2 options ================================================ #
    # 10 clusters
    optimal_k = 10
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(df_reduced)

    tsne_plot_2d(news_embeddings, labels)
    umap_plot_2d(news_embeddings, labels)

    # 9 clusters
    optimal_k = 9
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(df_reduced)

    tsne_plot_2d(news_embeddings, labels)
    umap_plot_2d(news_embeddings, labels)
