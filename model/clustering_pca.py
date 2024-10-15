import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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


def test_pca_kmeans(model_embed_name: str = "nomic-embed-text"):
    """
    :return:
    """
    news_embeddings = get_embed_dataframe(model_embed_name)
    scaled_data = StandardScaler().fit_transform(news_embeddings)
    pca = PCA(n_components=0.90)
    pca.fit(scaled_data)
    df_reduced = pd.DataFrame(scaled_data)

    # ====================================== Finding numbers of clusters ============================================ #
    elbow_plot(df_reduced, n=30, scaled_data=True)
    silhouette_plot(df_reduced, n=25, scaled_data=True)
    gap_statistic(df_reduced, max_clusters=25, nrefs=5, scaled_data=True)
    calinski_harabasz_plot(df_reduced, n=30, scaled_data=True)
    davies_bouldin_plot(df_reduced, n=30, scaled_data=True)

    # ============================================ Seleccion del numero de clusters ================================= #
    kmeans = KMeans(n_clusters=12, random_state=42)
    labels = kmeans.fit_predict(df_reduced)

    tsne_plot_2d(news_embeddings, labels)
    umap_plot_2d(news_embeddings, labels)
