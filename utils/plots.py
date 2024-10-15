import umap
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

warnings.filterwarnings("ignore")


def elbow_plot(df: pd.DataFrame, n: int = 2, scaled_data: bool = True):
    """
    Función que plotea el número de clusters vs WCSS para determinar el número de clusters óptimo.

    :param df: pd.DataFrame con la data original.
    :param n: int, opcional. Número máximo de clusters a comparar. Default es 2.
    :param scaled_data: bool, opcional. Indica si los datos ya están escalados. Default es True.
    :raises ValueError: Si el DataFrame contiene valores NaN o si no se puede calcular KMeans.
    :return: None

    **Nota: Si scaled_data es False, los datos se escalan utilizando StandardScaler.**
    """
    if df.isnull().values.any():
        raise ValueError("El DataFrame contiene valores NaN. KMeans no funcionará correctamente.")

    if not scaled_data:
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
    else:
        df_scaled = df

    try:
        inertias = [KMeans(n_clusters=k, random_state=42).fit(df_scaled).inertia_ for k in range(1, n + 1)]
    except Exception as e:
        raise ValueError(f"No se pudo calcular KMeans: {e}")

    data_plot = pd.DataFrame({"n_clusters": range(1, n + 1), "WCSS": inertias})

    (
        px.line(
            data_plot,
            x="n_clusters",
            y="WCSS",
            title="Finding optimal number of clusters",
            template="plotly_white",
            markers=True
        )
        .update_xaxes(title_text="Number of clusters")
        .update_yaxes(title_text="WCSS")
    ).show()


def silhouette_plot(df: pd.DataFrame, n: int = 2, scaled_data: bool = True):
    """
    Función que plotea el coeficiente de silueta para diferentes valores de k en el algoritmo KMeans.

    :param df: pd.DataFrame con la data original.
    :param n: int, opcional. Número máximo de clusters a comparar. Default es 2.
    :param scaled_data: bool, opcional. Indica si los datos ya están escalados. Default es True.
    :raises ValueError: Si el DataFrame contiene valores NaN o si no se puede calcular KMeans.
    :return: None

    **Nota: Si scaled_data es False, los datos se escalan utilizando StandardScaler.**
    """
    if df.isnull().values.any():
        raise ValueError("El DataFrame contiene valores NaN. KMeans no funcionará correctamente.")

    if not scaled_data:
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
    else:
        df_scaled = df

    try:
        models = [KMeans(n_clusters=k, random_state=42).fit(df_scaled) for k in range(2, n + 1)]
        silhouette_scores = [silhouette_score(df_scaled, model.labels_) for model in models]
    except Exception as e:
        raise ValueError(f"No se pudo calcular KMeans: {e}")

    data_plot = pd.DataFrame({"n_clusters": range(2, n + 1), "Silhouette_score": silhouette_scores})
    (
        px.line(
            data_plot,
            x="n_clusters",
            y="Silhouette_score",
            title="Silhouette method",
            template="plotly_white",
            markers=True
        )
        .update_xaxes(title_text="Number of clusters")
        .update_yaxes(title_text="Silhouette score")
    ).show()


def gap_statistic(df: pd.DataFrame, nrefs=10, max_clusters=10, scaled_data: bool = True, random_state=42):
    """
    Calcula el número óptimo de clusters (K) usando el método de la estadística Gap y muestra el gráfico.

    :param df: pd.DataFrame, los datos originales en formato DataFrame.
    :param nrefs: int, opcional. Número de conjuntos de referencia aleatorios que se crearán para la comparación.
           El valor por defecto es 3.
    :param max_clusters: int, opcional. Número máximo de clusters a probar. El valor por defecto es 10.
    :param scaled_data: bool, opcional. Si es True, se escalan los datos usando StandardScaler. Default es True.
    :param random_state: int, opcional. Semilla aleatoria para garantizar consistencia. Default es 42.
    :raises ValueError: Si el DataFrame contiene valores NaN o si ocurre un error con KMeans.
    :return: Muestra el gráfico interactivo con la estadística Gap.

    **Nota**: Si scaled_data es False, se utilizarán los datos tal como están.
    """
    if df.isnull().values.any():
        raise ValueError("El DataFrame contiene valores NaN. KMeans no funcionará correctamente.")

    if not scaled_data:
        scaler = StandardScaler()
        data = scaler.fit_transform(df)
    else:
        data = df.values

    np.random.seed(random_state)

    gaps = np.zeros((len(range(1, max_clusters)),))
    results = []

    for gap_index, k in enumerate(range(1, max_clusters)):

        ref_disps = np.zeros(nrefs)
        for i in range(nrefs):
            random_reference = np.random.random_sample(size=data.shape)

            km = KMeans(n_clusters=k, random_state=random_state)
            km.fit(random_reference)
            ref_disp = km.inertia_
            ref_disps[i] = ref_disp

        km = KMeans(n_clusters=k, random_state=random_state)
        km.fit(data)
        orig_disp = km.inertia_

        gap = np.mean(np.log(ref_disps)) - np.log(orig_disp)
        gaps[gap_index] = gap
        results.append({'clusterCount': k, 'gap': gap})

    results_df = pd.DataFrame(results)

    fig = px.line(
        results_df,
        x="clusterCount",
        y="gap",
        title="Gap Statistic Method",
        markers=True,
        template="plotly_white"
    )

    fig.update_xaxes(title_text="Number of clusters $K$")
    fig.update_yaxes(title_text="Gap Statistic")
    fig.show()


def calinski_harabasz_plot(df: pd.DataFrame, n: int = 2, scaled_data: bool = True):
    """
    Función que plotea el número de clusters vs el índice de Calinski-Harabasz para determinar el número de clusters
    óptimo.

    :param df: pd.DataFrame con la data original.
    :param n: int, opcional. Número máximo de clusters a comparar. Default es 2.
    :param scaled_data: bool, opcional. Indica si los datos ya están escalados. Default es True.
    :raises ValueError: Si el DataFrame contiene valores NaN o si no se puede calcular KMeans.
    :return: None

    **Nota: Si scaled_data es False, los datos se escalan utilizando StandardScaler.**
    """
    if df.isnull().values.any():
        raise ValueError("El DataFrame contiene valores NaN. Calinski-Harabasz no funcionará correctamente.")

    if not scaled_data:
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
    else:
        df_scaled = df

    try:
        calinski_scores = [calinski_harabasz_score(df_scaled, KMeans(n_clusters=k, random_state=42).fit_predict(df_scaled))
                           for k in range(2, n + 1)]
    except Exception as e:
        raise ValueError(f"No se pudo calcular KMeans o Calinski-Harabasz: {e}")

    data_plot = pd.DataFrame({"n_clusters": range(2, n + 1), "Calinski-Harabasz": calinski_scores})

    (
        px.line(
            data_plot,
            x="n_clusters",
            y="Calinski-Harabasz",
            title="Calinski-Harabasz Index vs Number of Clusters",
            template="plotly_white",
            markers=True
        )
        .update_xaxes(title_text="Number of clusters")
        .update_yaxes(title_text="Calinski-Harabasz Index")
    ).show()


def davies_bouldin_plot(df: pd.DataFrame, n: int = 2, scaled_data: bool = True):
    """
    Función que plotea el número de clusters vs el índice de Davies-Bouldin para determinar el número de clusters óptimo.

    :param df: pd.DataFrame con la data original.
    :param n: int, opcional. Número máximo de clusters a comparar. Default es 2.
    :param scaled_data: bool, opcional. Indica si los datos ya están escalados. Default es True.
    :raises ValueError: Si el DataFrame contiene valores NaN o si no se puede calcular KMeans.
    :return: None

    **Nota: Si scaled_data es False, los datos se escalan utilizando StandardScaler.**
    """
    if df.isnull().values.any():
        raise ValueError("El DataFrame contiene valores NaN. Davies-Bouldin no funcionará correctamente.")

    if not scaled_data:
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
    else:
        df_scaled = df

    try:
        davies_scores = [davies_bouldin_score(df_scaled, KMeans(n_clusters=k, random_state=42).fit_predict(df_scaled))
                         for k in range(2, n + 1)]
    except Exception as e:
        raise ValueError(f"No se pudo calcular KMeans o Davies-Bouldin: {e}")

    data_plot = pd.DataFrame({"n_clusters": range(2, n + 1), "Davies-Bouldin": davies_scores})

    (
        px.line(
            data_plot,
            x="n_clusters",
            y="Davies-Bouldin",
            title="Davies-Bouldin Index vs Number of Clusters",
            template="plotly_white",
            markers=True
        )
        .update_xaxes(title_text="Number of clusters")
        .update_yaxes(title_text="Davies-Bouldin Index")
    ).show()


def tsne_plot_2d(df: pd.DataFrame, cluster_label):
    """
    :param df:
    :param cluster_label:
    :return:
    """
    normalized_embeddings = normalize(df)
    tsne = TSNE(n_components=2, random_state=42)
    df_plot = pd.DataFrame(tsne.fit_transform(normalized_embeddings), columns=["tSNE1", "tSNE2"])
    df_plot["cluster"] = cluster_label + 1
    df_plot = df_plot.sort_values(by=["cluster"])
    df_plot["cluster"] = df_plot["cluster"].astype("string")

    (
        px.scatter(
            df_plot, x="tSNE1", y="tSNE2", color="cluster",
            title=f"Low dimension with TSNE",
            opacity=0.8,
            color_discrete_sequence=px.colors.qualitative.Dark24,
            template="plotly_white"
        )
        .update_traces(marker=dict(size=3))
        .show()
    )


def umap_plot_2d(df: pd.DataFrame, cluster_label):
    """
    Función que genera un gráfico en 2D utilizando UMAP y muestra los clústeres.

    :param df: pd.DataFrame con los datos originales (embeddings).
    :param cluster_label: Etiquetas de los clústeres para colorear los puntos.
    :return: None
    """
    reducer = umap.UMAP(n_components=2, random_state=42)
    df_plot = pd.DataFrame(reducer.fit_transform(df), columns=["UMAP1", "UMAP2"])

    df_plot["cluster"] = cluster_label + 1
    df_plot = df_plot.sort_values(by=["cluster"])
    df_plot["cluster"] = df_plot["cluster"].astype("string")

    (
        px.scatter(
            df_plot, x="UMAP1", y="UMAP2", color="cluster",
            title=f"Low dimension with UMAP",
            opacity=0.8,
            color_discrete_sequence=px.colors.qualitative.Dark24,
            template="plotly_white"
        )
        .update_traces(marker=dict(size=3))
        .show()
    )
