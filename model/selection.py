import logging
from pprint import pformat
from umap import UMAP
from utils.llms import get_embed_summary_df
from utils.custom_eval_metrics import silhouette_scorer, davies_bouldin_scorer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def build_param_grid_search(scorer="silhouette", n_iter=10, random_state=42):
    """
    Construye una cuadrícula de parámetros para la selección de modelos utilizando RandomizedSearchCV.

    Esta función construye una cuadrícula de parámetros para diferentes combinaciones de técnicas de preprocesamiento,
    métodos de reducción de dimensionalidad y algoritmos de agrupamiento. La cuadrícula resultante se utiliza en una
    búsqueda aleatoria para encontrar los mejores parámetros del modelo según la métrica de puntuación especificada.

    :param scorer: La métrica de puntuación para evaluar los modelos. Las opciones son "silhouette" para la puntuación
    de silueta o "davies_bouldin" para la puntuación de Davies-Bouldin.
    :param n_iter: El número de iteraciones para la búsqueda aleatoria. Esto controla cuántas combinaciones diferentes
    de parámetros se probarán.
    :param random_state: Semilla utilizada por el generador de números aleatorios para la reproducibilidad.
    :return: Un RandomizedSearchCV configurado con la cuadrícula de parámetros especificada y la métrica de puntuación.
    """
    scoring_dict = {
        "silhouette": silhouette_scorer,
        "davies-bouldin": davies_bouldin_scorer,
    }

    if isinstance(scorer, list):
        scoring_dict = {score: scoring_dict[score] for score in scorer if score in scoring_dict}
        if not scoring_dict:
            raise ValueError(f"Scorer: {scorer}, no esta implementado.")
        refit_score = "silhouette"

    elif isinstance(scorer, str):
        if scorer not in scoring_dict:
            raise ValueError(f"Scorer: {scorer}, no está implementado.")
        scoring_dict = {scorer: scoring_dict[scorer]}
        refit_score = scorer

    else:
        raise ValueError("El scorer debe ser una cadena o una lista de cadenas.")

    list_scalers = [StandardScaler(), MinMaxScaler(), RobustScaler()]

    pca_n_components = [num for num in range(5, 80)]
    umap_metrics = ["cosine", "correlation"]
    umap_n_neighbors = [5, 10, 15]
    umap_min_dist = [0.1, 0.3, 0.5]
    umap_n_components = [2, 3, 4]

    gauss_mix_n_components = [num for num in range(5, 13)]
    gauss_mix_covariance_type = ["full", "diag"]
    kmeans_n_clusters = [num for num in range(5, 21)]

    param_grid = [
        {
            "preprocessor": list_scalers,
            "dim_reduction": [PCA()],
            "dim_reduction__n_components": pca_n_components,
            "clusterer": [KMeans()],
            "clusterer__n_clusters": kmeans_n_clusters,
        },
        {
            "preprocessor": list_scalers,
            "dim_reduction": [PCA()],
            "dim_reduction__n_components": pca_n_components,
            "clusterer": [GaussianMixture()],
            "clusterer__n_components": gauss_mix_n_components,
            "clusterer__covariance_type": gauss_mix_covariance_type
        },
        {
            "preprocessor": list_scalers,
            "dim_reduction": [UMAP()],
            "dim_reduction__n_neighbors": umap_n_neighbors,
            "dim_reduction__min_dist": umap_min_dist,
            "dim_reduction__n_components": umap_n_components,
            "dim_reduction__metric": umap_metrics,
            "clusterer": [KMeans()],
            "clusterer__n_clusters": kmeans_n_clusters,
        },
        {
            "preprocessor": list_scalers,
            "dim_reduction": [UMAP()],
            "dim_reduction__n_neighbors": umap_n_neighbors,
            "dim_reduction__min_dist": umap_min_dist,
            "dim_reduction__n_components": umap_n_components,
            "dim_reduction__metric": umap_metrics,
            "clusterer": [GaussianMixture()],
            "clusterer__n_components": gauss_mix_n_components,
            "clusterer__covariance_type": gauss_mix_covariance_type
        }
    ]

    pipeline = Pipeline([
        ("preprocessor", "passthrough"),
        ("dim_reduction", "passthrough"),
        ("clusterer", "passthrough")
    ])

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        scoring=scoring_dict,
        cv=2,
        n_jobs=-1,
        verbose=3,
        n_iter=n_iter,
        refit=refit_score,
        random_state=random_state
    )

    return random_search


def random_search_best_model(embed_model_name="mxbai-embed-large:335m", scorer="silhouette", n_iter=10,
                             random_state=42):
    """
    Entrena y evalúa el mejor modelo de agrupamiento utilizando el modelo de embeddings especificado.

    Esta función recupera un DataFrame de embeddings utilizando el modelo de embeddings especificado y realiza
    una búsqueda aleatoria para la sintonización de hiperparámetros de diferentes algoritmos de agrupamiento y
    técnicas de reducción de dimensionalidad. Registra los mejores parámetros y la puntuación alcanzada durante la
    búsqueda.

    :param embed_model_name: El nombre del modelo de embeddings que se utilizará para generar el DataFrame.
    :param scorer: La métrica de puntuación utilizada para evaluar los modelos durante la búsqueda aleatoria. Las
    opciones son "silhouette" o "davies_bouldin".
    :param n_iter: El número de iteraciones para la búsqueda aleatoria, controlando cuántas combinaciones diferentes de
    parámetros se probarán.
    :param random_state: Semilla utilizada por el generador de números aleatorios para la reproducibilidad.
    :return: El mejor modelo de RandomizedSearchCV ajustado al DataFrame de embeddings.
    """
    random_search = build_param_grid_search(scorer, n_iter, random_state)
    df_embed = get_embed_summary_df(embed_model_name)
    random_search_best_model_fit = random_search.fit(df_embed)

    logging.info(f"Best params:\n{pformat(random_search_best_model_fit.best_params_)}\n", )
    logging.info(f"Best Scorer-{scorer} ----> {random_search_best_model_fit.best_score_:.5f}", )

    return random_search_best_model_fit
