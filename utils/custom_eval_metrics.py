from sklearn.metrics import silhouette_score, davies_bouldin_score


def silhouette_scorer(estimator, X):
    """
    Calcula el Silhouette Score utilizando la métrica de coseno para evaluar la calidad de los clústeres.

    El Silhouette Score mide cómo de similar es un punto de datos a otros puntos de su mismo clúster en comparación
    con los puntos de otros clústeres. El valor oscila entre -1 y 1, donde un valor más alto indica que los puntos
    están mejor agrupados y más separados de otros clústeres.

    :param estimator: Un modelo de clustering que debe tener un método `predict` para obtener las etiquetas de los
    clústeres de las muestras de `X`.
    :param X: Un array-like o matriz de datos (n_samples, n_features) sobre los cuales se calcularán los clústeres.
    :return: El Silhouette Score para el conjunto de datos `X` y las etiquetas de los clústeres obtenidos del
    `estimator`. Se utiliza la métrica de coseno para calcular las distancias.
    """
    labels = estimator.predict(X)
    return silhouette_score(X, labels, metric="cosine")


def davies_bouldin_scorer(estimator, X):
    """
    Calcula el índice de Davies-Bouldin para evaluar la calidad de los clústeres.

    El índice de Davies-Bouldin mide la media del cociente entre las distancias intra-clúster e inter-clúster para cada
    clúster. Un valor más bajo del índice indica una mejor formación de los clústeres, ya que minimiza la dispersión
    dentro de los clústeres y maximiza la separación entre ellos.

    :param estimator: Un modelo de clustering que debe tener un método `predict` para obtener las etiquetas de los
    clústeres de las muestras de `X`.
    :param X: Un array-like o matriz de datos (n_samples, n_features) sobre los cuales se calcularán los clústeres.
    :return: El valor negativo del índice de Davies-Bouldin para el conjunto de datos `X` y las etiquetas de los
    clústeres obtenidos del `estimator`. Se devuelve en negativo porque en este índice, un valor más bajo es mejor,
    y en algunas implementaciones de búsqueda de hiperparámetros, se maximiza la métrica.
    """
    labels = estimator.predict(X)
    return -davies_bouldin_score(X, labels)