import os
import requests
import pandas as pd
from io import StringIO
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_df_from_sitemap_el_tiempo():
    """
    Obtiene un DataFrame con información de noticias del sitemap de El Tiempo.

    Esta función realiza una solicitud GET al sitemap de El Tiempo, parsea el XML
    resultante y extrae información relevante sobre las noticias, incluyendo URLs,
    títulos, fechas y palabras clave.

    :return: DataFrame de pandas con la información extraída del sitemap.
    :rtype: pd.DataFrame

    :raises ValueError: Si hay un problema con el status_code de la solicitud al sitemap.

    Example:
        >>> df = get_df_from_sitemap_el_tiempo()
        >>> print(df.columns)
        Index(['url_page', 'title', 'publication_date', 'keywords'], dtype='object')
    """
    url_sitemap_el_tiempo = "https://www.eltiempo.com/sitemap-articles-current.xml"
    session = requests.Session()
    response = session.get(url_sitemap_el_tiempo)
    if response.raise_for_status:
        print(f"OK, with the sitemap: {url_sitemap_el_tiempo}. Status Code: {response.status_code}\n")
    else:
        ValueError(f"Problemas con el status_code: {response.status_code}")

    namespaces = {
        "ns": "http://www.sitemaps.org/schemas/sitemap/0.9",
        "news": "http://www.google.com/schemas/sitemap-news/0.9",
        "video": "http://www.google.com/schemas/sitemap-video/1.1"
    }
    
    df_urls_news_el_tiempo = pd.read_xml(StringIO(response.text), xpath=".//ns:url", namespaces=namespaces)
    df_title_date_keywords_news = pd.read_xml(StringIO(response.text), xpath=".//news:news", namespaces=namespaces)
    
    df_urls_news_el_tiempo = (
        df_urls_news_el_tiempo[["loc"]]
        .merge(
            df_title_date_keywords_news.drop(columns=["publication"]),
            left_index=True,
            right_index=True
        )
        .rename(columns={"loc": "url_page"})
        .drop_duplicates(subset=["url_page"])
    )
    
    return df_urls_news_el_tiempo


def get_content_news_from_url(url: str) -> Dict:
    """
    Obtiene el contenido de una noticia desde una URL específica.

    Esta función realiza una solicitud GET a la URL proporcionada, parsea el contenido HTML
    y extrae información específica como el autor y el cuerpo de la noticia.

    :param url: URL de la página de la noticia.
    :type url: str
    :return: Diccionario con la URL, autor y contenido de la noticia.
    :rtype: Dict

    :raises ValueError: Si hay un problema con el status_code de la solicitud.

    Example:
        >>> news_content = get_content_news_from_url("https://www.eltiempo.com/example-news")
        >>> print(news_content.keys())
        dict_keys(['url_page', 'autor', 'news_content'])
    """
    session = requests.Session()
    request = session.get(url)

    if request.raise_for_status:
        soup = BeautifulSoup(request.content, "html.parser")
        author_tag = soup.find("a", class_="c-detail__author__name").get_text()
        content = " ".join([tag.get_text() for tag in soup.find_all("div", class_="paragraph")])

        return {
            "url_page": url,
            "autor": author_tag,
            "news_content": content
        }
    else:
        ValueError(f"Problems with status_code: {request.status_code}")


def get_df_news_by_parallel_process_urls(urls: List[str], n_jobs=4) -> pd.DataFrame:
    """
    Procesa múltiples URLs en paralelo para extraer contenido de noticias.

    Esta función utiliza ThreadPoolExecutor para procesar concurrentemente múltiples URLs,
    extrayendo el contenido de cada noticia. Es útil para acelerar la extracción de datos
    de múltiples páginas web.

    :param urls: Lista de URLs a procesar.
    :type urls: List[str]
    :param n_jobs: Número de núcleos a utilizar para el procesamiento paralelo. Por defecto es 4.
    :type n_jobs: int
    :return: DataFrame con la información extraída de cada URL.
    :rtype: pd.DataFrame

    Example:
        >>> urls = ["https://www.eltiempo.com/news1", "https://www.eltiempo.com/news2"]
        >>> df = get_df_news_by_parallel_process_urls(urls)
        >>> print(df.shape)
        (2, 3)
    """
    cpus = os.cpu_count()
    print(f"Número de cores disponibles: {cpus}")
    if n_jobs == -1:
        n_jobs = cpus
    results = []

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(get_content_news_from_url, url) for url in urls]

        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error in: {future}\nWith error: {e}")

    return pd.DataFrame(results)


def get_scraper_df_news_el_tiempo(path_file: str = "data/bronze/news_data.parquet", n_jobs: int = 4) -> pd.DataFrame:
    """
    Obtiene un DataFrame de noticias listo para ser resumido.

    Esta función verifica si existe un archivo de datos previamente guardado. Si existe,
    lo carga; si no, obtiene nuevos datos del sitemap de El Tiempo, extrae el contenido
    de las noticias en paralelo, y guarda el resultado en un archivo Parquet.

    :param path_file: Ruta del archivo Parquet donde se guardarán o de donde se cargarán los datos.
    :type path_file: str
    :param n_jobs: Número de núcleos a utilizar para el procesamiento paralelo. Por defecto es 4.
    :type n_jobs: int
    :return: DataFrame con información completa de las noticias, incluyendo URL, contenido, autor, etc.
    :rtype: pd.DataFrame

    Example:
        >>> df = get_scraper_df_news_el_tiempo("data/bronze/news_data.parquet")
        >>> print(df.columns)
        Index(['url_page', 'title', 'publication_date', 'keywords', 'autor', 'news_content'], dtype='object')
    """
    if not Path(path_file).exists():
        df_sitemap = get_df_from_sitemap_el_tiempo()
        df_news_content = get_df_news_by_parallel_process_urls(df_sitemap["url_page"].to_list(), n_jobs)
        df_sitemap.merge(df_news_content, on=["url_page"]).to_parquet(path_file)
        return df_sitemap.merge(df_news_content, on=["url_page"])

    else:
        return pd.read_parquet(path_file)
