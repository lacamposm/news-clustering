import os
import json
import tiktoken
import pandas as pd
from pathlib import Path
from typing import List, Union
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


def get_count_tokens(sentence: str) -> int:
    """
    Cuenta el número de tokens en una oración dada utilizando el modelo de tokenización "gpt2".

    Esta función utiliza el modelo de tokenización GPT-2 para dividir la oración de entrada
    en tokens y contar cuántos tokens se generan. Es útil para estimar la longitud de un texto
    en términos de tokens, lo cual es relevante para modelos de lenguaje que tienen límites
    de tokens en su entrada.

    :param sentence: La oración o texto para el cual se quiere contar los tokens.
    :type sentence: str
    :return: El número de tokens en la oración de entrada.
    :rtype: int

    Example:
        >>> get_count_tokens("Hola, ¿cómo estás?")
        5
    """
    encoding = tiktoken.encoding_for_model("gpt2")
    return len(encoding.encode(sentence))


def get_embed_text(text: Union[str, List[str]], embed_model_name: str = "nomic-embed-text") -> List[List[float]]:
    """
    Genera embeddings para un texto o lista de textos utilizando el modelo de embeddings especificado.

    Esta función toma un texto o una lista de textos y utiliza un modelo de embeddings
    para generar representaciones vectoriales de estos textos. Los embeddings son útiles
    para diversas tareas de procesamiento de lenguaje natural, como búsqueda semántica
    o clustering de textos.

    :param text: Un string o una lista de strings para los cuales se generarán embeddings.
    :type text: Union[str, List[str]]
    :param embed_model_name: Nombre del modelo de embeddings a utilizar.
    :type embed_model_name: str
    :return: Una lista de embeddings, donde cada embedding es una lista de números flotantes.
    :rtype: List[List[float]]

    Example:
        >>> get_embed_text("Hola mundo")
        [[0.1, 0.2, 0.3, ...]] # (los valores reales dependerán del modelo utilizado)
    """
    if isinstance(text, str):
        text = [text]
    return OllamaEmbeddings(model=embed_model_name).embed_documents(text)


def get_summary_news(news: str, model_name: str = "llama3.2:3b") -> str:
    """
    Genera un resumen de una noticia utilizando un modelo de lenguaje especificado.

    Esta función toma el texto de una noticia y utiliza un modelo de lenguaje para generar
    un resumen conciso y preciso. El resumen se genera siguiendo pautas específicas definidas
    en el prompt del sistema, que incluyen capturar el evento principal, los actores clave,
    las consecuencias y el contexto relevante.

    :param news: El texto completo de la noticia a resumir.
    :type news: str
    :param model_name: El nombre del modelo de lenguaje a utilizar para la generación del resumen.
    :type model_name: str
    :return: Un resumen conciso de la noticia.
    :rtype: str

    Example:
        >>> news = "Texto largo de una noticia..."
        >>> get_summary_news(news)
        'Resumen conciso de la noticia...'
    """
    llm_ollama = ChatOllama(
        model=model_name,
        temperature=0.1,
        num_predict=4096,
        num_ctx=16000,
        keep_alive=0
    )

    prompt_summarization = ChatPromptTemplate(
        [
            SystemMessagePromptTemplate.from_template(
                """Por favor, crea un resumen natural, claro y preciso de esta noticia. Asegúrate de incluir los 
                siguientes puntos de manera fluida::"""
                """1. Evento principal o tema central: Describe en una o dos frases el evento clave o el tema más """
                """importante de la noticia."""
                """2. Actores principales y sus reacciones: Menciona las partes involucradas y sus posturas o 
                reacciones significativas (gobiernos, empresas, expertos, etc.)."""
                """3. Consecuencias o implicaciones: Señala las implicaciones principales o cualquier posible """
                """consecuencia a corto o largo plazo derivada de la noticia."""
                """4. Contexto adicional relevante: Incluye cualquier información de contexto que sea relevante para """
                """comprender mejor el evento o las decisiones tomadas."""
                """El resumen debe ser natural y no exceder las cinco frases. Evita juicios o interpretaciones, 
                    enfocándote en capturar la esencia de la noticia sin perder detalles importantes."""
            ),
            HumanMessagePromptTemplate.from_template("El texto de la noticias es:\n{news}")
        ]
    )

    llm_chain = prompt_summarization | llm_ollama | StrOutputParser()

    return llm_chain.invoke({"news": news})


def write_news_summary_from_scraper_df_in_json(file_path: str = "data/bronze/news.parquet", model_name: str = "llama3.2:3b"):
    """
    Lee noticias de un archivo Parquet, genera resúmenes y los guarda en un archivo JSON.

    Esta función lee noticias de un archivo Parquet, genera resúmenes para cada noticia
    que aún no ha sido procesada utilizando el modelo de lenguaje especificado, y guarda
    estos resúmenes en un archivo JSON. Implementa un sistema de caché para evitar
    reprocesar noticias ya resumidas.

    :param file_path: Ruta al archivo Parquet que contiene las noticias.
    :type file_path: str
    :param model_name: Nombre del modelo de lenguaje a utilizar para la generación de resúmenes.
    :type model_name: str
    :return: None

    Example:
        >>> write_news_summary_from_scraper_df_in_json()
        # Procesa las noticias y guarda los resúmenes en 'data/octubre_news_summary.json'
    """

    df_news_el_tiempo = pd.read_parquet(file_path)

    if not os.path.exists("data/silver/octubre_news_summary.json"):
        with open("data/silver/octubre_news_summary.json", "w") as f:
            json.dump({}, f)

    with open("data/silver/octubre_news_summary.json", "r") as f:
        results = json.load(f)

    for index, row in df_news_el_tiempo.iterrows():
        if row["url_page"] in results:
            continue
        summary = get_summary_news(row["news_content"], model_name)
        if summary is None or summary == "":
            summary = ""
        results[row["url_page"]] = summary
        with open("data/silver/octubre_news_summary.json", "w") as f:
            json.dump(results, f, indent=4)


def get_embed_summary_df(embed_model_name: str = "nomic-embed-text", write_embed_df: bool = False):
    """
    Carga resúmenes de noticias, genera embeddings y opcionalmente los guarda en un archivo Parquet.

    Esta función carga los resúmenes de noticias de un archivo JSON, genera embeddings
    para estos resúmenes utilizando el modelo especificado, y opcionalmente guarda estos
    embeddings en un archivo Parquet para su uso posterior.

    :param embed_model_name: Nombre del modelo de embeddings a utilizar. Debe ser el nombre en Ollama
    :type embed_model_name: str
    :param write_embed_df: Indica si se deben guardar los embeddings en un archivo Parquet.
    :type write_embed_df: bool
    :return: Un DataFrame de pandas con los embeddings de los resúmenes de noticias.
    :rtype: pandas.DataFrame

    Example:
        >>> df_embeddings = get_embed_summary_df(write_embed_df=True)
        # Genera embeddings y los guarda en 'data/embeddings/nomic-embed-text_summary_news_el_tiempo.parquet'
    """
    if not isinstance(embed_model_name, str):
        raise ValueError(f"El nombre del modelo de embeddings debe ser un string")

    name_to_write_or_read = embed_model_name.split(":")[0]

    if not Path(f"data/silver/{name_to_write_or_read}_summary_news_el_tiempo.parquet").exists():
        with open(f"data/silver/octubre_news_summary.json", "r") as f:
            summaries = json.load(f)

        df_news_summary = pd.DataFrame(list(summaries.items()), columns=["url_page", "summary"])
        embed_summary_news = get_embed_text(list(df_news_summary["summary"]), embed_model_name)

        if write_embed_df:
            (
                pd.DataFrame(embed_summary_news, index=df_news_summary["url_page"])
                .to_parquet(f"data/silver/{name_to_write_or_read}_summary_news_el_tiempo.parquet")
            )

        return pd.DataFrame(embed_summary_news, index=df_news_summary["url_page"])

    return pd.read_parquet(f"data/silver/{name_to_write_or_read}_summary_news_el_tiempo.parquet")
