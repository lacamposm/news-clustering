import json
import joblib
import logging
import pandas as pd
from utils.llms import get_embed_summary_df, get_topic_in_cluster

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def prepare_df_to_cluster_caracterization(embed_model_name: str = "mxbai-embed-large:latest"):
    """    
    :return: 
    """
    with open("data/silver/octubre_news_summary.json", "r") as file:
        data_dict = json.load(file)

    df_summary = pd.DataFrame.from_dict(data_dict, orient="index", columns=["summary"])
    model = joblib.load("model/news_clustering_model.joblib")
    df_embed = get_embed_summary_df(embed_model_name)
    labels_predicted = model.predict(df_embed) + 1
    df_predict = pd.DataFrame(labels_predicted, index=df_embed.index, columns=["cluster"])

    return df_summary.merge(df_predict, left_index=True, right_index=True)


def get_text_sample_from_cluster(embed_model_name="mxbai-embed-large:latest", size_sample=None):
    """
    :param size_sample:
    :param embed_model_name:
    :return:
    """
    df_cluster = prepare_df_to_cluster_caracterization(embed_model_name)
    if size_sample > df_cluster["cluster"].value_counts().min():
        raise ValueError(f"Tamanio de cluster menor a: {size_sample}")
    result_dict_cluster_text = dict()

    for label_cluster in df_cluster["cluster"].unique():
        df_sample_cluster = df_cluster.query("cluster == @label_cluster").sample(size_sample)
        text_to_llm = df_sample_cluster["summary"].to_list()
        text_to_llm = "\n".join([f"Noticia {i + 1}\n{news}" for i, news in enumerate(text_to_llm)])
        result_dict_cluster_text[f"cluster_{label_cluster}"] = text_to_llm

    return result_dict_cluster_text


def get_dict_caracterization_by_cluster(
        embed_model_name="mxbai-embed-large:latest", size_sample=None, model_name="llama3.2:3b"
):
    """
    :return:
    """
    dict_cluster_text = get_text_sample_from_cluster(embed_model_name, size_sample)

    return {
        key: get_topic_in_cluster(value, model_name) for key, value in dict_cluster_text.items()
    }
