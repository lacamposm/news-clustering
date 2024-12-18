import pandas as pd
from pathlib import Path
from pprint import pprint
from joblib import dump, load
from utils.scraper import get_scraper_df_news_el_tiempo
from utils.llms import write_news_summary_from_scraper_df_in_json, get_embed_summary_df
from utils.plots import tsne_plot_2d, umap_plot_2d
from model.selection import random_search_best_model
from model.cluster_validation import get_dict_caracterization_by_cluster
from utils.custom_eval_metrics import silhouette_scorer


ollama_llm = "qwen2.5:7b"                       # Puede seleccionar el modelo LLM Ollama que desees tipo instruct
get_scraper_df_news_el_tiempo("data/bronze/octubre_news_el_tiempo.parquet", n_jobs=-1)
write_news_summary_from_scraper_df_in_json("data/bronze/octubre_news_el_tiempo.parquet", ollama_llm)

embed_model_name = "mxbai-embed-large:latest"   # Puedes usar otros modelos de embeddings si deseas.
n_iter = 1000                                   # Número de ajustes en RandomizedSearchCV.
scorers = ["silhouette", "davies-bouldin"]      # Puedes seleccionar diferentes metrica de adecuación de clusters
random_state = 42                               # Semilla aletoria de RandomizedSearchCV para replicar resultados.

df_emdeb = get_embed_summary_df(embed_model_name, write_embed_df=True)
tsne_plot_2d(df_emdeb)
umap_plot_2d(df_emdeb)

name_file_model = "news_clustering_model"

if not Path(f"model/{name_file_model}.joblib").exists():
    best_model = random_search_best_model(embed_model_name, scorers, n_iter, random_state)
    dump(best_model.best_estimator_, f"model/{name_file_model}.joblib")
    pd.DataFrame(best_model.cv_results_).to_excel(f"model/cv_results_{name_file_model}.xlsx", index=False)

# =========================================== Use del modelo obtenido =============================================== #
model_clustering = load(f"model/{name_file_model}.joblib")
tsne_plot_2d(df_emdeb, model_clustering, silhouette_scorer)
umap_plot_2d(df_emdeb, model_clustering, silhouette_scorer)

df_to_predict = df_emdeb.sample(5)
print(df_to_predict)
labels = model_clustering.predict(df_to_predict)
print("Predict results:")
print(pd.DataFrame({"cluster": labels}, index=df_to_predict.index))

model_name = "llama3.2:3b"      # Modelo que se usara para la caracterizacion de los clusters. Más capaz sería mejor.
size_sample = 3                 # Tamanio de muestra de noticias del cluster
pprint(get_dict_caracterization_by_cluster(embed_model_name, size_sample, model_name))
