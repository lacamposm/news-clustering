import pandas as pd
from joblib import dump, load
from utils.scraper import get_scraper_df_news_el_tiempo
from utils.llms import write_news_summary_from_scraper_df_in_json, get_embed_summary_df
from utils.plots import tsne_plot_2d, umap_plot_2d
from model.selection import random_search_best_model


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

best_model = random_search_best_model(embed_model_name, scorers, n_iter, random_state)
labels = best_model.predict(df_emdeb)
tsne_plot_2d(df_emdeb, labels, best_model)
umap_plot_2d(df_emdeb, labels, best_model)

name_file_model = "news_clustering_model"
dump(best_model.best_estimator_, f"model/{name_file_model}.joblib")
pd.DataFrame(best_model.cv_results_).to_excel(f"model/cv_results_{name_file_model}.xlsx", index=False)
model_clustering = load(f"model/{name_file_model}.joblib")

df_to_predict = df_emdeb.sample(5)
print(df_to_predict)
labels = model_clustering.predict(df_to_predict)
print("Predict results:")
print(pd.DataFrame({"cluster": labels}, index=df_to_predict.index))
