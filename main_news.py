from utils.scraper import get_df_news_to_summarize
from utils.llms import write_summary_from_dataframe_in_json, get_embed_dataframe

get_df_news_to_summarize("data/bronze/octubre_news_el_tiempo.parquet", num_cores=14)
write_summary_from_dataframe_in_json("data/bronze/octubre_news_el_tiempo.parquet", "qwen2.5:7b")

list_names_model_embed = [
    "nomic-embed-text",
    "mxbai-embed-large:335m",
    "snowflake-arctic-embed:latest",
    "bge-m3",
]

for model_embed in list_names_model_embed:
    df_emdeb = get_embed_dataframe(model_embed, write_embed_df=True)
    print("Name embedding:", model_embed)
    print(df_emdeb)
