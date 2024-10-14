# News Clustering

## Descripción

Este proyecto realiza un análisis de clustering de noticias del periódico El Tiempo de Colombia. El proceso incluye la obtención de URLs de noticias, extracción del texto completo, generación de resúmenes, creación de embeddings de los resúmenes y finalmente la aplicación de técnicas de modelado y clustering.

## Requisitos

- Python 3.10.12
- Ollama (https://ollama.com/)
  - Se sugiere que en Ollama descarge los siguientes modelos para realizar los resumenes (ambos en version instruct): 
      - [llama3.2:3b](https://ollama.com/library/llama3.2)
      - [qwen2.5:7b](https://ollama.com/library/qwen2.5)
  - Una opcion de modelo de embeddings es: [nomic-embed-text:v1.5](https://ollama.com/library/nomic-embed-text:v1.5) 
## Dependencias

El proyecto utiliza las bibliotecas de Python en requirements.txt

## Instalación

Clone este repositorio:

```bash
git clone https://github.com/lacamposm/news-clustering.git
cd news-clustering
```

Instale las dependencias:

```bash
pip install -r requirements.txt
```

Descargue, instale y ejecute Ollama siguiendo las instrucciones en [https://ollama.com/](https://ollama.com/).

## Estructura del Proyecto

El proyecto consta de cuatro scripts principales:

- `utils/scraper.py`: Contiene funciones para obtener URLs de noticias y extraer su contenido.
- `utils/llms.py`: Incluye funciones para generar resúmenes y embeddings de las noticias.
- `clustering.py`: (En desarrollo) Realiza el análisis de clustering utilizando técnicas como t-SNE, K-means y HDBSCAN.
- `main_news.py`: Script principal que orquesta el proceso completo de obtención de datos, generación de resúmenes y embeddings.

## Uso

### Para ejecutar el proyecto completo:

```bash
python main_news.py
```

Este script realizará los siguientes pasos:

- Obtendrá las noticias de El Tiempo.
- Generará resúmenes de las noticias.
- Creará embeddings de los resúmenes utilizando varios modelos.

### Para realizar el análisis de clustering:

```bash
python model/clustering.py
```

Este script aplicará técnicas de visualización (t-SNE) y clustering (K-means y HDBSCAN) a los embeddings generados.

## Notas Adicionales

- Todos los archivos de datos se generan desde cero durante la ejecución del proyecto.
- Asegúrese de tener suficiente espacio en disco y recursos computacionales, ya que el procesamiento de grandes cantidades de noticias puede ser intensivo.

## Contribuciones

Las contribuciones a este proyecto son bienvenidas. Por favor, abra un issue para discutir cambios mayores antes de enviar un pull request.

## Licencia

Este proyecto es de código abierto y está disponible para su uso y reproduccion.
