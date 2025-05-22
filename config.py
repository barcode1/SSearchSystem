import os

DATASET_PATH = "C:/Users/Barcode/PycharmProjects/SSearchSystem/data/ms_marco.json"
WORD2VEC_MODEL_PATH = "C:/Users/Barcode/PycharmProjects/SSearchSystem/models/word2vec.model"
PROCESSED_DATA_PATH = "C:/Users/Barcode/PycharmProjects/SSearchSystem/data/processed/"
LOG_PATH = "C:/Users/Barcode/PycharmProjects/SSearchSystem/logs/app.log"

WORD2VEC_VECTOR_SIZE = 300
WORD2VEC_WINDOW = 5  # Context window size
WORD2VEC_MIN_COUNT = 1  # Minimum word frequency
WORD2VEC_EPOCHS = 5  # Number of training epochs
WORD2VEC_WORKERS = 4  # Number of worker threads

ES_HOST = "127.0.0.1"
ES_PORT = 9200
ELS_PASSWORD = "JhxAy7rJ0vfnsU*sdkg8"
ELS_USER = "elastic"
ES_INDEX_NAME = "smart_search_docs"
ES_EMBEDDING_FIELD = "embedding"
ES_BATCH_SIZE = 1000

PG_HOST = "localhost"
PG_PORT = 5432
PG_DATABASE = "postgres"
PG_USER = "postgres"
PG_PASSWORD = "gholami1378"

SIMILARITY_THRESHOLD = 0.5
MAX_EXPANSION_TERMS = 5

STOPWORDS_LANGUAGE = "english"
BATCH_SIZE = 1000  # Batch size for processing data

# Ensure directories exist
os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(WORD2VEC_MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)