import json
import os
from elasticsearch import Elasticsearch, helpers
from config import ES_HOST, ES_PORT, ES_INDEX_NAME, ES_EMBEDDING_FIELD, ES_BATCH_SIZE, PROCESSED_DATA_PATH, ELS_USER, \
    ELS_PASSWORD


class ElasticsearchStorage:
    def __init__(self):
        self.es = Elasticsearch([{'host': ES_HOST, 'port': ES_PORT, 'scheme': 'https'}],
                                basic_auth=(ELS_USER,ELS_PASSWORD),
                                ca_certs="C:/Users/Barcode/Desktop/elasticsearch-8.17.0/config/certs/http_ca.crt",
                                request_timeout=30,
                                max_retries=10,
                                )
        self.index_name = ES_INDEX_NAME

    def create_index(self):
        """Create an Elasticsearch index with custom mapping."""
        mapping = {
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "original_text": {"type": "text"},
                    "url": {"type": "keyword"},
                    "is_selected": {"type": "integer"},
                    ES_EMBEDDING_FIELD: {
                        "type": "dense_vector",
                        "dims": 300,  # Must match WORD2VEC_VECTOR_SIZE
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }

        # Delete the index if it exists
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)

        # Create the index with the mapping
        self.es.indices.create(index=self.index_name, body=mapping)

    def index_documents(self):
        """Index documents and their embeddings in Elasticsearch."""
        self.create_index()

        # Collect all embedding batch files
        batch_files = [f for f in os.listdir(PROCESSED_DATA_PATH) if f.startswith("embeddings_batch_") and f.endswith(".json")]

        for batch_file in batch_files:
            with open(os.path.join(PROCESSED_DATA_PATH, batch_file), 'r', encoding='utf-8') as f:
                docs = json.load(f)
                actions = []

                for doc in docs:
                    actions.append({
                        "_index": self.index_name,
                        "_id": doc['doc_id'],
                        "_source": {
                            "doc_id": doc['doc_id'],
                            "original_text": doc['original_text'],
                            "url": doc['url'],
                            "is_selected": doc['is_selected'],
                            ES_EMBEDDING_FIELD: doc['embedding']
                        }
                    })

                    # Index batch if size reached
                    if len(actions) >= ES_BATCH_SIZE:
                        helpers.bulk(self.es, actions)
                        actions = []

                # Index remaining actions
                if actions:
                    helpers.bulk(self.es, actions)

if __name__ == "__main__":
    es_storage = ElasticsearchStorage()
    es_storage.index_documents()