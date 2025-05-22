from elasticsearch import Elasticsearch
from config import ES_HOST, ES_PORT, ES_INDEX_NAME, ES_EMBEDDING_FIELD, ELS_USER, ELS_PASSWORD


class SearchEngine:
    def __init__(self):
        self.es = Elasticsearch([{'host': ES_HOST, 'port': ES_PORT, 'scheme': 'https'}],
                                basic_auth=(ELS_USER,ELS_PASSWORD),
                                ca_certs="C:/Users/Barcode/Desktop/elasticsearch-8.17.0/config/certs/http_ca.crt",
                                request_timeout=30,
                                max_retries=10,)
        self.index_name = ES_INDEX_NAME

    def search(self, query_embedding, top_k=10):
        """Search for documents using the query embedding and return top-k results."""
        '''query = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": f"cosineSimilarity(params.query_vector, '{ES_EMBEDDING_FIELD}') + 1.0",
                        "params": {"query_vector": query_embedding}
                    }
                }
            },
            "size": top_k
        }'''
        query = {
            "query":{
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": f"""
                                    double sim = cosineSimilarity(params.query_vector, '{ES_EMBEDDING_FIELD}');
                                    if (Double.isNaN(sim)) {{
                                        return 0.0;
                                    }}
                                    return sim + 1.0;
                                """,
                    "params": {"query_vector": query_embedding}
                }
            }
        },
        "size": top_k

    }

        # Execute the search
        response = self.es.search(index=self.index_name, body=query)
        hits = response['hits']['hits']

        # Process the results
        results = []
        for hit in hits:
            doc = hit['_source']
            # Similarity score is (cosineSimilarity + 1) / 2 to normalize to [0, 1]
            similarity = (hit['_score'] - 1.0)  # Undo the +1.0 from the script
            results.append({
                'doc_id': doc['doc_id'],
                'original_text': doc['original_text'],
                'url': doc['url'],
                'is_selected': doc['is_selected'],
                'similarity': similarity
            })

        return results

if __name__ == "__main__":
    # Example usage
    search_engine = SearchEngine()
    # Dummy query embedding (replace with actual embedding from query_processing)
    dummy_embedding = [0.1] * 300  # Example: 300-dimensional vector
    results = search_engine.search(dummy_embedding, top_k=5)
    for result in results:
        print(f"Doc ID: {result['doc_id']}, Similarity: {result['similarity']:.4f}")
        print(f"Text: {result['original_text'][:100]}...")  # Print first 100 chars