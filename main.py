from modules.preprocessing import TextPreprocessor
from modules.embedding import EmbeddingGenerator
from modules.elasticsearch_storage import ElasticsearchStorage
from modules.query_processing import QueryProcessor
from modules.query_expansion import QueryExpander
from modules.search_engine import SearchEngine
from modules.history_storage import HistoryStorage

def main():
    # Initialize components
    preprocessor = TextPreprocessor()
    embedder = EmbeddingGenerator()
    es_storage = ElasticsearchStorage()
    query_processor = QueryProcessor()
    query_expander = QueryExpander()
    search_engine = SearchEngine()
    history_storage = HistoryStorage()

    # Step 1: Preprocess the dataset
    print("Preprocessing dataset...")
    preprocessor.process_dataset()

    # Step 2: Generate embeddings
    print("Generating embeddings...")
    embedder.process_batches()

    # Step 3: Index documents in Elasticsearch
    print("Indexing documents in Elasticsearch...")
    es_storage.index_documents()

    # Step 4: Example search
    query_text = "What is RBA?"
    print(f"\nSearching for: {query_text}")

    # Process query
    tokens, query_embedding = query_processor.process_query(query_text)
    if query_embedding is None:
        print("Unable to process query.")
        return

    # Initial search
    initial_results = search_engine.search(query_embedding)
    print("Initial search results:", len(initial_results))

    # Expand query if needed
    expanded_tokens = query_expander.expand_query(tokens, initial_results)
    if expanded_tokens != tokens:
        print("Expanded tokens:", expanded_tokens)
        expanded_text = " ".join(expanded_tokens)
        _, query_embedding = query_processor.process_query(expanded_text)

    # Final search
    results = search_engine.search(query_embedding)
    print("Final search results:", len(results))

    # Save to history
    history_storage.save_search(query_text, results)

    # Display results
    for result in results:
        print(f"\nDoc ID: {result['doc_id']}, Similarity: {result['similarity']:.4f}")
        print(f"Text: {result['original_text'][:100]}...")

if __name__ == "__main__":
    main()