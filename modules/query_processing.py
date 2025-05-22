from modules.preprocessing import TextPreprocessor
from modules.embedding import EmbeddingGenerator

class QueryProcessor:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.embedder = EmbeddingGenerator()
        self.embedder.load_word2vec()

    def process_query(self, query_text):
        tokens = self.preprocessor.preprocess_text(query_text)
        if not tokens:
            return None
        query_embedding = self.embedder.get_document_vector(tokens)
        return tokens, query_embedding

if __name__ == "__main__":
    query_processor = QueryProcessor()
    query_text = "What is RBA?"
    tokens, embedding = query_processor.process_query(query_text)
    print(f"Tokens: {tokens}")
    print(f"Embedding: {embedding[:5]}...")