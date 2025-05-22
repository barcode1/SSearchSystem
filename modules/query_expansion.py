import numpy as np
from gensim.models import Word2Vec
from config import SIMILARITY_THRESHOLD, MAX_EXPANSION_TERMS, WORD2VEC_MODEL_PATH

class QueryExpander:
    def __init__(self):
        self.model = Word2Vec.load(WORD2VEC_MODEL_PATH)

    def get_similar_words(self, word, top_n=3):
        try:
            similar_words = self.model.wv.most_similar(word, topn=top_n)
            return [word for word, similarity in similar_words]
        except KeyError:
            print(f"Word '{word}' not in vocabulary.")
            return []

    def expand_query(self, tokens, search_results):
        if not search_results:
            avg_similarity = 0.0
        else:
            similarities = [result['similarity'] for result in search_results]
            avg_similarity = np.mean(similarities) if similarities else 0.0
        print(f"Average similarity: {avg_similarity}, Threshold: {SIMILARITY_THRESHOLD}")
        if avg_similarity < SIMILARITY_THRESHOLD:
            print("Expanding query due to low similarity...")
            expanded_tokens = set(tokens)
            for token in tokens:
                similar_words = self.get_similar_words(token)
                print(f"Similar words for '{token}': {similar_words}")
                for similar_word in similar_words:
                    if len(expanded_tokens) - len(tokens) >= MAX_EXPANSION_TERMS:
                        break
                    expanded_tokens.add(similar_word)
            expanded_tokens = list(expanded_tokens)
            print(f"Expanded tokens: {expanded_tokens}")
            return expanded_tokens
        else:
            print("No expansion needed.")
            return tokens

if __name__ == "__main__":
    expander = QueryExpander()
    tokens = ["rba"]
    search_results = [{"similarity": 0.1}, {"similarity": 0.2}]
    expanded_tokens = expander.expand_query(tokens, search_results)
    print(f"Final expanded tokens: {expanded_tokens}")