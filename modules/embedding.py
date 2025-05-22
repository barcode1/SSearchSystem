import json
import os
import numpy as np
from gensim.models import Word2Vec
from config import (PROCESSED_DATA_PATH, WORD2VEC_MODEL_PATH,
                    WORD2VEC_VECTOR_SIZE, WORD2VEC_WINDOW,
                    WORD2VEC_MIN_COUNT, WORD2VEC_EPOCHS, WORD2VEC_WORKERS, ES_BATCH_SIZE)

class EmbeddingGenerator:
    def __init__(self):
        self.model = None
        self.vector_size = WORD2VEC_VECTOR_SIZE

    def train_word2vec(self, sentences):
        """Train a Word2Vec model on the given sentences."""
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=WORD2VEC_WINDOW,
            min_count=WORD2VEC_MIN_COUNT,
            workers=WORD2VEC_WORKERS,
            epochs=WORD2VEC_EPOCHS
        )
        self.model.save(WORD2VEC_MODEL_PATH)

    def load_word2vec(self):
        """Load a pre-trained Word2Vec model."""
        if os.path.exists(WORD2VEC_MODEL_PATH):
            self.model = Word2Vec.load(WORD2VEC_MODEL_PATH)
        else:
            raise FileNotFoundError("Word2Vec model not found. Train the model first.")

    def get_document_vector(self, tokens):
        """Compute the average word embedding for a document."""
        if not self.model:
            self.load_word2vec()

        vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]
        if not vectors:
            return np.zeros(self.vector_size)
        return np.mean(vectors, axis=0).tolist()

    def process_batches(self):
        """Process all batches and generate embeddings for each document."""
        all_sentences = []
        doc_embeddings = []

        batch_files = [f for f in os.listdir(PROCESSED_DATA_PATH) if f.startswith("batch_") and f.endswith(".json")]
        for batch_file in batch_files:
            with open(os.path.join(PROCESSED_DATA_PATH, batch_file), 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
                for doc in batch_data:
                    tokens = doc['tokens']
                    if tokens:  # Only add non-empty token lists
                        all_sentences.append(tokens)

        if all_sentences:
            self.train_word2vec(all_sentences)

        batch_count = 0
        batch_data = []
        for batch_file in batch_files:
            with open(os.path.join(PROCESSED_DATA_PATH, batch_file), 'r', encoding='utf-8') as f:
                docs = json.load(f)
                for doc in docs:
                    embedding = self.get_document_vector(doc['tokens'])
                    batch_data.append({
                        'doc_id': doc['doc_id'],
                        'original_text': doc['original_text'],
                        'embedding': embedding,
                        'url': doc['url'],
                        'is_selected': doc['is_selected']
                    })

                    # Save batch if size reached
                    if len(batch_data) >= ES_BATCH_SIZE:
                        self.save_embeddings(batch_data, batch_count)
                        batch_data = []
                        batch_count += 1

        if batch_data:
            self.save_embeddings(batch_data, batch_count)

    def save_embeddings(self, data, batch_num):
        """Save document embeddings to file."""
        output_path = os.path.join(PROCESSED_DATA_PATH, f"embeddings_batch_{batch_num}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    embedder = EmbeddingGenerator()
    embedder.process_batches()