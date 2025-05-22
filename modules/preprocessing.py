import json
import re
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sympy.codegen.ast import continue_

from config import DATASET_PATH, PROCESSED_DATA_PATH, STOPWORDS_LANGUAGE, BATCH_SIZE

if os.path.exists("C:/Users/Barcode/AppData/Roaming/nltk_data"):
    print('The file existed.')
else:
    nltk.download('punkt')
    nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words(STOPWORDS_LANGUAGE))

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = ' '.join(text.split())
        return text

    def tokenize(self, text):
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.stop_words]

    def preprocess_text(self, text):
        if not text:
            return []
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        return tokens

    def process_dataset(self):
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Dataset file not found at: {DATASET_PATH}")

        processed_data = []
        batch_count = 0
        doc_id = 0
        skipped_docs = 0
        empty_text_docs = 0

        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    # Access the 'passages' field
                    passages = doc.get('passages', {})
                    if isinstance(passages, list):
                        passages_list = passages
                    else:
                        passages_list = [passages]

                    for passage in passages_list:
                        passage_text = passage.get('passage_text', '')
                        url = passage.get('url', '')
                        is_selected = passage.get('is_selected', 0)

                        if not passage_text.strip():
                            empty_text_docs += 1
                            skipped_docs += 1
                            continue

                        tokens = self.preprocess_text(passage_text)

                        if not tokens:
                            print(f"Document {doc_id} has no tokens after preprocessing. Original text: {passage_text}")
                            skipped_docs += 1
                            continue
                        processed_data.append({
                            'doc_id': doc_id,
                            'original_text': passage_text,
                            'tokens': tokens,
                            'url': url,
                            'is_selected': is_selected
                        })
                        doc_id += 1
                        if len(processed_data) >= BATCH_SIZE:
                            self.save_batch(processed_data, batch_count)
                            processed_data = []
                            batch_count += 1

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {line}. Error: {e}")
                    continue
            if processed_data:
                self.save_batch(processed_data, batch_count)

        print(f"Preprocessing complete. Total documents processed: {doc_id}")
        print(f"Skipped {skipped_docs} documents with no tokens.")
        print(f"Found {empty_text_docs} documents with empty passage_text.")

    def save_batch(self, data, batch_num):
        output_path = os.path.join(PROCESSED_DATA_PATH, f"batch_{batch_num}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    preprocessor.process_dataset()