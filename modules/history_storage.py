import psycopg2
import json
from datetime import datetime
from config import PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD

class HistoryStorage:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            database=PG_DATABASE,
            user=PG_USER,
            password=PG_PASSWORD
        )
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS search_history (
            id SERIAL PRIMARY KEY,
            query_text TEXT NOT NULL,
            results JSONB NOT NULL,
            search_time TIMESTAMP NOT NULL
        );
        """
        self.cursor.execute(create_table_query)
        self.conn.commit()

    def save_search(self, query_text, results):
        search_time = datetime.now()
        results_json = json.dumps(results)
        insert_query = """
        INSERT INTO search_history (query_text, results, search_time)
        VALUES (%s, %s, %s);
        """
        self.cursor.execute(insert_query, (query_text, results_json, search_time))
        self.conn.commit()

    def get_history(self, limit=5):
        select_query = """
        SELECT query_text, results, search_time
        FROM search_history
        ORDER BY search_time DESC
        LIMIT %s;
        """
        self.cursor.execute(select_query, (limit,))
        return self.cursor.fetchall()

    def __del__(self):
        self.cursor.close()
        self.conn.close()

if __name__ == "__main__":
    history_storage = HistoryStorage()
    query_text = "What is RBA?"
    results = [{"doc_id": "1", "original_text": "The Reserve Bank of Australia...", "similarity": 0.9}]
    history_storage.save_search(query_text, results)
    history = history_storage.get_history()
    for entry in history:
        print(f"Query: {entry[0]}, Time: {entry[2]}")