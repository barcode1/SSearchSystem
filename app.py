from flask import Flask, render_template, request
from modules.query_processing import QueryProcessor
from modules.search_engine import SearchEngine
from modules.query_expansion import QueryExpander
from modules.history_storage import HistoryStorage

app = Flask(__name__)

# Initialize components
query_processor = QueryProcessor()
search_engine = SearchEngine()
query_expander = QueryExpander()
history_storage = HistoryStorage()

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    expanded_query = []
    no_match_message = None
    query_not_found_message = None  # New variable for "query not in collection"
    history = history_storage.get_history(limit=5)

    if request.method == 'POST':
        query = request.form['query']
        try:
            # Process the query
            tokens, query_embedding = query_processor.process_query(query)
            if tokens is None or query_embedding is None:
                query_not_found_message = "همچین کوئری نیست"
            else:
                # Initial search
                initial_results = search_engine.search(query_embedding, top_k=10)

                # Check if the query matches any document
                if not initial_results or max(result['similarity'] for result in initial_results) < 0.01:
                    query_not_found_message = "همچین کوئری نیست"
                else:
                    # Expand query if necessary
                    expanded_tokens = query_expander.expand_query(tokens, initial_results)
                    expanded_query = expanded_tokens  # Store expanded tokens to display

                    # If query was expanded, search again
                    if expanded_tokens != tokens:
                        _, expanded_embedding = query_processor.process_query(' '.join(expanded_tokens))
                        results = search_engine.search(expanded_embedding, top_k=10)
                    else:
                        results = initial_results

                    # Check if results have low similarity (less than 0.1)
                    if results:
                        max_similarity = max(result['similarity'] for result in results)
                        if max_similarity < 0.1:
                            no_match_message = "Sorry, your search didn't match any documents. Try a different query!"
                    else:
                        no_match_message = "Sorry, your search didn't match any documents. Try a different query!"

                    # Save search history only if query is found in collection
                    history_storage.save_search(query, results)

                    # Update history
                    history = history_storage.get_history(limit=5)

        except Exception as e:
            return str(e), 500

        return render_template('index.html', results=results, query=query, expanded_query=expanded_query,
                               history=history, no_match_message=no_match_message, query_not_found_message=query_not_found_message)

    return render_template('index.html', results=results, query="", expanded_query=expanded_query, history=history,
                           no_match_message=no_match_message, query_not_found_message=query_not_found_message)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)