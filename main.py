import pickle
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import openai
import os
from pydantic import BaseModel
import time
from collections import defaultdict

app = FastAPI()

# Simple rate limiter
request_counts = defaultdict(list)
RATE_LIMIT_WINDOW = 60  # 1 minute
RATE_LIMIT_MAX_REQUESTS = 100  # max requests per minute

def check_rate_limit(client_ip: str = "default"):
    """Simple rate limiter - returns True if request should be allowed"""
    current_time = time.time()
    # Clean old requests outside the window
    request_counts[client_ip] = [req_time for req_time in request_counts[client_ip] 
                                if current_time - req_time < RATE_LIMIT_WINDOW]
    
    # Check if we're under the limit
    if len(request_counts[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
        return False
    
    # Add current request
    request_counts[client_ip].append(current_time)
    return True

MILVUS_HOST = os.getenv("MILVUS_HOST", "in03-874be76b9aa0be7.serverless.gcp-us-west1.cloud.zilliz.com")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "268c3796886a41827afcee6560f083fbfc4992ae7265598b4d3582979748054380929293cd76ea79244845abf9773e4e9128de0e")
MILVUS_COLLECTION_SPARSE = os.getenv("MILVUS_COLLECTION_SPARSE", "githubSparseVectorRag")
MILVUS_COLLECTION_DENSE = os.getenv("MILVUS_COLLECTION_DENSE", "githubDenseVectorRag")


# Milvus client setup (assume you have this)
from pymilvus import MilvusClient
milvus_client = MilvusClient(
    uri=f"https://{MILVUS_HOST}",
    token=MILVUS_TOKEN
)
COLLECTION_DENSE = os.getenv("MILVUS_COLLECTION_DENSE", "githubDenseVectorRag")
COLLECTION_SPARSE = os.getenv("MILVUS_COLLECTION_SPARSE", "githubSparseVectorRag")

# --- Helper Functions ---
def search_dense(query, top_k=5):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(input=[query], model="text-embedding-3-large", dimensions=3072)
    query_embedding = response.data[0].embedding
    results = milvus_client.search(
        collection_name=COLLECTION_DENSE,
        data=[query_embedding],
        #anns_field="vector",
        #search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["content", "metadata"]
    )
    #print("len(dense results): ", len(results))
    #print("dense results: ", results)
    #print("type(results): ", type(results))
    #print("type(results[0]): ", type(results[0]))
    return results[0]

def search_sparse(query, top_k=5):
    query_sparse = vectorizer.transform([query])
    indices = query_sparse.indices.tolist()
    values = query_sparse.data.tolist()
    sparse_query = {int(idx): float(val) for idx, val in zip(indices, values)}
    results = milvus_client.search(
        collection_name=COLLECTION_SPARSE,
        data=[sparse_query],
        #anns_field="vector",
        #search_params={"metric_type": "IP"},
        limit=top_k,
        output_fields=["content", "metadata"]
    )
    return results[0]

def combine_and_rerank(sparse_results, dense_results, query, rerank_top_n=5):
    # Combine by unique id (primary_key or content), keep best score from either
    combined = {}
    #print("sparse_results: ", sparse_results, flush=True)
    #print("dense_results: ", dense_results, flush=True)
    for r in sparse_results + dense_results:
        entity = r.get('entity', {})
        content = entity.get('content')
        if isinstance(content, list) and content:
            content = content[0]
        if not content:
            continue  # skip if content is missing or empty
        unique_id = r.get('primary_key')
        score = r.get('distance', float('inf'))
        cur_score = (combined or {}).get(unique_id, [float('inf')])[0] or float('inf')
        #print("cur_score: ", cur_score, flush=True)
        #print("score: ", score, flush=True)
        if cur_score is None:
            cur_score = float('inf')
        if score < cur_score:
            combined[unique_id] = [score, r]
    # Build pk_to_hit mapping
    pk_to_hit = {pk: hit for pk, (score, hit) in combined.items()}
    # Prepare rerank prompt
    prompt = f"""
Given the query: '{query}', analyze the following passages for relevance.

IMPORTANT: If NONE of the passages actually answer the query or contain relevant information, return an empty list []. This includes cases where:
- The query asks for specific functions, classes, or features that don't exist in the codebase
- The query asks for private repository content or secrets
- The query asks for content that's not present in the indexed documentation
- The passages are completely unrelated to the query

Only return results if they genuinely contain information that answers the query.

Passages to analyze:
"""
    i = 0
    for (key, value) in combined.items():
        snippet = value[1].get('entity', {}).get('content')
        if isinstance(snippet, list):
            snippet = snippet[0] if snippet else ''
        snippet = snippet[:300]  # Truncate to 300 chars
        prompt += f"[{i+1}] (primary_key: {key}) {snippet}\n"
        i += 1
    prompt += (
        f"\nIf any passages are relevant, pick the top {rerank_top_n} most relevant results. "
        "For each relevant result, return an object with the fields 'primary_key' (copied exactly from above) and 'content' (copied exactly from above). "
        "If NO passages are relevant to the query, return an empty list []. "
        "Return a JSON list of these objects. Do not return a list of numbers or any other format."
    )
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    #print("response from openai: ", response.choices[0].message.content, flush=True)

    import json
    import re
    # Use non-greedy regex to extract the first JSON list
    match = re.search(r'\[.*?\]', response.choices[0].message.content, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            reranked = json.loads(json_str)
            # Robust type/structure check
            if not isinstance(reranked, list):
                print("Warning: LLM did not return a list. Got:", reranked, flush=True)
                reranked = []
            elif reranked and isinstance(reranked[0], int):
                print("Warning: LLM returned a list of ints, mapping to candidates. Got:", reranked, flush=True)
                # Map indices to candidates
                candidates_list = list(combined.values())
                mapped_results = []
                for idx in reranked:
                    if 1 <= idx <= len(candidates_list):
                        score, hit = candidates_list[idx - 1]
                        entity = hit.get('entity', {})
                        content = entity.get('content')
                        if isinstance(content, list):
                            content = content[0] if content else ""
                        metadata = entity.get('metadata')
                        if isinstance(metadata, list):
                            metadata = metadata[0] if metadata else ""
                        mapped_results.append({
                            "primary_key": hit.get('primary_key'),
                            "content": content,
                            "metadata": metadata
                        })
                reranked = mapped_results
            elif reranked and not isinstance(reranked[0], dict):
                print("Warning: LLM returned a list of non-dicts. Got:", reranked, flush=True)
                reranked = []
        except Exception as e:
            print("JSON decode error:", e, "Match was:", json_str, flush=True)
            # Try to recover by truncating to the last closing bracket
            last_bracket = json_str.rfind(']')
            if last_bracket != -1:
                try:
                    reranked = json.loads(json_str[:last_bracket+1])
                except Exception as e2:
                    print("JSON decode error after truncation:", e2, "Match was:", json_str[:last_bracket+1], flush=True)
                    # Try to salvage valid objects
                    objects = re.findall(r'\{.*?\}', json_str[:last_bracket+1], re.DOTALL)
                    reranked = []
                    for obj_str in objects:
                        try:
                            obj = json.loads(obj_str)
                            reranked.append(obj)
                        except Exception as e3:
                            print("Skipping invalid object:", e3, "Object was:", obj_str, flush=True)
                    
                    # Also try to extract partial objects from any remaining content
                    print("Extracting partial objects as well...", flush=True)
                    # Find all opening braces and try to extract partial content
                    brace_positions = [m.start() for m in re.finditer(r'\{', json_str[:last_bracket+1])]
                    for start_pos in brace_positions:
                        try:
                            # Try to find the last complete field before truncation
                            partial_str = json_str[start_pos:]
                            # Look for the last complete field (ends with comma or closing brace)
                            last_comma = partial_str.rfind(',')
                            last_quote = partial_str.rfind('"')
                            if last_comma > last_quote and last_comma > 0:
                                # Object ends with a comma, try to parse up to that point
                                partial_obj = partial_str[:last_comma] + "}"
                                obj = json.loads(partial_obj)
                                # Check if this object is not already in reranked (avoid duplicates)
                                if not any(r.get('primary_key') == obj.get('primary_key') for r in reranked):
                                    reranked.append(obj)
                                    print("Successfully extracted partial object", flush=True)
                            elif partial_str.endswith('}'):
                                # Complete object
                                obj = json.loads(partial_str)
                                # Check if this object is not already in reranked (avoid duplicates)
                                if not any(r.get('primary_key') == obj.get('primary_key') for r in reranked):
                                    reranked.append(obj)
                                    print("Successfully extracted complete object", flush=True)
                        except Exception as e4:
                            print("Failed to extract partial object from position", start_pos, ":", e4, flush=True)
                            continue
                    if not reranked:
                        print("No valid JSON objects found, attempting manual field extraction...", flush=True)
                        pk_match = re.search(r'"primary_key"\s*:\s*(\d+)', json_str)
                        content_match = re.search(r'"content"\s*:\s*"([^"]*)', json_str)
                        if pk_match and content_match:
                            reranked.append({
                                "primary_key": int(pk_match.group(1)),
                                "content": content_match.group(1),
                                "metadata": ""
                            })
                            print("Manually extracted partial object with primary_key and content.", flush=True)                      
            else:
                reranked = []
    else:
        reranked = []
    
    print("reranked: ", reranked, flush=True)

    # Build results with only content and metadata
    results = []
    for doc in reranked:
        pk = doc.get('primary_key')
        hit = pk_to_hit.get(pk)
        if not hit:
            continue
        entity = hit.get('entity', {})
        content = entity.get('content')
        if isinstance(content, list):
            content = content[0] if content else ""
        metadata = entity.get('metadata')
        if isinstance(metadata, list):
            metadata = metadata[0] if metadata else ""
        results.append({
            "content": content,
            "metadata": metadata
        })
    
    # Additional confidence check: if results seem irrelevant to the query, filter them out
    if results and len(results) > 0:
        # Check if the query contains terms that suggest it's looking for non-existent content
        query_lower = query.lower()
        suspicious_terms = ['private', 'secret', 'internal', 'confidential', 'api_key', 'token', 'password']
        if any(term in query_lower for term in suspicious_terms):
            # Double-check relevance with a simple keyword match
            relevant_results = []
            for result in results:
                content_lower = result.get('content', '').lower()
                # If the content doesn't contain any words from the query (excluding common words), it's likely irrelevant
                query_words = [word for word in query_lower.split() if len(word) > 3 and word not in ['what', 'how', 'when', 'where', 'with', 'from', 'this', 'that', 'they', 'have', 'been', 'will', 'your', 'repo', 'function']]
                if any(word in content_lower for word in query_words):
                    relevant_results.append(result)
            results = relevant_results
    
    return results[:rerank_top_n]

from sklearn.feature_extraction.text import TfidfVectorizer

def fit_vectorizer_from_milvus(collection_name, field="content"):
    all_docs = []
    offset = 0
    batch_size = 1000
    while True:
        results = milvus_client.query(
            collection_name=collection_name,
            output_fields=[field],
            offset=offset,
            limit=batch_size
        )
        if not results:
            break
        for doc in results:
            val = doc.get(field)
            if isinstance(val, list) and val:
                all_docs.append(val[0])
            elif isinstance(val, str):
                all_docs.append(val)
        offset += batch_size
    print(f"Fitting TF-IDF vectorizer on {len(all_docs)} documents from Milvus...")
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=0.001,
        max_df=0.999,
        lowercase=True,
        analyzer='word'
    )
    vectorizer.fit(all_docs)
    return vectorizer

# Only fit the vectorizer from Milvus at startup
vectorizer = fit_vectorizer_from_milvus(COLLECTION_SPARSE)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Milvus Search UI</title>
    </head>
    <body>
        <h2>Milvus Search UI</h2>
        <form id=\"searchForm\">
            <input type=\"text\" id=\"query\" placeholder=\"Enter your search query\" size=\"50\"/>
            <button type=\"submit\">Search</button>
        </form>
        <div id=\"results\"></div>
        <script>
        document.getElementById('searchForm').onsubmit = async function(e) {
            e.preventDefault();
            let query = document.getElementById('query').value;
            let res = await fetch('/search?query=' + encodeURIComponent(query));
            let data = await res.json();
            let html = '<h3>Results:</h3>';
            if (data.error) {
                html += '<div style=\"color:red\">' + data.error + '</div>';
            } else if (!data.results || data.results.length === 0) {
                html += '<div>No results found.</div>';
            } else {
                for (let hit of data.results) {
                    let score = hit.score || hit.distance || '';
                    let entity = hit.entity || hit;
                    let content = entity.content;
                    if (Array.isArray(content)) content = content[0];
                    let metadata = entity.metadata;
                    if (Array.isArray(metadata)) metadata = metadata[0];
                    html += '<div><b>Score:</b> ' + score + '<br/>';
                    html += '<b>Content:</b> <pre>' + (content || '') + '</pre><br/>';
                    html += '<b>Metadata:</b> ' + (metadata || '') + '</div><hr/>';
                }
            }
            document.getElementById('results').innerHTML = html;
        }
        </script>
    </body>
    </html>
    """

@app.get("/search")
def search(query: str, top_k: int = 5, request: Request = None):
    try:
        # Rate limiting
        client_ip = request.client.host if request and request.client else "default"
        if not check_rate_limit(client_ip):
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded. Please try again later."}
            )
        
        # Input validation
        if not query or not isinstance(query, str):
            return JSONResponse(
                status_code=400,
                content={"error": "Query must be a non-empty string"}
            )
        
        # Check query length (reasonable limit for search queries)
        if len(query) > 10000:  # 10KB limit
            return JSONResponse(
                status_code=413,
                content={"error": "Query too large. Maximum length is 10,000 characters."}
            )
        
        # Validate top_k parameter
        if not isinstance(top_k, int) or top_k <= 0 or top_k > 100:
            return JSONResponse(
                status_code=400,
                content={"error": "top_k must be a positive integer between 1 and 100"}
            )
        
        print("Received /search request with query:", query, flush=True)
        print("Starting dense search", flush=True)
        #dense_results = search_dense(query, top_k)
        dense_results = list(search_dense(query, top_k))
        #print("dense_results: ", dense_results, flush=True)
        print("Dense search complete", flush=True)
        print("Starting sparse search", flush=True)
        #sparse_results = search_sparse(query, top_k)
        sparse_results = list(search_sparse(query, top_k))
        #print("sparse_results: ", sparse_results, flush=True)
        print("Sparse search complete", flush=True)
        print("Combining and reranking results", flush=True)
        reranked_results = combine_and_rerank(sparse_results, dense_results, query, rerank_top_n=5)
        print(f"Returning {len(reranked_results)} reranked results", flush=True)
        import json as _json
        print("Returning to UI:", _json.dumps({"results": reranked_results}, indent=2), flush=True)
        return {"results": reranked_results}
    except Exception as e:
        print("Exception in /search:", e, flush=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        ) 
