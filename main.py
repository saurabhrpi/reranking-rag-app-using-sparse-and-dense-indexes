import pickle
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import openai
import os
from pydantic import BaseModel
from typing import List

app = FastAPI()

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

# --- Models ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    rerank_top_n: int = 5

class SearchResult(BaseModel):
    id: str
    score: float
    text: str

class RerankResponse(BaseModel):
    results: List[SearchResult]

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
    print("type(results): ", type(results))
    print("type(results[0]): ", type(results[0]))
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
    j = 0
    for r in sparse_results + dense_results:
        if j == 0:
            print("r: ", r, flush=True)
        j += 1
        entity = r.get('entity', {})
        content = entity.get('content')
        if isinstance(content, list) and content:
            content = content[0]
        print("content: ", content, flush=True)
        if not content:
            continue  # skip if content is missing or empty
        unique_id = r.get('primary_key') #or content
        score = r.get('distance', float('inf'))        
        #cur_score = combined[unique_id][0]
        cur_score = (combined or {}).get(unique_id, [float('inf')])[0] or float('inf')  # handles the case where it was None
        print("cur_score: ", cur_score, flush=True)
        if cur_score is None:
            cur_score = float('inf')
        if score < cur_score:
            combined[unique_id] = [score, content]
    #candidates = list(combined.values())
    print("combined: ", combined, flush=True)
    max_candidates = 20
    #candidates = sorted(combined.items(), key=lambda kv: kv[1][0])[:max_candidates]
    print("type(combined): ", type(combined), flush=True)
    candidates = combined
    print("len(sparse_results): ", len(sparse_results), flush=True)
    print("len(dense_results): ", len(dense_results), flush=True)
    #print("candidates: ", candidates, flush=True)
    # Limit the number of candidates for reranking
    
    #candidates = candidates[:max_candidates]
    #print("type(candidates): ", type(candidates), flush=True)
    #print("type(candidates[0]): ", type(candidates[0]), flush=True)
    #print("type(candidates[0][0]): ", type(candidates[0][0]), flush=True)
    #print("type(candidates[0][1]): ", type(candidates[0][1]), flush=True)    # Rerank with OpenAI (use text-davinci-003 or gpt-3.5-turbo)
    prompt = f"""
Given the query: '{query}', rank the following passages by relevance. Return the top {rerank_top_n} as a JSON list of objects with 'primary_key' and 'content'.

"""
    i = 0
    for (key, value) in candidates.items():
        #snippet = c['entity'].get('content', '')
        #print("c: ", c)
        #key = c[0]
        #print("key: ", key)
        #print(len(candidates[key]))
        snippet = value[1]
        if isinstance(snippet, list):
            snippet = snippet[0] if snippet else ''
        snippet = snippet[:300]  # Truncate to 300 chars
        #print("snippet: ", snippet)
        prompt += f"[{i+1}] (primary_key: {key}) {snippet}\n"
        i += 1
        
    prompt += f"\nPick the 5 most relevant results (by number) and return them as a Python list of numbers."
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    #print("response: ", response)
    import json
    import re
    # Extract JSON from response
    match = re.search(r'\[.*\]', response.choices[0].message.content, re.DOTALL)
    if match:
        reranked = json.loads(match.group(0))
    else:
        reranked = []
    # Attach scores if possible

    print("reranked: ", reranked, flush=True)

    def get_unique_id(entity):
        pk = entity.get('primary_key')
        content = entity.get('content')
        if isinstance(content, list) and content:
            content = content[0]
        return pk or content

    #id2score = {get_unique_id(c['entity']): c.get('score', 0) for c in candidates}
    #print("id2score: ", id2score)
    results = []
    i = 0
    for doc in reranked:
        pk = doc.get('primary_key') #or doc.get('content')
        #if i == 0:
        #    print("candidates.get(pk): ", candidates.get(pk))
        #    print("type(candidates.get(pk)): ", type(candidates.get(pk)))
        #    print("candidates.get(pk)[1]: ", candidates.get(pk)[1])
        #i += 1
        results.append(candidates)
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

# --- API Endpoint ---
@app.post("/query", response_model=RerankResponse)
def query_endpoint(req: QueryRequest):
    sparse_results = search_sparse(req.query, req.top_k)
    dense_results = search_dense(req.query, req.top_k)
    reranked = combine_and_rerank(sparse_results, dense_results, req.query, req.rerank_top_n)
    return {"results": reranked} 

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
def search(query: str, top_k: int = 5):
    try:
        print("Received /search request with query:", query, flush=True)
        print("Starting dense search", flush=True)
        dense_results = search_dense(query, top_k)
        #print("dense_results: ", dense_results, flush=True)
        print("Dense search complete", flush=True)
        print("Starting sparse search", flush=True)
        sparse_results = search_sparse(query, top_k)
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
        return {"error": str(e)} 
