import os
import glob
import ast
import argparse
from git import Repo
from sklearn.feature_extraction.text import TfidfVectorizer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from pymilvus import MilvusClient
from tqdm import tqdm
import openai
import json
import time


# Milvus connection
MILVUS_HOST = os.getenv("MILVUS_HOST", "in03-874be76b9aa0be7.serverless.gcp-us-west1.cloud.zilliz.com")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "268c3796886a41827afcee6560f083fbfc4992ae7265598b4d3582979748054380929293cd76ea79244845abf9773e4e9128de0e")
MILVUS_COLLECTION_SPARSE = os.getenv("MILVUS_COLLECTION_SPARSE", "githubSparseVectorRag")
MILVUS_COLLECTION_DENSE = os.getenv("MILVUS_COLLECTION_DENSE", "githubDenseVectorRag")

class MilvusVectorDB:
    def __init__(self, milvus_client, collection_name, vectorizer=None, is_sparse=False):
        self.milvus_client = milvus_client
        self.collection = collection_name
        self.vectorizer = vectorizer
        self.is_sparse = is_sparse
    
    def _get_embedding(self, content):
        """Get embedding for content based on collection type"""
        if self.is_sparse:
            # For sparse vectors, use TF-IDF
            if self.vectorizer:
                sparse_vector = self.vectorizer.transform([content])
                indices = sparse_vector.indices.tolist()
                values = sparse_vector.data.tolist()
                # Milvus expects {index: value, ...}
                return {int(idx): float(val) for idx, val in zip(indices, values)}
        else:
            # For dense vectors, use OpenAI (new API)
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.embeddings.create(input=[content], model="text-embedding-3-large", dimensions=3072)
            return response.data[0].embedding
    
    def add_document(self, content, metadata):
        """Add a single document to the collection"""
        try:
            embedding = self._get_embedding(content)
            metadata_str = json.dumps(metadata or {})
            # Generate a unique primary key (e.g., using current timestamp in ms)
            primary_key = int(time.time() * 1000)
            
            data = {
                "primary_key": primary_key,
                "vector": embedding,
                "content": [content],
                "metadata": [metadata_str]
            }
            
            self.milvus_client.insert(self.collection, data)
            return True
        except Exception as e:
            print(f"Error adding document: {e}")
            return False


# 1. Clone repo if not present
def clone_repo(repo_url, local_dir):
    if not os.path.exists(local_dir):
        print(f"Cloning {repo_url} into {local_dir}...")
        Repo.clone_from(repo_url, local_dir)
    else:
        print(f"Repo already exists at {local_dir}.")

# 2. Chunking functions
def chunk_by_file(repo_dir, file_ext=".py"):
    files = glob.glob(f"{repo_dir}/**/*{file_ext}", recursive=True)
    docs = []
    for i, f in enumerate(files):
        with open(f, encoding="utf-8", errors="ignore") as file:
            docs.append({"id": i, "file_path": f, "text": file.read()})
    return docs

def chunk_by_function(repo_dir, file_ext=".py"):
    files = glob.glob(f"{repo_dir}/**/*{file_ext}", recursive=True)
    docs = []
    doc_id = 0
    for f in files:
        with open(f, encoding="utf-8", errors="ignore") as file:
            source = file.read()
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    start = node.lineno - 1
                    end = node.end_lineno if hasattr(node, "end_lineno") else None
                    func_code = "\n".join(source.splitlines()[start:end])
                    docs.append({"id": doc_id, "file_path": f"{f}::{node.name}", "text": func_code})
                    doc_id += 1
        except Exception as e:
            continue
    return docs

# 3. Vectorize
def vectorize_docs(docs):
    texts = [doc["text"] for doc in docs]
    vectorizer = TfidfVectorizer(
        max_features=1000,        # VOCABULARY SIZE: 1000 words
        stop_words='english',     # Remove common English words
        ngram_range=(1, 2),       # Unigrams and bigrams
        min_df=0.001,            # Minimum document frequency (0.1%)
        max_df=0.999,            # Maximum document frequency (99.9%)
        lowercase=True,          # Convert to lowercase
        analyzer='word'          # Word-level analysis
    )
    vectors = vectorizer.fit_transform(texts)
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Sparsity: {1 - vectors.nnz / (vectors.shape[0] * vectors.shape[1]):.2%}")
    return vectors, docs, vectorizer

def vectorize_dense_docs(docs):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    MAX_CHARS = 8000  # Truncate long docs for safety
    texts = [doc["text"][:MAX_CHARS] for doc in docs]
    vectors = []
    batch_size = 10  # Reduce batch size to avoid token limit
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding (OpenAI)"):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(input=batch, model="text-embedding-3-large", dimensions=3072)
        vectors.extend([d.embedding for d in response.data])
    return vectors, docs

# 4. Milvus connection and collection access
def connect_to_milvus():
    """Connect to Zilliz Cloud using MilvusClient"""
    try:
        milvus_client = MilvusClient(
            uri=f"https://{MILVUS_HOST}",
            token=MILVUS_TOKEN
        )
        print(f"Successfully connected to Milvus Cloud at {MILVUS_HOST}")
        return milvus_client
    except Exception as e:
        print(f"Failed to connect to Milvus Cloud: {e}")
        print("Please check:")
        print("1. Your internet connection")
        print("2. The Milvus Cloud host and token are correct")
        print("3. Your Milvus Cloud instance is running")
        raise

def init_sparse_collection(dimension, vectorizer=None):
    """Initialize the sparse collection using MilvusVectorDB"""
    milvus_client = connect_to_milvus()
    
    # Check if collection exists and get its schema
    if milvus_client.has_collection(MILVUS_COLLECTION_SPARSE):
        print(f"✅ Using existing collection: {MILVUS_COLLECTION_SPARSE}")
        try:
            schema = milvus_client.describe_collection(MILVUS_COLLECTION_SPARSE)
            print(f"Collection schema: {schema}")
        except Exception as e:
            print(f"Could not get schema: {e}")
    else:
        print(f"⚠️ Collection {MILVUS_COLLECTION_SPARSE} does not exist")
        print("Please create the collection manually with sparse_float_vector field type")
        return None
    
    # Return MilvusVectorDB wrapper with the fitted vectorizer
    return MilvusVectorDB(milvus_client, MILVUS_COLLECTION_SPARSE, vectorizer, is_sparse=True)

def init_dense_collection(dimension):
    """Initialize the dense collection using MilvusVectorDB"""
    milvus_client = connect_to_milvus()
    
    if not milvus_client.has_collection(MILVUS_COLLECTION_DENSE):
        schema = {
            "fields": [
                {"name": "primary_key", "description": "PK", "type": "INT64", "is_primary": True, "autoID": True},
                {"name": "content", "description": "Content", "type": "VARCHAR", "max_length": 65535},
                {"name": "metadata", "description": "Metadata", "type": "VARCHAR", "max_length": 65535},
                {"name": "vector", "description": "OpenAI embedding vector", "type": "FLOAT_VECTOR", "dim": dimension}
            ],
            "description": "Dense OpenAI vectors collection"
        }
        milvus_client.create_collection(MILVUS_COLLECTION_DENSE, schema=schema)
        print(f"✅ Created Milvus collection: {MILVUS_COLLECTION_DENSE}")
        
        # Create IVF_FLAT index
        index_params = {
            "field_name": "vector",
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": 128}
        }
        milvus_client.create_index(MILVUS_COLLECTION_DENSE, index_params)
        print(f"✅ Created index for: {MILVUS_COLLECTION_DENSE}")
    else:
        print(f"✅ Using existing Milvus collection: {MILVUS_COLLECTION_DENSE}")
    
    # Return MilvusVectorDB wrapper for dense vectors
    return MilvusVectorDB(milvus_client, MILVUS_COLLECTION_DENSE, is_sparse=False)

# 5. Insert
def insert_to_milvus(vector_db, vectors, docs):
    """Insert data into sparse collection using vector_db.add_document()"""
    
    # Debug: Check the first vector
    first_vector = vectors[0]
    print(f"First vector type: {type(first_vector)}")
    print(f"First vector shape: {first_vector.shape}")
    print(f"First vector nnz (non-zero): {first_vector.nnz}")
    
    # Add documents one by one using vector_db.add_document()
    success_count = 0
    for i, doc in enumerate(docs):
        try:
            print(f"DEBUG: Metadata being sent to Milvus: {doc['file_path']}")
            success = vector_db.add_document(
                content=doc["text"],
                metadata=doc["file_path"]
            )
            if success:
                success_count += 1
            if i % 50 == 0:
                print(f"✅ Processed {i+1}/{len(docs)} documents")
        except Exception as e:
            print(f"❌ Error with document {i}: {e}")
            break
    
    print(f"✅ Successfully inserted {success_count}/{len(docs)} documents into sparse collection")

def insert_dense_to_milvus(vector_db, vectors, docs):
    """Insert data into dense collection using vector_db.add_document()"""
    success_count = 0
    for i, (vector, doc) in enumerate(zip(vectors, docs)):
        if len(vector) != 3072:
            print(f"ERROR: Vector length is {len(vector)}, expected 3072. Skipping doc {i} ({doc.get('file_path', '')})")
            continue
        try:
            print(f"DEBUG: Metadata being sent to Milvus: {doc['file_path']}")
            # Patch: pass the vector directly to add_document if needed, or set as attribute
            # If your add_document uses its own embedding, you may need to adjust this
            vector_db.add_document(
                content=doc["text"],
                metadata=doc["file_path"]
            )
            success_count += 1
            if i % 50 == 0:
                print(f"✅ Processed {i+1}/{len(docs)} documents")
        except Exception as e:
            print(f"❌ Error with document {i}: {e}")
            break
    print(f"✅ Successfully inserted {success_count}/{len(docs)} documents into dense collection")

# 6. Main
def main():
    parser = argparse.ArgumentParser(description="Load Cython repo into Milvus with both TF-IDF sparse and OpenAI dense vectors.")
    parser.add_argument('--chunking', choices=['file', 'function'], default='file', help='Chunking strategy: by file or by function')
    parser.add_argument('--repo_url', default='https://github.com/cython/cython', help='GitHub repo URL')
    parser.add_argument('--repo_dir', default='./cython', help='Local directory for repo')
    args = parser.parse_args()
    
    #Clone the repo
    #clone_repo(args.repo_url, args.repo_dir)

    if args.chunking == "file":
        docs = chunk_by_file(args.repo_dir)
    else:
        docs = chunk_by_function(args.repo_dir)
    print(f"Chunked {len(docs)} documents.")

    # Create sparse vectors (TF-IDF)
    sparse_vectors, docs, vectorizer = vectorize_docs(docs)
    print(f"Vectorized documents (TF-IDF). Vector dim: {sparse_vectors.shape[1]}")
    sparse_vector_db = init_sparse_collection(sparse_vectors.shape[1], vectorizer)
    if sparse_vector_db:
        print(f"Inserting into Milvus sparse collection '{MILVUS_COLLECTION_SPARSE}'...")
        insert_to_milvus(sparse_vector_db, sparse_vectors, docs)

    # Clean docs: only keep those with non-empty, non-whitespace text
    docs = [doc for doc in docs if isinstance(doc["text"], str) and doc["text"].strip()]
    print(f"After cleaning, {len(docs)} documents remain.")

    # Create dense vectors (OpenAI)
    dense_vectors, docs = vectorize_dense_docs(docs)
    print(f"Vectorized documents (OpenAI). Vector dim: {len(dense_vectors[0]) if dense_vectors else 0}")
    dense_vector_db = init_dense_collection(len(dense_vectors[0]))
    print(f"Inserting into Milvus dense collection '{MILVUS_COLLECTION_DENSE}'...")
    insert_dense_to_milvus(dense_vector_db, dense_vectors, docs)

if __name__ == "__main__":
    main()