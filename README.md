# Milvus Sparse Vector Loader for Cython Repo

This script loads data from the [Cython GitHub repository](https://github.com/cython/cython) into a Milvus sparse vector index using TF-IDF vectorization. You can experiment with chunking the data by file or by function.

## Requirements

- Python 3.7+
- Milvus instance running (local or remote)
- See `requirements.txt` for Python dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python load_to_milvus.py --chunking file      # Chunk by file (default)
python load_to_milvus.py --chunking function  # Chunk by function
```

Optional arguments:
- `--repo_url`   : GitHub repo URL (default: https://github.com/cython/cython)
- `--repo_dir`   : Local directory for repo (default: ./cython)
- `--collection` : Milvus collection name (default: cython_<chunking>)

## Notes
- The script will clone the repo if not already present.
- The script will drop and recreate the Milvus collection if it already exists.
- Make sure Milvus is running and accessible before running the script. 