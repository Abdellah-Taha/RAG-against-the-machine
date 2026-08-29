import bm25s
import pathlib
import time
import io

from data_loading import retrieve_files, read_file

data_path = pathlib.Path("vllm-0.10.1")

def index_files():
    sample = retrieve_files(data_path)
    
    documents = read_file(sample)
    
    corpus = bm25s.tokenize(documents)
    
    retriever = bm25s.BM25()
    retriever.index(corpus)

if __name__ == "__main__":
    start_time = time.time()
    index_files()
    end_time = time.time()
    print(f"Indexing complete in {end_time - start_time:.2f} seconds. You can now use the BM25 retriever for searching.")