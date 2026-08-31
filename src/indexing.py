import bm25s
import pathlib
import time
import io
from typing import List
from langchain_core.documents import Document


from data_loading import retrieve_files, load_and_split

data_path = pathlib.Path("../vllm-0.10.1")
chunk_size=5

def index_files():
    sample = retrieve_files(data_path)
    documents: List[Document] = load_and_split(sample, chunk_size)
    content = []
    for document in documents:    
        content.append(document.page_content)
        
    corpus = bm25s.tokenize(content)
    retriever = bm25s.BM25()
    retriever.index(corpus)

if __name__ == "__main__":
    start_time = time.time()
    index_files()
    end_time = time.time()
    print(f"Indexing complete in {end_time - start_time:.2f} seconds. You can now use the BM25 retriever for searching.")