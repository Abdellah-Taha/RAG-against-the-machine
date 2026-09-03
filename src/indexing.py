from typing import List
from data_loading import retrieve_files, load_and_split
from langchain_core.documents import Document
import bm25s, pathlib


data_path = pathlib.Path("data/raw/vllm-0.10.1")

def index_files(chunk_size: int) -> List[dict]:
    try:
        sample = retrieve_files(data_path)
        documents: List[Document] = load_and_split(sample, chunk_size)
        content = []
        metadata = []
        for document in documents:    
            content.append(document.page_content)
            metadata.append({
            "file_path": document.metadata["source"],
            "start": document.metadata["start_index"],
            "end": document.metadata["start_index"] + len(document.page_content),
            })
            #add metadata list to track the result of the retriever
            
        corpus = bm25s.tokenize(content)
        indexer = bm25s.BM25()
        indexer.index(corpus)
        indexer.save("data/processed/bm25_index")
        return metadata
    except Exception as e:
        print(f"Error during indexing: {e}")
        exit(3)
