from typing import List
from data_loading import retrieve_files, load_and_split
from langchain_core.documents import Document
import bm25s, pathlib
import chromadb

BATCH_SIZE = 3500
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
            
        corpus = bm25s.tokenize(content)
        indexer = bm25s.BM25()
        indexer.index(corpus)
        indexer.save("data/processed/bm25_index")
        return metadata
    except Exception as e:
        print(f"Error during indexing: {e}")
        exit(3)

def chromadb_indexing(chunk_size: int):
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
            "chunk_id": document.metadata["source"] + "_" + str(document.metadata["start_index"])
            })
        chunk_ids = [doc.metadata["source"] + "_" + str(doc.metadata["start_index"]) for doc in documents]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Duplicate chunk_id found in metadata. Each chunk must have a unique chunk_id.")
        client = chromadb.Client()
        chromadb.PersistentClient(path="data/processed/chroma_index")
        collection = client.get_or_create_collection(name="rag_collection")
        for i in range(0, len(content), BATCH_SIZE):
            batch_content = content[i:i + BATCH_SIZE]
            batch_metadata = metadata[i:i + BATCH_SIZE]
            batch_ids = [str(i) for i in range(i, i + len(batch_content))]
            collection.add(
                documents=batch_content,
                metadatas=batch_metadata,
                ids=batch_ids
            )
        return metadata
    except Exception as e:
        print(f"Error during ChromaDB indexing: {e}")
        exit(3)

import time
def main():
    start_time = time.time()
    chunk_size = 2000
    index_files(chunk_size)
    chromadb_indexing(chunk_size)
    end_time = time.time()
    print(f"Indexing complete in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()