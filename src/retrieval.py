import time
import bm25s
from data_loading import *
from indexing import index_files


def retrieval(query: str, k: int):
    try:
        retriever = bm25s.BM25.load("data/processed/bm25_index")
        query_tokens = bm25s.tokenize(query)
        results, scores = retriever.retrieve(query_tokens, k=k)
        return results, scores
    except Exception as e:
        print(f"Error during retrieval: {e}, line: {e.__traceback__.tb_lineno}")
        exit(4)

def main():
    query="What activation formats does the fused batched MoE layer return in vLLM?"
    k=5
    start_time = time.time()
    meta_data = index_files()
    end_time = time.time()
    print(f"Indexing complete in {end_time - start_time:.2f} seconds. You can now use the BM25 retriever for searching.")
    results, scores = retrieval(query, k)
    print(f"Top {k} results for query: '{query}'")
    for i in range(len(results[0])):
        print(f"\nResult {i + 1}:")
        # print(f"Score: {score}")
        # print(f"Content: {result}")
        print(f"{meta_data[i]['file_path']}, [{meta_data[i]['start']}, {meta_data[i]['end']}]")


if __name__ == "__main__":
    main()