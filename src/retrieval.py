import time
import bm25s


def retrieval(query: str, k: int):
    try:
        retriever = bm25s.BM25.load("data/processed/bm25_index")
        query_tokens = bm25s.tokenize(query)
        results, scores = retriever.retrieve(query_tokens, k=k)
        return results, scores
    except Exception as e:
        print(f"Error during retrieval: {e}, line: {e.__traceback__.tb_lineno}")
        exit(4)
