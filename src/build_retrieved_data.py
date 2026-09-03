from data_models import MinimalSearchResults, MinimalSource, StudentSearchResults
from typing import List
from retrieval import retrieval
from functools import lru_cache
import time

@lru_cache(maxsize=128)
def cached_retrieval(query: str, max_k=50):
    return retrieval(query, max_k)

def build_retrieved_data(query: str, k: int, meta_data: List[dict]):
    minimal_search_results = MinimalSearchResults(question_id="", question=query, retrieved_sources=[])
    results, scores = cached_retrieval(query, max_k=50)
    
    limit = min(k, len(results[0]))
    
    for i in range(limit):
        doc_idx = results[0][i]
        minimal_source = MinimalSource(
            file_path=meta_data[doc_idx]['file_path'],
            first_character_index=meta_data[doc_idx]['start'],
            last_character_index=meta_data[doc_idx]['end'],
        )
        minimal_search_results.retrieved_sources.append(minimal_source)
        
    return minimal_search_results

def total_search_results(queries: List[str], k: int, meta_data: List[dict]):
    student_search_results = StudentSearchResults(search_results=[], k=k)
    for query in queries:
        search_result = build_retrieved_data(query, k, meta_data)
        student_search_results.search_results.append(search_result)
    return student_search_results
