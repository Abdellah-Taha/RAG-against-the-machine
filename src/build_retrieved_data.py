from data_models import MinimalSearchResults, MinimalSource, StudentSearchResults
from typing import List
from retrieval import retrieval
from indexing import index_files
import time

def build_retrieved_data(query: str, k: int, meta_data: List[dict]):
    minimal_search_results = MinimalSearchResults(question_id="", question=query, retrieved_sources=[])
    results, scores = retrieval(query, k)
    for i in range(len(results[0])):
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

def main():
    queries = ["What activation formats does the fused batched MoE layer return in vLLM?",
               "What are the default values for FP8_MIN and FP8_MAX constants in vLLM's triton_flash_attention module?",
               "What determines whether vLLM's sampler returns Pythonized results or deferred Pythonization arguments?"]
    k = 5
    start_time = time.time()
    meta_data = index_files()
    end_time = time.time()
    print(f"Indexing complete in {end_time - start_time:.2f} seconds. You can now use the BM25 retriever for searching.")
    student_search_results = total_search_results(queries, k, meta_data)
    for result in student_search_results.search_results:
        print(f"Question: {result.question}")
        print(f"Retrieved Sources:")
        for source in result.retrieved_sources:
            print(f"  File Path: {source.file_path}, Start Index: {source.first_character_index}, End Index: {source.last_character_index}")
            
    
if __name__ == "__main__":
    main()
