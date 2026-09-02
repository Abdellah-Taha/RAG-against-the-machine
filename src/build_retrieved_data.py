from retrieval import *
from data_models import *


def build_retrieved_data(query: str, k: int):
    minimal_search_results = MinimalSearchResults(question_id="", question=query, retrieved_sources=[])
    minimal_source = MinimalSource(file_path="", first_character_index=0, last_character_index=0)
    start_time = time.time()
    meta_data = index_files()
    end_time = time.time()
    print(f"Indexing complete in {end_time - start_time:.2f} seconds. You can now use the BM25 retriever for searching.")
    results, scores = retrieval(query, k)
    for i in range(len(results[0])):
        minimal_source.file_path = meta_data[i]['file_path']
        minimal_source.first_character_index = meta_data[i]['start']
        minimal_source.last_character_index = meta_data[i]['end']
        minimal_search_results.retrieved_sources.append(minimal_source)
    return minimal_search_results

def total_search_results(queries: List[str], k: int):
    student_search_results = StudentSearchResults(search_results=[], k=k)
    for query in queries:
        search_result = build_retrieved_data(query, k)
        student_search_results.search_results.append(search_result)
    return student_search_results

def main():
    queries = ["What is the capital of France?", "Who wrote 'To Kill a Mockingbird'?", "What is the boiling point of water?"]
    k = 3
    student_search_results = total_search_results(queries, k)
    for result in student_search_results.search_results:
        print(f"Question: {result.question}")
        print(f"Retrieved Sources:")
        for source in result.retrieved_sources:
            print(f"  File Path: {source.file_path}, Start Index: {source.first_character_index}, End Index: {source.last_character_index}")
            
    
if __name__ == "__main__":
    main()
