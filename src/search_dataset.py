from typing import List
from indexing import index_files
from build_retrieved_data import total_search_results
import json, time

path_code = "datasets_public/public/AnsweredQuestions/dataset_code_public.json"
path_docs = "datasets_public/public/AnsweredQuestions/dataset_docs_public.json"

def parse_data_set(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        son = json.load(f)
    
    return son.get("rag_questions", [])

def retrieve_questions(file_path: str):
    data_set = parse_data_set(file_path)
    return [item["question"] for item in data_set]

def retrieve_data_source(file_path: str):
    data_set = parse_data_set(file_path)
    return [item["sources"] for item in data_set]
    
    
def review_results(path: str, k: int, meta_data: List[dict]):
    correct_count, total = 0, 0
    data_set = retrieve_questions(path)
    source = retrieve_data_source(path)

    student_search_results = total_search_results(data_set, k, meta_data)
    for i, result in enumerate(student_search_results.search_results):
        # print(f"\nQuestion {i+1}: {result.question}")
        
        ground_truth_paths = [s["file_path"] for s in source[i]]
        student_paths = result.retrieved_sources
        matches = [path.file_path for path in student_paths if path.file_path in ground_truth_paths]
        if matches:
            # print(f"✅ Success! Found true path(s): {matches}")
            correct_count += 1
            total +=1
        else:
            # print(f"❌ Missed. Expected: {ground_truth_paths[0]}")
            total +=1
    print(f"Total Questions: {total}")
    print(f"total student answers: {len(student_search_results.search_results)}")
    print(f"recall@{k}: {correct_count}/{total}  ({(correct_count/total)*100:.2f}%)")
    
def main():
    start_time = time.time()
    meta_data = index_files()
    end_time = time.time()
    print(f"Indexing complete in {end_time - start_time:.2f} sec    onds. You can now use the BM25 retriever for searching.")
    review_results(path_docs, k=1, meta_data=meta_data)
    review_results(path_docs, k=2, meta_data=meta_data)
    review_results(path_docs, k=5, meta_data=meta_data)
    review_results(path_docs, k=10, meta_data=meta_data)
    print("==================================================")
    review_results(path_code, k=1, meta_data=meta_data)
    review_results(path_code, k=2, meta_data=meta_data)
    review_results(path_code, k=5, meta_data=meta_data)
    review_results(path_code, k=10, meta_data=meta_data)

if __name__ == "__main__":
    main()