import argparse
import time
from search_dataset import retrieve_questions, review_results
from indexing import index_files
from llm_call import call_llm, call_llm_foreach_query, json_dump_search_and_answers, json_dump_search_results
# from retrieval import retrieval
from build_retrieved_data import total_search_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max_chunk_size", type=int, default=2000)
    parser.add_argument("--dataset_path", type=str, default="datasets_public/public/AnsweredQuestions/dataset_docs_public.json")
    args = parser.parse_args()
    start_time = time.time()
    #indexing the data files
    meta_data = index_files(args.max_chunk_size)
    end_time = time.time()
    print(f"Indexing complete in {end_time - start_time:.2f} seconds. You can now use the BM25 retriever for searching.")
    #retrieving the questions from the json file
    questions = retrieve_questions(args.dataset_path)
    # retrieving the relevent data for each query (RA ANA LI KANTB HACHI MACHI AI AW9S) 
    search_results = total_search_results(questions, args.k, meta_data=meta_data)
    #sending the search results to the llm
    llm = call_llm()
    start = time.time()
    student_result_and_answers = call_llm_foreach_query(search_results)
    end = time.time()
    print(f"\nTime taken: {end - start:.2f}")
    json_dump_search_and_answers(student_result_and_answers, "data/output/search_results_and_answer/output.json")
    json_dump_search_results(student_result_and_answers, "data/output/search_results/output.json")
    review_results("datasets_public/public/AnsweredQuestions/dataset_docs_public.json", args.k, meta_data)

if __name__ == "__main__":
    main()