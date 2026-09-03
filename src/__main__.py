import argparse
import time, tqdm
from search_dataset import review_results
from indexing import index_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max_chunk_size", type=int, default=2000)
    parser.add_argument("--dataset_path", type=str, default="datasets_public/public/AnsweredQuestions/dataset_docs_public.json")
    args = parser.parse_args()
    start_time = time.time()
    meta_data = index_files(args.max_chunk_size)
    end_time = time.time()
    print(f"Indexing complete in {end_time - start_time:.2f} sec    onds. You can now use the BM25 retriever for searching.")
    review_results(args.dataset_path, args.k, meta_data=meta_data)
    print("==================================================")
    review_results("datasets_public/public/AnsweredQuestions/dataset_code_public.json", args.k, meta_data=meta_data)
    

if __name__ == "__main__":
    main()