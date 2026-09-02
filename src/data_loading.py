import pathlib
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PERMITTED_EXTENSIONS = {".py", ".md", ".rst", ".txt", ".json", ".yaml", ".yml"}
SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".venv"}
data = pathlib.Path("vllm-0.10.1")


def retrieve_files(path: pathlib.Path) -> list[pathlib.Path]:
    try:
        list_of_files = []
        for item in path.rglob("*"):
            if item.is_file() and item.suffix in PERMITTED_EXTENSIONS and not any(skip_dir in item.parts for skip_dir in SKIP_DIRS):
                list_of_files.append(item)
        return list_of_files
    except BaseException as e:
        print(e)
        exit(1)

def load_and_split(list_of_files: list[pathlib.Path], chunk_size) -> List[Document]:
    try:
        if chunk_size <= 0 or chunk_size > 2000:
            raise ValueError("chunk_size must be between 1 and 2000")
        documents: List[Document] = []
        splitter = RecursiveCharacterTextSplitter(
            add_start_index=True,
            chunk_size=chunk_size,
            chunk_overlap=0
            )
        for file_path in list_of_files:
            try:
                loader = TextLoader(file_path, autodetect_encoding=True)
                loaded_documents = loader.load()
                documents.extend(splitter.split_documents(loaded_documents))
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
        return documents
    except BaseException as e:
        print(e)
        exit(2) 


# def read_file(list_of_files: list[pathlib.Path], chunk_size: int = 200) -> str:
#     documents = []
#     for file_path in list_of_files:
#         try:
#             with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
#                 content = f.read()
#                 if content.strip():  # Only add non-empty files
#                     documents.extend([content[i:i + chunk_size] for i in range(0, len(content), chunk_size)])
#         except Exception as e:
#             print(f"Warning: Could not read {file_path}: {e}")
#     return documents

def main():
    files = retrieve_files(data)
    print(f"Found {len(files)} files")
    for f in files[:5]:
        print(f"  {f}")

    documents = load_and_split(files)
    print(f"\nProduced {len(documents)} chunks")
    if documents:
        print("\nFirst chunk preview:")
        print(f"  source: {documents[0].metadata.get('source')}")
        print(f"  content: {documents[0].page_content[:200]!r}")


if __name__ == "__main__":
    main()