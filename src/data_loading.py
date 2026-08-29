import pathlib, io

PERMITTED_EXTENSIONS = {".py", ".md", ".rst", ".txt", ".json", ".yaml", ".yml"}
SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".venv"}
data = pathlib.Path("vllm-0.10.1")

def retrieve_files(path: pathlib.Path) -> list[pathlib.Path]:
    list_of_files = []
    for item in data.rglob("*"):
        if item.is_file() and item.suffix in PERMITTED_EXTENSIONS and not any(skip_dir in item.parts for skip_dir in SKIP_DIRS):
            list_of_files.append(item)
    return list_of_files

def read_file(list_of_files: list[pathlib.Path], chunk_size: int = 200) -> str:
    documents = []
    for file_path in list_of_files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if content.strip():  # Only add non-empty files
                    documents.extend([content[i:i + chunk_size] for i in range(0, len(content), chunk_size)])
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
    return documents


# if __name__ == "__main__":
#     files = retrieve_files(data)
#     buffer = read_file(files)
#     print(buffer.getvalue().decode("utf-8", errors="ignore"))