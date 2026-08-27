import pathlib, tqdm

PERMITTED_EXTENSIONS = {".py", ".md", ".rst", ".txt", ".json", ".yaml", ".yml"}
SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".venv"}
data = pathlib.Path("vllm-0.10.1")


def retrieve_files() -> list[pathlib.Path]:
    list_of_files = []
    for item in data.rglob("*"):
        if item.is_file() and item.suffix in PERMITTED_EXTENSIONS and not any(skip_dir in item.parts for skip_dir in SKIP_DIRS):
            list_of_files.append(item)
    return list_of_files

def 