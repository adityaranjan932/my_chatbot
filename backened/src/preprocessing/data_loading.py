# import os
# from langchain.document_loaders import (
#     PyPDFLoader,
#     UnstructuredWordDocumentLoader,
#     UnstructuredEmailLoader
# )

# def load_file_auto(file_path):
#     ext = os.path.splitext(file_path)[1].lower()

#     if ext == ".pdf":
#         loader = PyPDFLoader(file_path)
#     elif ext == ".docx":
#         loader = UnstructuredWordDocumentLoader(file_path)
#     elif ext in [".msg", ".eml"]:
#         loader = UnstructuredEmailLoader(file_path)
#     else:
#         raise ValueError(f"Unsupported file type: {ext}")

#     return loader.load()


# # Directory containing files
# data_dir = "backend/data"
# file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

# all_docs = []

# for path in file_paths:
#     try:
#         docs = load_file_auto(path)
#         all_docs.extend(docs)
#     except Exception as e:
#         print(f"Failed to load {path}: {e}")

# # Print snippet
# for doc in all_docs:
#     print(doc.page_content[:200])  # First 200 characters

import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredEmailLoader
)

def load_file_auto(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    elif ext in [".msg", ".eml"]:
        loader = UnstructuredEmailLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return loader.load()

