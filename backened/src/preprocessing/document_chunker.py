import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from data_loading import load_file_auto  

# Set up text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

# Path to documents
data_dir = "../../data"
file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if not f.startswith(".")]

# Load and chunk documents
all_chunks = []

for path in file_paths:
    try:
        docs = load_file_auto(path)
        chunks = text_splitter.split_documents(docs)
        all_chunks.extend(chunks)
    except Exception as e:
        print(f"Failed to process {path}: {e}")

# Preview first 5 chunks
for i, chunk in enumerate(all_chunks[:5]):
    print(f"\n--- Chunk {i+1} ---\n{chunk.page_content[:500]}")

