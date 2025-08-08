import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain.docstore.document import Document as LangchainDocument

# Ensure environment variable is set
if "GOOGLE_API_KEY" not in os.environ:
    raise EnvironmentError("GOOGLE_API_KEY environment variable not set.")

# Directory for Chroma DB
CHROMA_DIR = "chroma_db"

def embed_and_store(documents: list[LangchainDocument]):
    # Set up Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create and persist vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    
    # Save the DB to disk
    vectorstore.persist()
    print(f"Stored {len(documents)} documents in Chroma DB at '{CHROMA_DIR}'")


    ##Function call used for testing

if __name__ == "__main__":
    # Import your pre-processed chunks
    from document_chunker import all_chunks  # using existing document_chunker.py

    if not all_chunks:
        print("No chunks to embed.")
    else:
        embed_and_store(all_chunks)
