import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

# Directory for Chroma DB - use absolute path
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "preprocessing", "chroma_db")

def load_retriever():
    # Ensure API Key is set
    if "GOOGLE_API_KEY" not in os.environ:
        raise EnvironmentError("GOOGLE_API_KEY environment variable not set.")
    
    # Reuse the same embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Check if Chroma DB directory exists
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(f"Chroma DB directory not found at: {CHROMA_DIR}")

    # Load the existing vector store
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    
    # Get the retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",   # can be "similarity", "mmr", or "similarity_score_threshold"
        search_kwargs={"k": 5}      # top 5 most relevant documents
    )
    
    return retriever

