from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
from typing import Dict, List
from contextlib import asynccontextmanager

from src.llm.model import create_qa_chain

# Global variable to store the QA chain
qa_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    global qa_chain
    try:
        # Ensure API key is set
        if "GOOGLE_API_KEY" not in os.environ:
            raise EnvironmentError("GOOGLE_API_KEY environment variable not set.")
        
        qa_chain = create_qa_chain()
        print("QA chain initialized successfully!")
    except Exception as e:
        print(f"Error initializing QA chain: {e}")
        qa_chain = None
    
    yield
    
    # Shutdown (if needed)
    pass

# Initialize FastAPI app
app = FastAPI(
    title="Document QA Bot",
    description="A FastAPI application for querying documents using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Document QA Bot API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "qa_chain_ready": qa_chain is not None
    }

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the documents with a user question
    """
    if qa_chain is None:
        raise HTTPException(
            status_code=503, 
            detail="QA chain not initialized. Please check the logs."
        )
    
    if not request.question.strip():
        raise HTTPException(
            status_code=400, 
            detail="Question cannot be empty"
        )
    
    try:
        # Process the query using the correct parameter name for ConversationalRetrievalChain
        result = qa_chain.invoke({"question": request.question})
        
        # Format the sources
        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        
        return QueryResponse(
            answer=result["answer"],  # ConversationalRetrievalChain returns "answer", not "result"
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing query: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
