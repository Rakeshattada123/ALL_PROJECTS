import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Custom modules
from rag_engine import RAGQueryEngine, RAG_Response as EngineResponse

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration and Initialization ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "rag-crow-story"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

if not all([GOOGLE_API_KEY, PINECONE_API_KEY]):
    raise ValueError("API keys for Google and Pinecone must be set in the .env file.")

# --- Initialize Models and Services (on startup) ---
try:
    print("Initializing models and services...")
    # Configure Gemini
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')

    # Configure Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(INDEX_NAME)

    # Load Embedding Model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Initialize RAG Query Engine
    query_engine = RAGQueryEngine(
        llm_model=gemini_model,
        embedding_model=embedding_model,
        pinecone_index=pinecone_index
    )
    print("Initialization complete. API is ready.")
except Exception as e:
    print(f"FATAL: Could not initialize models. Error: {e}")
    # In a real app, you might want to exit or handle this more gracefully
    query_engine = None


# --- FastAPI Application ---
app = FastAPI(
    title="RAG Query API",
    description="An API for querying a story about a crow using RAG.",
    version="1.0.0"
)

# --- Pydantic Models for API validation ---
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    retrieved_context: list[str]
    context_scores: list[float]


@app.on_event("startup")
async def startup_event():
    if query_engine is None:
        # This prevents the app from starting if initialization failed
        raise RuntimeError("RAG Query Engine failed to initialize. Check logs for errors.")
    # You can also add a check to ensure the Pinecone index is not empty
    stats = pinecone_index.describe_index_stats()
    if stats['total_vector_count'] == 0:
        print("WARNING: Pinecone index is empty. Run the setup_index.py script.")


@app.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def handle_query(request: QueryRequest):
    """
    Accepts a question and returns a generated answer based on retrieved context.
    """
    if not request.question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty."
        )

    try:
        # Use the engine to get a structured response
        engine_response: EngineResponse = query_engine.query(
            question=request.question,
            top_k=request.top_k
        )
        # Convert the engine's dataclass response to the Pydantic response model
        return QueryResponse(
            answer=engine_response.answer,
            retrieved_context=engine_response.retrieved_context,
            context_scores=engine_response.context_scores
        )
    except Exception as e:
        print(f"An error occurred during query processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the query."
        )

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "RAG API is running. Go to /docs for the API documentation."}