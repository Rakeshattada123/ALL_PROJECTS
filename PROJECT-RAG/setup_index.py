import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rag_engine import process_and_chunk_data # Import helper function

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "rag-crow-story"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
DATA_FILE_PATH = "transcript_with_custom_emotions.txt"

def setup():
    """
    Connects to Pinecone, creates an index if it doesn't exist,
    and upserts the data from the transcript file.
    """
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set in the environment variables.")

    # --- Initialize Models and Services ---
    print("Initializing services...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embedding_dimension = embedding_model.get_sentence_embedding_dimension()

    # --- Create Pinecone Index if it doesn't exist ---
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=embedding_dimension,
            metric='cosine',
        )
        time.sleep(2) # wait for index to be ready
        print("Index created successfully.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")

    # --- Connect to the index ---
    index = pc.Index(INDEX_NAME)
    print("\nIndex Stats:")
    print(index.describe_index_stats())

    # --- Check if data needs to be upserted ---
    if index.describe_index_stats()['total_vector_count'] > 0:
        print("\nIndex already contains data. Skipping upsert.")
        return

    # --- Process and Upsert Data ---
    print("\nIndex is empty. Processing and upserting data...")
    try:
        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except FileNotFoundError:
        print(f"Error: The data file was not found at '{DATA_FILE_PATH}'")
        return

    story_chunks = process_and_chunk_data(raw_text)
    print(f"Successfully processed {len(story_chunks)} chunks of text.")

    # Generate embeddings
    embeddings = embedding_model.encode(story_chunks, show_progress_bar=True)

    # Prepare data for upsert
    vectors_to_upsert = [
        {"id": f"chunk_{i}", "values": embedding.tolist(), "metadata": {"text": chunk}}
        for i, (chunk, embedding) in enumerate(zip(story_chunks, embeddings))
    ]

    # Upsert data in batches
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        index.upsert(vectors=vectors_to_upsert[i:i + batch_size])

    print("\nUpsert complete! Waiting for vectors to be indexed...")
    time.sleep(5)
    print("\nUpdated Index Stats:")
    print(index.describe_index_stats())

if __name__ == "__main__":
    setup()