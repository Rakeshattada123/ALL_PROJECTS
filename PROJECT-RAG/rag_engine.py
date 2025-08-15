import re
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

# --- Data Processing Function (from notebook) ---

def process_and_chunk_data(text_content: str):
    """Cleans the raw text from the file and chunks it."""
    lines = text_content.splitlines()
    # Skip the header (Title, ---, etc.)
    # Adjust this index if your header is longer/shorter
    content_lines = [line for line in lines if line.startswith('[')]

    cleaned_chunks = []
    for line in content_lines:
        # Regex to remove the pattern `[timestamp] emoji `
        cleaned_line = re.sub(r'^\\[.*?\\]\\s\\S\\s', '', line).strip()
        if cleaned_line:  # Only add non-empty lines
            cleaned_chunks.append(cleaned_line)

    return cleaned_chunks

# --- RAG Query Engine (Refactored for API) ---

PROMPT_TEMPLATE = """
You are a helpful and precise assistant who answers questions based ONLY on the provided context.
Do not use any information outside of the given context.
If the answer is not found in the context, respond with "I cannot find the answer in the provided story."

CONTEXT:
---
{context}
---

QUESTION:
{question}

ANSWER:
"""

@dataclass
class RAG_Response:
    """Dataclass for a structured RAG response."""
    answer: str
    retrieved_context: list[str]
    context_scores: list[float]


class RAGQueryEngine:
    def __init__(self, llm_model, embedding_model, pinecone_index):
        """Initializes the RAG Query Engine."""
        self.llm = llm_model
        self.embedder = embedding_model
        self.index = pinecone_index
        self.prompt_template = PROMPT_TEMPLATE
        print("RAG Query Engine Initialized.")

    def query(self, question: str, top_k: int = 3) -> RAG_Response:
        """
        Performs retrieval and generation, returning a structured response.
        """
        # 1. Retrieval Phase
        print(f"Retrieving top-{top_k} relevant chunks for: '{question}'")
        query_embedding = self.embedder.encode([question])[0].tolist()

        retrieval_results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # Build context from retrieved chunks
        context_chunks = []
        context_scores = []
        if retrieval_results['matches']:
            for item in retrieval_results['matches']:
                context_chunks.append(item['metadata']['text'])
                context_scores.append(round(item['score'], 4))
        
        context_str = "\\n".join(context_chunks)
        print("Retrieved Context Chunks (with scores):")
        for text, score in zip(context_chunks, context_scores):
            print(f"(Score: {score}) {text}")
        print("-" * 20)

        # 2. Generation Phase
        print("Generating answer with Gemini 1.5 Pro...")
        formatted_prompt = self.prompt_template.format(context=context_str, question=question)

        try:
            response = self.llm.generate_content(formatted_prompt)
            final_answer = response.text
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            final_answer = "An error occurred during generation."

        return RAG_Response(
            answer=final_answer,
            retrieved_context=context_chunks,
            context_scores=context_scores
        )