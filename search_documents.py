import os
import psycopg2
from dotenv import load_dotenv
import argparse
import google.generativeai as genai
from typing import List
# Note: numpy is required by pgvector's adapter for some operations.
# Let's import it to be safe, although we will use a string cast here.
import numpy as np

# --- Part 1: Load Environment Variables & Config ---
print("Loading environment variables and configuring API...")
load_dotenv() 

POSTGRES_URL = os.getenv("POSTGRES_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY or not POSTGRES_URL:
    raise ValueError("GEMINI_API_KEY and POSTGRES_URL must be set in the .env file.")

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# --- Part 2: Database and Embedding Functions ---
def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    print("Connecting to the database...")
    try:
        conn = psycopg2.connect(POSTGRES_URL)
        return conn
    except psycopg2.OperationalError as e:
        print(f"CRITICAL ERROR: Could not connect to database: {e}")
        return None

def get_query_embedding(text: str) -> List[float]:
    """Generates an embedding for a given user query."""
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="RETRIEVAL_QUERY"
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return None

def search_similar_documents(conn, embedding: List[float], top_k: int = 5):
    """Searches for the top_k most similar documents in the database."""
    with conn.cursor() as cur:
        # --- KEY FIX ---
        # We must cast the Python list to a string. pgvector is smart enough
        # to parse this string as a vector for the comparison.
        embedding_str = str(embedding)
        
        cur.execute(
            """
            SELECT 
                chunk_text, 
                (1 - (embedding <=> %s)) as similarity_score
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT %s;
            """,
            (embedding_str, embedding_str, top_k)
        )
        results = cur.fetchall()
        return results

# --- Part 3: Main Execution Block ---
def main():
    parser = argparse.ArgumentParser(description="Search for relevant document chunks based on a query.")
    parser.add_argument("--query", type=str, required=True, help="The search query text.")
    args = parser.parse_args()

    connection = None
    try:
        print(f"\nGenerating embedding for your query: '{args.query}'")
        query_embedding = get_query_embedding(args.query)
        
        if not query_embedding:
            print("Could not generate an embedding for the query. Aborting.")
            return

        connection = get_db_connection()
        if not connection:
            return

        print(f"Searching for the top 5 most relevant chunks...")
        search_results = search_similar_documents(connection, query_embedding, top_k=5)

        print("\n--- Search Results ---")
        if not search_results:
            print("No relevant documents found.")
        else:
            for i, (text, score) in enumerate(search_results):
                print(f"\n--- Result {i+1} (Similarity: {score:.4f}) ---")
                print(text)
        
        print("\n----------------------")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if connection:
            connection.close()
            print("\nDatabase connection closed.")

if __name__ == "__main__":
    main()