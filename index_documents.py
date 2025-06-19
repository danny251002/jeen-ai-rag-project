import os
import psycopg2
from dotenv import load_dotenv
import argparse
import pypdf
import docx
from typing import List, Tuple
import google.generativeai as genai
import re
from psycopg2.extras import execute_values # --- NEW: More efficient for bulk inserts ---


# --- Part 1: Load Environment Variables & Config ---
print("Loading environment variables...")
load_dotenv() 

POSTGRES_URL = os.getenv("POSTGRES_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Configure the Gemini API ---
print("Configuring Google Gemini API...")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")
genai.configure(api_key=GEMINI_API_KEY)

# --- Part 2: Database Functions ---
def get_db_connection():
    print("Attempting to connect to the database...")
    try:
        conn = psycopg2.connect(POSTGRES_URL)
        print("Database connection successful!")
        return conn
    except psycopg2.OperationalError as e:
        print(f"CRITICAL ERROR: Could not connect to the database.")
        print(f"Error details: {e}")
        return None

def setup_database(conn):
    with conn.cursor() as cur:
        print("Ensuring 'vector' extension is enabled...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("Creating 'documents' table if it doesn't exist...")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding VECTOR(768) NOT NULL,
            split_strategy VARCHAR(50),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """)
        # --- NEW: Add an index for faster similarity search ---
        print("Creating index on 'embedding' column if it doesn't exist...")
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_embedding 
        ON documents 
        USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = 100);
        """)
        conn.commit()
    print("Database setup completed successfully.")

# --- Part 3: File and Text Processing Functions ---
def extract_text_from_file(file_path: str) -> str:
    print(f"Extracting text from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file was not found at: {file_path}")
    _, file_extension = os.path.splitext(file_path)
    text_content = ""
    if file_extension.lower() == '.pdf':
        with open(file_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text_content += page.extract_text() or ""
    elif file_extension.lower() == '.docx':
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text_content += para.text + '\n'
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Please use PDF or DOCX.")
    print(f"Successfully extracted {len(text_content)} characters.")
    return text_content

def split_text_by_sentences_simple(text: str, sentences_per_chunk: int = 3) -> List[str]:
    print(f"Splitting text into chunks of {sentences_per_chunk} sentences each...")
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)
    print(f"Successfully created {len(chunks)} chunks.")
    return chunks

# --- Part 4: Embedding and Database Insertion ---
def get_embedding(text: str) -> List[float]:
    try:
        result = genai.embed_content(model="models/embedding-001", content=text, task_type="RETRIEVAL_DOCUMENT")
        return result['embedding']
    except Exception as e:
        print(f"ERROR: Failed to generate embedding for a chunk. API Error: {e}")
        return None

def insert_data(conn, data_to_insert: List[Tuple]):
    """Inserts a list of tuples into the documents table."""
    with conn.cursor() as cur:
        print(f"Inserting {len(data_to_insert)} records into the database...")
        # The execute_values function is highly efficient for inserting many rows.
        execute_values(
            cur,
            "INSERT INTO documents (filename, chunk_text, embedding, split_strategy) VALUES %s",
            data_to_insert
        )
        conn.commit()
    print("Insertion complete.")

# --- Part 5: Main Execution Block (MODIFIED) ---
def main():
    parser = argparse.ArgumentParser(description="Index a document (PDF or DOCX) into the database.")
    parser.add_argument("--file", type=str, required=True, help="Path to the document file to index.")
    args = parser.parse_args()

    connection = None
    try:
        # --- Re-enabled database connection ---
        connection = get_db_connection()
        if not connection: return
        setup_database(connection)
        
        raw_text = extract_text_from_file(args.file)
        if not raw_text.strip():
            print("Warning: The extracted text is empty. No data will be indexed.")
            return

        text_chunks = split_text_by_sentences_simple(raw_text)
        
        print("\nGenerating embeddings and preparing data for insertion...")
        data_to_insert = []
        for i, chunk in enumerate(text_chunks):
            clean_chunk = chunk.replace("\n", " ").strip()
            if not clean_chunk:
                print(f"  - Skipping empty chunk {i+1}")
                continue

            print(f"  - Processing chunk {i+1}/{len(text_chunks)}...")
            embedding = get_embedding(clean_chunk)
            if embedding:
                filename = os.path.basename(args.file)
                split_strategy = "sentence_split_simple"
                # Create a tuple that matches the INSERT statement
                data_to_insert.append((filename, clean_chunk, embedding, split_strategy))
        
        if not data_to_insert:
            print("\nCRITICAL: No data was prepared for insertion. Halting.")
            return
        
        # --- Call the insertion function ---
        insert_data(connection, data_to_insert)
        
        print("\n--- Document successfully indexed! ---")

    except (FileNotFoundError, ValueError, psycopg2.Error) as e:
        print(f"An ERROR occurred: {e}")
    finally:
        if connection:
            connection.close()
            print("Database connection closed.")


if __name__ == "__main__":
    main()