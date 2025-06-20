# RAG Project for JEEN.AI

This project implements a simple but powerful Retrieval-Augmented Generation (RAG) system using **Python**, **Google Gemini**, and **PostgreSQL** with the **pgvector** extension.

The system can index text from PDF and DOCX files, store it as vector embeddings in a database, and then search for the most relevant text chunks based on a user's natural-language query.

---

## Features
- **File Ingestion** – Extracts text from both `.pdf` and `.docx` files.  
- **Text Chunking** – Splits extracted text into smaller, manageable chunks based on sentences.  
- **Vector Embeddings** – Uses the Google Gemini `embedding-001` model to create semantic vector representations of text.  
- **Vector Storage** – Stores text chunks and their corresponding embeddings in a PostgreSQL database, optimized with `pgvector`.  
- **Semantic Search** – Takes a user query, creates an embedding for it, and uses cosine similarity to find and return the most relevant stored text chunks.  

---

## Project Structure
    .
    ├── .env                 # Secret keys (API key, DB URL) – NOT committed to Git
    ├── .gitignore           # Files/folders ignored by Git
    ├── docs/                # (Optional) Example documents
    │   └── sample.pdf
    ├── index_documents.py   # Script to process & index a document
    ├── search_documents.py  # Script to search the indexed documents
    ├── requirements.txt     # Python dependencies
    └── README.md            # This file

---

## Setup and Installation

### 1 Prerequisites
* Python 3.8 or higher  
* PostgreSQL with the `pgvector` extension installed (Docker is recommended for convenience).

### 2 Clone the Repository
    git clone https://github.com/danny251002/jeen-ai-rag-project.git
    cd jeen-ai-rag-project

### 3 Set Up a Virtual Environment
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # macOS / Linux
    source .venv/bin/activate

### 4 Install Dependencies
    pip install -r requirements.txt

### 5 Configure Environment Variables
Create a file named `.env` in the project root and add:

    # .env file

    # Your Google Gemini API Key
    GEMINI_API_KEY="AIzaSy...your_actual_key"

    # Your PostgreSQL connection string
    # Format: postgresql://user:password@host:port/dbname
    POSTGRES_URL="postgresql://postgres:mysecretpassword@localhost:5432/postgres"

> `.gitignore` already prevents `.env` from being committed to source control.

---

## How to Use

Make sure your virtual environment is activated before running the scripts.

### 1 Indexing a Document
    python index_documents.py --file path/to/your/document.pdf
    # Example
    python index_documents.py --file docs/sample.pdf

The script connects to the database, creates the required table and index, extracts text, splits it into chunks, generates embeddings, and stores everything.

### 2 Searching for Information
    python search_documents.py --query "Your search question here"
    # Example
    python search_documents.py --query "what does the document say about the Natural Language Toolkit"

The script generates an embedding for your query, searches the database for the top 5 most similar chunks, and prints them with similarity scores.