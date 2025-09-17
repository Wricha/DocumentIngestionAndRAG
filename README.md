# Document Ingestion, RAG Chatbot & Booking System
This project is a Retrieval-Augmented Generation (RAG) chatbot combined with a document-based knowledge system and a booking functionality. Users can upload documents (PDF/TXT), query them using natural language, and manage simple bookings through the API.
# Features
- Upload PDF or TXT documents.
- Text extraction and chunking (sliding window or sentence-based).
- Embedding generation using all-MiniLM-L6-v2.
- Store embeddings in vector database (Pinecone).
- RAG-based chatbot: answers questions based only on document context.
- Session-based conversation memory using Redis.
- Interview booking
- Store booking information

# Installation
1. Clone the repository
2. Create virtual environment
3. Install dependencies
4. Set environment variables
5. Start the fast api backend

# Endpoints
## Upload Document
POST /ingest/upload
---
Form Data:
- file: PDF or TXT file
- source: Source of the document (optional)
- metadata: Custom metadata (optional)
- chunking_strategy: 'sliding' or 'sentences' (default: sliding)
- chunk_size: Max tokens per chunk (default: 500)
- overlap: Overlap between chunks (default: 50)

## Chat with RAG
POST /rag/chat
---
Json Body:
{
    "query": "Your question about the documents",
    "session_id": "session_id",
    "top_k": 3
}

## Booking Management
POST /rag/book
---
JSON Body:
{
    "user_name": "name",
    "email": "richa@example.com",
    "date": "2025-09-20",
    "time": "15:30",
}

GET /rag/book
---
PUT /rag/book/{booking_id}
---
DELETE /rag/book/{booking_id}
---



