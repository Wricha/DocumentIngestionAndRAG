import requests
import time

FASTAPI_URL = "http://127.0.0.1:8000"
session_id = "test_verification_session_99"

print("1. Creating unique sample document...")
sample_text = "The secret passphrase for the 2026 conference is 'Nebula-Galaxy-99'. Richa's favorite color is deep violet."
files = {"file": ("secret_report.txt", sample_text.encode('utf-8'), "text/plain")}
data = {"session_id": session_id}

print("2. Uploading and embedding document locally into SQLite...")
res_upload = requests.post(f"{FASTAPI_URL}/ingest/upload", files=files, data=data)
if res_upload.status_code == 200:
    print("✅ Ingestion response:", res_upload.json())
else:
    print("❌ Ingestion failed:", res_upload.status_code, res_upload.text)
    exit(1)

print("\n3. Querying RAG system with Gemini and SQLite search...")
payload = {
    "query": "What is the secret passphrase for the 2026 conference and what is Richa's favorite color?",
    "session_id": session_id,
    "top_k": 3,
    "search_mode": "never"
}

res_chat = requests.post(f"{FASTAPI_URL}/rag/chat", json=payload)
if res_chat.status_code == 200:
    chat_data = res_chat.json()
    print("\n✅ Chat response received successfully!")
    print("\n[Reasoning Trace]:")
    print(chat_data.get("reasoning_trace"))
    print("\n[Answer]:")
    print(chat_data.get("answer"))
    print("\n[Sources]:")
    for s in chat_data.get("sources", []):
        print(f" - {s.get('metadata', {}).get('source')}: \"{s.get('metadata', {}).get('text_preview')}\" (Score: {s.get('score'):.2f})")
else:
    print("❌ Chat failed:", res_chat.status_code, res_chat.text)
