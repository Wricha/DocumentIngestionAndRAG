import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests
import uuid

st.set_page_config(
    page_title="AuraRAG — Intelligent Web-Searching Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

    /* Global styling */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #0b0f19 !important;
        color: #f3f4f6 !important;
    }

    /* Headings */
    h1, h2, h3, .outfit-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(19, 25, 38, 0.9) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(20px);
    }

    /* Message bubbles styling */
    .message-container {
        padding: 1.2rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .user-message {
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.08), rgba(124, 77, 255, 0.08));
        border: 1px solid rgba(0, 229, 255, 0.2);
        color: #ffffff;
    }

    .assistant-message {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #e5e7eb;
    }

    /* Reasoning expander styling */
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.01) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        margin-bottom: 0.5rem;
    }
    
    /* Glowing logo */
    .glowing-logo {
        background: linear-gradient(135deg, #00e5ff, #7c4dff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
        font-family: 'Outfit', sans-serif;
        text-shadow: 0 0 30px rgba(0, 229, 255, 0.2);
    }

    /* Cards */
    .source-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
    }
    .source-card:hover {
        background: rgba(255, 255, 255, 0.04);
        border-color: rgba(0, 229, 255, 0.2);
    }

    .badge-web {
        background: rgba(0, 229, 255, 0.1);
        color: #00e5ff;
        border: 1px solid rgba(0, 229, 255, 0.2);
        padding: 0.15rem 0.4rem;
        border-radius: 5px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-right: 0.5rem;
    }

    .badge-doc {
        background: rgba(124, 77, 255, 0.1);
        color: #7c4dff;
        border: 1px solid rgba(124, 77, 255, 0.2);
        padding: 0.15rem 0.4rem;
        border-radius: 5px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-right: 0.5rem;
    }

    /* Upload zone */
    .upload-zone {
        background: rgba(124, 77, 255, 0.04);
        border: 1px dashed rgba(124, 77, 255, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        text-align: center;
    }

    /* Uploaded doc pill */
    .doc-pill {
        background: rgba(124, 77, 255, 0.08);
        border: 1px solid rgba(124, 77, 255, 0.2);
        border-radius: 8px;
        padding: 0.4rem 0.75rem;
        margin-bottom: 0.4rem;
        font-size: 0.8rem;
        color: #c4b5fd;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION MANAGEMENT -----------------
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []   # list of filenames successfully ingested

# API Backend Connection URL
FASTAPI_URL = "http://127.0.0.1:8000"


# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("<div class='glowing-logo'>AuraRAG</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.85rem; color:#9ca3af; margin-top:-0.5rem; margin-bottom:2rem;'>"
        "Real-Time Search Chatbot</p>",
        unsafe_allow_html=True
    )

    # ── Session Management ──────────────────────────────
    st.subheader("Active Session")
    st.code(st.session_state.session_id, language="text")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Copy ID"):
            st.toast("Copied session ID!")
    with col2:
        if st.button("Reset Chat"):
            st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"
            st.session_state.messages = []
            st.session_state.uploaded_docs = []
            st.rerun()

    st.markdown("---")

    # ── Document Upload ─────────────────────────────────
    st.subheader("📄 Document Upload")
    st.markdown(
        "<p style='font-size:0.8rem; color:#9ca3af;'>"
        "Upload PDFs or text files. AuraRAG will search them when answering your questions.</p>",
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Only ingest files not already uploaded this session
            if uploaded_file.name not in st.session_state.uploaded_docs:
                with st.spinner(f"Ingesting {uploaded_file.name}..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        params = {"session_id": st.session_state.session_id}
                        res = requests.post(
                            f"{FASTAPI_URL}/rag/upload",
                            files=files,
                            params=params,
                            timeout=60,
                        )
                        if res.status_code == 200:
                            data = res.json()
                            st.session_state.uploaded_docs.append(uploaded_file.name)
                            st.toast(f"✅ {uploaded_file.name} ingested ({data.get('chunks', '?')} chunks)")
                        else:
                            st.error(f"Failed to ingest {uploaded_file.name}: {res.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Upload error: {str(e)}")

    # Show ingested docs
    if st.session_state.uploaded_docs:
        st.markdown(
            f"<p style='font-size:0.8rem; color:#9ca3af; margin-top:0.5rem;'>"
            f"{len(st.session_state.uploaded_docs)} document(s) ingested:</p>",
            unsafe_allow_html=True
        )
        for doc in st.session_state.uploaded_docs:
            st.markdown(
                f"<div class='doc-pill'>📄 {doc} <span style='color:#6b7280;'>✓</span></div>",
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            "<p style='font-size:0.75rem; color:#6b7280; text-align:center; margin-top:0.5rem;'>"
            "No documents uploaded yet.</p>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── AI Model Configuration ──────────────────────────
    st.subheader("AI Model Configuration")
    gemini_model = st.selectbox(
        "Google Gemini Model",
        ["gemini-2.5-flash", "gemini-2.0-flash-thinking-exp", "gemini-2.5-pro"],
        index=0
    )

    st.subheader("Web Integration")
    search_mode = st.selectbox(
        "Web Search Mode",
        ["auto", "always", "never"],
        index=0,
        help="auto: classify if search is needed. always: force search. never: answer from general knowledge."
    )

    # Show doc search status
    st.subheader("Document Search")
    doc_search_active = len(st.session_state.uploaded_docs) > 0
    if doc_search_active:
        st.success(f"✅ Active — {len(st.session_state.uploaded_docs)} doc(s) loaded")
    else:
        st.info("Upload documents above to enable RAG")


# ----------------- MAIN CHAT UI -----------------
st.markdown(
    "<h2 style='margin-bottom: 0.2rem; font-family: Outfit, sans-serif;'>Interactions</h2>",
    unsafe_allow_html=True
)

# Welcome screen
if not st.session_state.messages:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        ### Welcome to AuraRAG Search Studio
        A next-generation conversational chatbot powered entirely by Google Gemini and real-time internet search.
        - **Web Integration**: Enter any query about current events, weather, stock prices, or general knowledge.
        - **Document RAG**: Upload PDFs or text files — AuraRAG will search them to answer your questions.
        - **Intelligent Routing**: Select 'Auto' search mode, and Gemini will automatically decide whether a web or document search is needed.
        - **Reasoning Monologues**: Toggle model to `gemini-2.0-flash-thinking-exp` to inspect native step-by-step thinking traces.
        """)
    with col2:
        st.markdown(f"""
        ### Active Configuration
        - **Engine**: `Google Gemini Server API`
        - **Model**: `{gemini_model}`
        - **Search Mode**: `{search_mode.upper()}`
        - **Session**: `{st.session_state.session_id}`
        - **Documents Loaded**: `{len(st.session_state.uploaded_docs)}`
        """)

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="message-container user-message">
            <strong>You</strong><br>
            {msg["text"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        if msg.get("reasoning_trace"):
            with st.expander("🧠 Thought Process & Reasoning Trace", expanded=False):
                if isinstance(msg["reasoning_trace"], list):
                    st.code("\n".join(msg["reasoning_trace"]), language="text")
                else:
                    st.code(msg["reasoning_trace"], language="text")

        st.markdown(f"""
        <div class="message-container assistant-message">
            <strong>AuraRAG</strong><br>
            {msg["text"]}
        </div>
        """, unsafe_allow_html=True)

        if msg.get("sources"):
            with st.expander(f"Citations & Context Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    # Handle both old format (metadata dict) and new SourceItem format
                    if isinstance(s, dict) and "metadata" in s:
                        meta     = s.get("metadata", {})
                        title    = meta.get("title") or "Web Page"
                        src_url  = meta.get("source")
                        snippet  = meta.get("text_preview") or ""
                        src_type = s.get("type", "web")
                    else:
                        title    = s.get("title", "Source")
                        src_url  = s.get("url")
                        snippet  = s.get("preview", "")
                        src_type = s.get("type", "web")

                    badge    = "badge-web" if src_type == "web" else "badge-doc"
                    icon     = "🌐" if src_type == "web" else "📄"
                    link_html = (
                        f'<a href="{src_url}" target="_blank" '
                        f'style="font-size:0.75rem; color:#00e5ff; word-break:break-all;">{src_url}</a><br>'
                        if src_url else ""
                    )
                    st.markdown(f"""
                    <div class="source-card">
                        <span class="{badge}">{icon} {src_type.upper()}</span><strong>{title}</strong><br>
                        {link_html}
                        <span style="font-size:0.8rem; font-style:italic; color:#9ca3af;">"{snippet}"</span>
                    </div>
                    """, unsafe_allow_html=True)


# ----------------- CHAT INPUT -----------------
query = st.chat_input("Type a message or ask a question...")

if query:
    st.markdown(f"""
    <div class="message-container user-message">
        <strong>You</strong><br>
        {query}
    </div>
    """, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "text": query})

    answer = ""
    sources = []
    reasoning_trace = []

    with st.status("Performing search and synthesizing response...", expanded=True) as status_box:
        if st.session_state.uploaded_docs:
            status_box.write("📄 Searching your documents...")
        status_box.write("🛰️ Querying backend RAG server...")

        payload = {
            "query": query,
            "session_id": st.session_state.session_id,
            "search_mode": search_mode,
            "model": gemini_model,
        }
        try:
            res = requests.post(f"{FASTAPI_URL}/rag/chat", json=payload, timeout=60)
            if res.status_code == 200:
                data = res.json()
                answer         = data.get("answer", "")
                sources        = data.get("sources", [])
                reasoning_trace = data.get("reasoning_trace", [])
                status_box.update(label="Response synthesized successfully!", state="complete", expanded=False)
            else:
                answer = f"Error from backend: {res.json().get('detail', 'Server error')}"
                status_box.update(label="Error compiling response.", state="error", expanded=False)
        except Exception as e:
            answer = f"Could not connect to backend. Ensure uvicorn is running on port 8000. Error: {str(e)}"
            status_box.update(label="Connection timed out.", state="error", expanded=False)

    # Reasoning trace
    if reasoning_trace:
        with st.expander("🧠 Thought Process & Reasoning Trace", expanded=False):
            trace = reasoning_trace if isinstance(reasoning_trace, str) else "\n".join(reasoning_trace)
            st.code(trace, language="text")

    # Answer
    st.markdown(f"""
    <div class="message-container assistant-message">
        <strong>AuraRAG</strong><br>
        {answer}
    </div>
    """, unsafe_allow_html=True)

    # Sources
    if sources:
        with st.expander(f"Citations & Context Sources ({len(sources)})"):
            for s in sources:
                if isinstance(s, dict) and "metadata" in s:
                    meta     = s.get("metadata", {})
                    title    = meta.get("title") or "Web Page"
                    src_url  = meta.get("source")
                    snippet  = meta.get("text_preview") or ""
                    src_type = s.get("type", "web")
                else:
                    title    = s.get("title", "Source")
                    src_url  = s.get("url")
                    snippet  = s.get("preview", "")
                    src_type = s.get("type", "web")

                badge    = "badge-web" if src_type == "web" else "badge-doc"
                icon     = "🌐" if src_type == "web" else "📄"
                link_html = (
                    f'<a href="{src_url}" target="_blank" '
                    f'style="font-size:0.75rem; color:#00e5ff; word-break:break-all;">{src_url}</a><br>'
                    if src_url else ""
                )
                st.markdown(f"""
                <div class="source-card">
                    <span class="{badge}">{icon} {src_type.upper()}</span><strong>{title}</strong><br>
                    {link_html}
                    <span style="font-size:0.8rem; font-style:italic; color:#9ca3af;">"{snippet}"</span>
                </div>
                """, unsafe_allow_html=True)

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "text": answer,
        "sources": sources,
        "reasoning_trace": reasoning_trace,
    })