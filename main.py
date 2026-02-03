import streamlit as st
from app.chat_utils import get_chat_model, ask_chat_model
from app.pdf_utils import extract_text_from_pdf
from app.ui import pdf_uploader
from app.vectorstore_utils import create_faiss_index, retrieve_similar_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
import json
import uuid
from pathlib import Path

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="MediChatBot",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== STYLES =====================
st.markdown("""
<style>
.chat-message { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
.chat-message.user { background-color: #2b313e; color: white; }
.chat-message.assistant { background-color: #f0f2f6; color: black; }
.stButton > button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 0.5rem;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ===================== PERSISTENT CHAT MEMORY =====================
MEMORY_DIR = Path("chat_memory")
MEMORY_DIR.mkdir(exist_ok=True)

INDEX_FILE = MEMORY_DIR / "chat_index.json"

def load_chat_index():
    return json.loads(INDEX_FILE.read_text()) if INDEX_FILE.exists() else []

def save_chat_index(index):
    INDEX_FILE.write_text(json.dumps(index, indent=2))

def load_chat(chat_id):
    f = MEMORY_DIR / f"chat_{chat_id}.json"
    return json.loads(f.read_text()) if f.exists() else []

def save_chat(chat_id, messages):
    f = MEMORY_DIR / f"chat_{chat_id}.json"
    f.write_text(json.dumps(messages, indent=2))

def delete_chat(chat_id):
    f = MEMORY_DIR / f"chat_{chat_id}.json"
    if f.exists():
        f.unlink()
    index = [c for c in load_chat_index() if c["id"] != chat_id]
    save_chat_index(index)

def generate_chat_title(chat_model, messages):
    convo = "\n".join(
        [f"{m['role']}: {m['content']}" for m in messages[-8:]]
    )
    prompt = f"""
Create a short, clear title (5–7 words) summarizing this medical conversation.
Do not use quotes.

Conversation:
{convo}

Title:
"""
    return ask_chat_model(chat_model, prompt).strip()

# ===================== SESSION STATE =====================
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = load_chat(st.session_state.chat_id)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_model" not in st.session_state:
    st.session_state.chat_model = None

# ===================== HEADER =====================
st.markdown("""
<div style="text-align:center; padding:2rem 0;">
<h1 style="color:#ff4b4b;">🏥 MediChat Pro</h1>
<p>Your Intelligent Medical Document Assistant</p>
</div>
""", unsafe_allow_html=True)

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("### 🕘 Recent Chats")

    chat_index = load_chat_index()
    recent_chats = chat_index[-5:][::-1]

    for chat in recent_chats:
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(chat["title"], key=f"open_{chat['id']}"):
                st.session_state.chat_id = chat["id"]
                st.session_state.messages = load_chat(chat["id"])
                st.rerun()
        with col2:
            if st.button("🗑", key=f"del_{chat['id']}"):
                delete_chat(chat["id"])
                st.session_state.chat_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.rerun()

    st.divider()

    if st.button("➕ New Chat"):
        st.session_state.chat_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    if st.button("🧹 Clear Current Chat"):
        st.session_state.messages = []
        save_chat(st.session_state.chat_id, [])
        st.rerun()

    st.divider()

    st.markdown("### 📁 Upload Medical PDFs")
    uploaded_files = pdf_uploader()

    if uploaded_files and st.button("🛠️ Process Documents"):
        with st.spinner("Processing documents..."):

            # Extract text from uploaded PDFs
            texts = [extract_text_from_pdf(f) for f in uploaded_files]

            # Split text into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = []
            for t in texts:
                chunks.extend(splitter.split_text(t))

            # Merge with existing vectorstore if present
            if st.session_state.vectorstore:
                existing_texts = [d.page_content for d in st.session_state.vectorstore.docstore._dict.values()]
                chunks = existing_texts + chunks

            # Create or update vectorstore
            st.session_state.vectorstore = create_faiss_index(chunks)
            st.session_state.chat_model = get_chat_model(st.secrets["EURI_API_KEY"])

            st.success("✅ Documents processed successfully!")

# ===================== MAIN CHAT =====================
st.markdown("## 💬 Chat with your Medical Documents")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        st.caption(msg["timestamp"])

# ===================== CHAT INPUT =====================
if prompt := st.chat_input("Ask about your medical documents..."):
    timestamp = time.strftime("%H:%M")

    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp
    })
    save_chat(st.session_state.chat_id, st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(timestamp)

    if st.session_state.vectorstore and st.session_state.chat_model:
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents..."):
                docs = retrieve_similar_documents(
                    st.session_state.vectorstore, prompt, k=3
                )

                context = "\n\n".join([d.page_content for d in docs])

                history_text = "\n".join(
                    [f"{m['role'].upper()}: {m['content']}"
                     for m in st.session_state.messages[-12:]]
                )

                system_prompt = f"""
You are MediChat Pro, an intelligent medical document assistant.

Rules:
- Use ALL relevant conversation context.
- Answer strictly from the provided medical documents.
- Be medically accurate and clear.
- If information is missing, explicitly say so.
- Do not hallucinate.

Conversation History:
{history_text}

Medical Documents:
{context}

User Question:
{prompt}

Provide a detailed, helpful answer:
"""

                response = ask_chat_model(
                    st.session_state.chat_model, system_prompt
                )

            st.markdown(response)
            st.caption(timestamp)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": timestamp
        })
        save_chat(st.session_state.chat_id, st.session_state.messages)

        # ===== CHAT TITLE UPDATE =====
        index = load_chat_index()
        if not any(c["id"] == st.session_state.chat_id for c in index):
            title = generate_chat_title(
                st.session_state.chat_model,
                st.session_state.messages
            )
            index.append({
                "id": st.session_state.chat_id,
                "title": title
            })
            save_chat_index(index)

    else:
        st.warning("⚠️ Please upload and process documents first.")

# ===================== FOOTER =====================
st.markdown("""
<hr>
<div style="text-align:center; font-size:0.9rem; color:#666;">
🤖 Powered by Euri AI & LangChain | 🏥 Medical Document Intelligence
</div>
""", unsafe_allow_html=True)
