import streamlit as st
import os
import pymupdf4llm
import tempfile
import json
import requests
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import MarkdownTextSplitter
from openai import OpenAI
import chromadb

# --- Helper Functions ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def fetch_openrouter_models():
    try:
        response = requests.get("https://openrouter.ai/api/v1/models")
        if response.status_code == 200:
            data = response.json()["data"]
            
            free_models = []
            paid_models = []
            
            for model in data:
                # Check pricing
                pricing = model.get("pricing", {})
                prompt_price = float(pricing.get("prompt", 0))
                completion_price = float(pricing.get("completion", 0))
                
                model_id = model["id"]
                model_name = model.get("name", model_id)
                
                if prompt_price == 0 and completion_price == 0:
                    free_models.append(model_id)
                else:
                    paid_models.append(model_id)
            
            # Sort for better UX
            free_models.sort()
            paid_models.sort()
            
            return {"Free": free_models, "Paid": paid_models}
    except Exception as e:
        st.error(f"Failed to fetch OpenRouter models: {e}")
        return None
    return None

# Configuration
CONFIG_FILE = "config.json"
CHROMA_PATH = "./data/chroma_db"

st.set_page_config(page_title="NotebookLM-Lite", layout="wide", page_icon="📓")

# --- Custom CSS for NotebookLM Aesthetic ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Google Sans', sans-serif;
        color: #e8eaed;
    }
    
    /* Main Background */
    .stApp {
        background-color: #131314;
    }
    
    /* Sidebar "Sources" Panel */
    [data-testid="stSidebar"] {
        background-color: #1e1f20;
        border-right: 1px solid #3c4043;
    }
    
    /* Card Style for Sources */
    .source-card {
        background-color: #2d2e31;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid #3c4043;
        color: #e8eaed;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    /* Chat Message Bubbles */
    .stChatMessage {
        background-color: transparent;
    }
    
    [data-testid="stChatMessageContent"] {
        background-color: #1e1f20;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #3c4043;
        color: #e8eaed;
    }

    /* Buttons */
    .stButton button {
        border-radius: 20px;
        font-weight: 500;
        background-color: #8ab4f8; /* Google Blue Light */
        color: #202124;
        border: none;
    }
    .stButton button:hover {
        background-color: #aecbfa;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e8eaed;
        font-weight: 400;
    }
    
    /* Input Fields */
    .stTextInput input {
        border-radius: 8px;
        background-color: #2d2e31;
        color: #e8eaed;
        border: 1px solid #3c4043;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e1f20;
        color: #e8eaed;
    }
</style>
""", unsafe_allow_html=True)

# --- Persistence ---
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)

# --- Model & Logic ---
@st.cache_resource
def get_embedding_function():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_vector_db():
    # Helper to get the persistent DB object
    embedding_func = get_embedding_function()
    # Ensure directory exists
    os.makedirs(CHROMA_PATH, exist_ok=True)
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_func)

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        md_text = pymupdf4llm.to_markdown(tmp_path)
        return md_text
    finally:
        os.remove(tmp_path)

def add_document_to_db(uploaded_file, md_text):
    text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([md_text])
    
    # Add metadata to identify source if needed later
    for doc in docs:
        doc.metadata["source"] = uploaded_file.name
        
    db = get_vector_db()
    db.add_documents(docs)
    return db

def clear_database():
    if os.path.exists(CHROMA_PATH):
        # Determine how to clear. Chroma's LangChain wrapper doesn't have a simple 'clear'.
        # We can delete the directory or use the underlying client.
        # Simplest consistency hack:
        db = get_vector_db()
        # Delete the collection
        db.delete_collection()
        # Re-initialize
        get_vector_db.clear() # Clear streamlit cache
        st.toast("Database cleared!")

# --- Main App ---
def main():
    config = load_config()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sources" not in st.session_state:
        st.session_state.sources = [] # List of filenames

    # --- Sidebar: Sources & Settings ---
    with st.sidebar:
        st.title("📓 Notebook")
        
        # New Chat Button
        if st.button("+ New Chat", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
        st.divider()
        
        # Source Management
        st.subheader("Sources")
        uploaded_file = st.file_uploader("Add Source (PDF)", type="pdf", label_visibility="collapsed")
        
        if uploaded_file:
            # Check if already added
            if uploaded_file.name not in st.session_state.sources:
                if st.button(f"Process {uploaded_file.name}", type="primary"):
                    with st.spinner("Reading paper..."):
                        md_text = process_pdf(uploaded_file)
                        add_document_to_db(uploaded_file, md_text)
                        
                        st.session_state.sources.append(uploaded_file.name)
                        st.toast(f"Added {uploaded_file.name} to knowledge base!", icon="✅")
                        st.rerun()
            else:
                st.info(f"{uploaded_file.name} is already processed.")

        # List active sources
        if st.session_state.sources:
            st.caption("Active Sources:")
            for source in st.session_state.sources:
                st.markdown(f"""
                <div class="source-card">
                    <span>📄 {source}</span>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("Clear All Sources", type="secondary"):
                clear_database()
                st.session_state.sources = []
                st.rerun()
        else:
             st.caption("No sources loaded.")

        st.divider()
        
        # Settings
        with st.expander("⚙️ Settings", expanded=False):
            st.caption("Provider: OpenRouter")
            
            api_key = config.get("api_key", "")
            new_key = st.text_input("OpenRouter Key", value=api_key, type="password")
            if new_key != api_key:
                config["api_key"] = new_key
                save_config(config)
                st.toast("API Key saved!")
                api_key = new_key
            st.caption("Key is saved locally.")
            
            # Dynamic Model Selection
            models_data = fetch_openrouter_models()
            model_name = "google/gemini-2.0-flash-exp:free" # Default fallback
            
            if models_data:
                category = st.radio("Model Tier", ["Free", "Paid"], horizontal=True)
                available_models = models_data.get(category, [])
                
                if available_models:
                        # Attempt to restore last used model
                    last_model = config.get("last_model")
                    idx = 0
                    if last_model in available_models:
                        idx = available_models.index(last_model)
                        
                    model_name = st.selectbox("Model", available_models, index=idx)
                    
                    if model_name != config.get("last_model"):
                        config["last_model"] = model_name
                        save_config(config)
                else:
                    st.warning("No models found in this category.")
                    model_name = st.text_input("Model Name", value=model_name)
            else:
                st.error("Could not fetch models from OpenRouter.")
                model_name = st.text_input("Model Name", value=model_name)

            st.divider()

            if st.button("Test API Key"):
                with st.spinner(f"Verifying key with {model_name}..."):
                    try:
                        # Test request
                        test_client = OpenAI(
                            base_url="https://openrouter.ai/api/v1",
                            api_key=api_key.strip(),
                            default_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": "NotebookLM-Lite"}
                        )
                        # Simple generation to check auth
                        test_resp = test_client.chat.completions.create(
                            model=model_name, # Use the selected model!
                            messages=[{"role": "user", "content": "Hi"}],
                            max_tokens=1
                        )
                        st.success(f"✅ Key is valid! Connected to OpenRouter using {model_name}.")
                    except Exception as e:
                        if "429" in str(e):
                            st.warning(f"⚠️ Key seems valid, but **{model_name}** is rate-limited (busy). Try selecting a different model!")
                        else:
                            st.error(f"❌ Key verification failed: {e}")

    # --- Main Chat Area ---
    st.markdown("### Chat")
    
    # Welcome Message
    if not st.session_state.messages:
        st.markdown("""
         <div style="text-align:center; padding: 40px; color: #666;">
            <h2>Welcome to your Notebook</h2>
            <p>Upload PDF sources to the sidebar to get started.</p>
         </div>
         """, unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            db = get_vector_db()
            # Simple check if DB is empty-ish (optional, but good UX)
            # For now, we just try to retrieve.
            
            # Retrieve
            try:
                retriever = db.as_retriever(search_kwargs={"k": 5}) # Increased k for multi-paper
                context_docs = retriever.invoke(prompt)
                
                if not context_docs:
                    st.warning("No relevant context found in loaded documents.")
                    context_text = "No context found."
                else:
                    # Format context with source names
                    context_text = ""
                    for cw in context_docs:
                        src = cw.metadata.get("source", "Unknown")
                        context_text += f"Source: {src}\nContent: {cw.page_content}\n\n"
            except Exception:
                # Likely empty DB
                context_text = "No documents loaded."
            
            # Prompt
            template = """<|im_start|>system
You are a smart research assistant called NotebookLM-Lite. 
Answer based strictly on the context provided. If unsure, say "I don't find that in the source".
Context:
{context}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
            full_prompt = template.format(context=context_text, question=prompt)

            response_placeholder = st.empty()
            response_text = ""

            # Generation
            try:
                if not api_key:
                    response_text = "Please configure your OpenRouter API Key in settings."
                    response_placeholder.error(response_text)
                else:
                    client = OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=api_key.strip(),
                        default_headers={
                            "HTTP-Referer": "http://localhost:8501", # Required by OpenRouter
                            "X-Title": "NotebookLM-Lite", # Required by OpenRouter
                        }
                    )
                    stream = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": f"Answer using this context:\n{context_text}"},
                            {"role": "user", "content": prompt}
                        ],
                        stream=True
                    )
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            response_text += chunk.choices[0].delta.content
                            response_placeholder.markdown(response_text + "▌")
            except Exception as e:
                response_text = f"Error: {e}"
                response_placeholder.error(response_text)
            
            response_placeholder.markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()
