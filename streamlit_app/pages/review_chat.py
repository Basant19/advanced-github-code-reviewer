import requests
import streamlit as st
import os
import time  # Added for better streaming control
from dotenv import load_dotenv

# ── Configuration ──
load_dotenv()
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Review Chat", layout="wide", page_icon="💬")

# ── Data Fetching Helpers ──

def get_thread_id() -> str:
    """
    Robustly extract thread_id from query params.
    Uses the new st.query_params API (Streamlit 1.30+).
    """
    # Fix: Ensure we handle the case where thread_id might be a list or missing
    params = st.query_params
    thread_id = params.get("thread_id")
    
    if not thread_id:
        st.error("### ❌ Missing `thread_id`")
        st.info(
            "This page requires a valid thread context. \n\n"
            "Please open this chat via the **'Open Chat'** button on the Review Detail page."
        )
        if st.button("⬅️ Back to Dashboard"):
            st.switch_page("dashboard.py")
        st.stop()
    return thread_id

def fetch_history(thread_id: str) -> list:
    """Load existing messages from GET /chat/{thread_id}/messages."""
    try:
        r = requests.get(f"{API_BASE}/chat/{thread_id}/messages", timeout=10)
        if r.status_code == 200:
            # Matches the ThreadMessagesResponse schema from chat.py
            return r.json().get("messages", [])
        return []
    except Exception as e:
        st.error(f"Error connecting to Chat API: {e}")
        return []

def fetch_review_context(thread_id: str) -> dict:
    """
    Finds the specific review linked to this thread.
    Required because chat needs metadata (PR#, Repo) for the header.
    """
    try:
        # We use the list endpoint which we verified returns thread_id
        r = requests.get(f"{API_BASE}/reviews/", timeout=10)
        if r.status_code == 200:
            reviews = r.json()
            # Search for the review matching our thread
            return next((rev for rev in reviews if rev.get("thread_id") == thread_id), {})
    except Exception:
        pass
    return {}

def send_chat_message(thread_id: str, content: str) -> dict:
    """POST /chat/{thread_id}/messages."""
    try:
        r = requests.post(
            f"{API_BASE}/chat/{thread_id}/messages",
            json={"content": content, "role": "user"},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Failed to send message: {e}")
        return {}

# ── Session Management ──

thread_id = get_thread_id()

if "messages" not in st.session_state or st.session_state.get("last_thread") != thread_id:
    st.session_state.messages = fetch_history(thread_id)
    st.session_state.last_thread = thread_id

# ── UI Layout ──

# Header Area
review_ctx = fetch_review_context(thread_id)
col1, col2 = st.columns([3, 1])

with col1:
    if review_ctx:
        repo = review_ctx.get('repo', 'Unknown Repo')
        pr = review_ctx.get('pr_number', '??')
        st.title(f"💬 Chat: {repo} #{pr}")
    else:
        st.title("💬 Review Chat")
    st.caption(f"Thread Session: `{thread_id}`")

with col2:
    if st.button("🔄 Refresh History", use_container_width=True):
        st.session_state.messages = fetch_history(thread_id)
        st.rerun()

st.divider()

# ── Chat Display ──

# Display history
for msg in st.session_state.messages:
    # Handle both database objects and dictionary formats
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    with st.chat_message(role):
        st.markdown(content)

# Chat Input
if prompt := st.chat_input("Ask a question about the code or requested changes..."):
    # 1. Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Assistant Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("*(Gemini is analyzing context...)*")
        
        result = send_chat_message(thread_id, prompt)
        
        if result:
            full_response = result.get("reply", "No response received.")
            
            # Streaming effect: avoid the "split()" flickering bug by 
            # building the string properly
            displayed_text = ""
            for char in full_response:
                displayed_text += char
                response_placeholder.markdown(displayed_text + "▌")
                time.sleep(0.005) # Subtle realistic pacing
            
            response_placeholder.markdown(full_response)
            
            # Update session state
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response
            })
        else:
            response_placeholder.error("I encountered an error processing that request.")

# ── Sidebar ──

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2111/2111432.png", width=50) # GitHub Icon
    st.header("Context")
    
    if review_ctx:
        st.success(f"**Verdict:** {review_ctx.get('verdict', 'PENDING')}")
        st.write(f"**Status:** {review_ctx.get('status', '—')}")
        st.write(f"**Created:** {review_ctx.get('created_at', '—')[:10]}")
    else:
        st.warning("Context not found in active reviews.")

    st.divider()
    
    if st.button("🗑️ Clear Thread", use_container_width=True, type="secondary"):
        try:
            requests.delete(f"{API_BASE}/chat/{thread_id}", timeout=5)
            st.session_state.messages = []
            st.success("History cleared locally and on server.")
            st.rerun()
        except Exception as e:
            st.error(f"Delete failed: {e}")
            
    if st.button("🏠 Exit to Dashboard", use_container_width=True):
        st.switch_page("dashboard.py")