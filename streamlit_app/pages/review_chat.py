"""
streamlit_app/pages/review_chat.py

Per-PR Chat UI — P4 Stub / P5 Foundation
------------------------------------------
P4: Display and send messages using the real /chat/{thread_id}/messages endpoints.
P5: Add Gemini streaming responses and memory-aware conversation.

Usage:
    streamlit run streamlit_app/dashboard.py
    Navigate to this page and pass ?thread_id=<uuid> in URL.

The thread_id is the Review.thread_id UUID — visible in GET /reviews/id/{id}
or GET /reviews/{id}/status response.
"""

import streamlit as st
import requests
from typing import List, Dict

API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Review Chat", layout="wide")


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_thread_id() -> str:
    """Get thread_id from URL query params."""
    params = st.query_params
    thread_id = params.get("thread_id", None)

    if not thread_id:
        st.error(
            "Missing thread_id in URL.\n\n"
            "Example: ?thread_id=d6e5c473-beeb-44fd-b314-ebef4d3de874\n\n"
            "Find the thread_id in the review details response."
        )
        st.stop()

    return thread_id


def fetch_messages(thread_id: str) -> List[Dict]:
    """Fetch chat history from GET /chat/{thread_id}/messages."""
    try:
        res = requests.get(
            f"{API_BASE_URL}/chat/{thread_id}/messages",
            timeout=10,
        )
        if res.status_code == 200:
            data = res.json()
            return data.get("messages", [])
        elif res.status_code == 404:
            return []  # thread has no messages yet
        return []
    except Exception as e:
        st.warning(f"Could not load message history: {e}")
        return []


def send_message(thread_id: str, content: str, role: str = "user") -> Dict:
    """Send message via POST /chat/{thread_id}/messages."""
    res = requests.post(
        f"{API_BASE_URL}/chat/{thread_id}/messages",
        json={"content": content, "role": role},
        timeout=30,
    )

    if res.status_code not in (200, 201):
        raise Exception(
            f"Failed to send message: {res.status_code} {res.text}"
        )

    return res.json()


# ── Session State ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_loaded" not in st.session_state:
    st.session_state.thread_loaded = False


# ── Main UI ───────────────────────────────────────────────────────────────────

thread_id = get_thread_id()

st.title(f"💬 Review Chat")
st.caption(f"Thread: `{thread_id}`")

# ── Load history once ─────────────────────────────────────────────────────────

if not st.session_state.thread_loaded:
    history = fetch_messages(thread_id)
    st.session_state.messages = history
    st.session_state.thread_loaded = True

# ── Display messages ──────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    role    = msg.get("role", "assistant")
    content = msg.get("content", "")
    with st.chat_message(role):
        st.markdown(content)

# ── Note about P5 ─────────────────────────────────────────────────────────────

st.info(
    "**P4 stub:** Messages are stored but AI responses are not yet generated. "
    "P5 will add Gemini-powered responses with review context awareness."
)

# ── Input ─────────────────────────────────────────────────────────────────────

user_input = st.chat_input("Ask about this PR review...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Store user message via API
    with st.chat_message("assistant"):
        placeholder = st.empty()

        try:
            send_message(thread_id, user_input, role="user")

            # P4 stub response — P5 will replace this with Gemini streaming
            stub_response = (
                "✅ Message received and stored. "
                "AI responses will be available in P5 once "
                "the Gemini chat integration is complete."
            )

            placeholder.markdown(stub_response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": stub_response,
            })

        except Exception as e:
            error_msg = f"❌ Error: {e}"
            placeholder.markdown(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })