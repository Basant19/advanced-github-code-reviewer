import requests
import streamlit as st
import os
import time
from dotenv import load_dotenv

# ── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Advanced GitHub Code Reviewer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data Fetching ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=5)  # Snappier refresh for HITL status tracking
def fetch_all_reviews():
    """Fetch the enriched list of reviews for the dashboard."""
    try:
        r = requests.get(f"{API_BASE}/reviews/", timeout=10)
        if r.status_code == 200:
            return r.json()
        return []
    except Exception as e:
        st.sidebar.error(f"Backend Connection Error: {e}")
        return []

all_reviews = fetch_all_reviews()

# ── Sidebar: Trigger & History ────────────────────────────────────────────────
with st.sidebar:
    st.title("🚀 Reviewer Home")
    
    # 1. Trigger Section
    with st.expander("⚡ New Review", expanded=False):
        owner = st.text_input("Owner", value="Basant19")
        repo  = st.text_input("Repo",  value="agent-eval-lab")
        pr    = st.number_input("PR Number", min_value=1, value=1, step=1)

        if st.button("Start Review", type="primary", use_container_width=True):
            with st.spinner("Initializing Graph..."):
                try:
                    r = requests.post(
                        f"{API_BASE}/reviews/trigger",
                        json={"owner": owner.strip(), "repo": repo.strip(), "pr_number": int(pr)},
                        timeout=120,
                    )
                    if r.status_code in (200, 202):
                        st.success("Review Triggered!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Trigger Failed: {r.status_code}")
                except Exception as e:
                    st.error(f"Request Error: {e}")

    st.divider()

    # 2. History Section - Quick access to chat sessions
    st.subheader("📜 Recent Chats")
    if not all_reviews:
        st.caption("No active threads found.")
    else:
        # Show last 10 threads with valid IDs
        threads_found = 0
        for rev in reversed(all_reviews):
            t_id = rev.get("thread_id")
            if t_id and threads_found < 10:
                label = f"💬 {rev.get('repo')} #{rev.get('pr_number')}"
                # URL matches review_chat.py query param logic
                chat_url = f"/review_chat?thread_id={t_id}"
                st.link_button(label, url=chat_url, use_container_width=True)
                threads_found += 1
            

    st.divider()
    
    # 3. Indexing Section (Placeholder for vector DB management)
    with st.expander("📦 Repository Indexing", expanded=False):
        st.caption("Sync repository to ChromaDB for semantic code search.")
        idx_owner = st.text_input("Owner", value="Basant19", key="idx_owner")
        idx_repo  = st.text_input("Repo",  value="agent-eval-lab", key="idx_repo")
        if st.button("Index Now", use_container_width=True):
            st.info("Indexing service is running in background...")

# ── Main Content: Metrics ─────────────────────────────────────────────────────
st.title("🤖 Advanced GitHub Code Reviewer")
st.caption("Agentic Platform | LangGraph Orchestration | Gemini 1.5 Pro")

if all_reviews:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Reviews", len(all_reviews))
    c2.metric("Awaiting HITL", sum(1 for r in all_reviews if r.get("status") == "pending_hitl"))
    c3.metric("Completed", sum(1 for r in all_reviews if r.get("status") == "completed"))

st.divider()

# ── Reviews Table ─────────────────────────────────────────────────────────────
st.header("📋 Active Review Pipeline")

if not all_reviews:
    st.info("No reviews found. Trigger a new one from the sidebar to begin.")
else:
    # Filter UI
    f_col1, f_col2 = st.columns([2, 1])
    with f_col1:
        search = st.text_input("🔍 Search", placeholder="Filter by repository name or PR...")
    with f_col2:
        status_filter = st.selectbox("Status", ["All Statuses", "pending_hitl", "completed", "failed"])

    # Filtering Logic
    filtered = [
        r for r in all_reviews 
        if (not search or search.lower() in f"{r.get('repo')} {r.get('pr_number')}".lower())
        and (status_filter == "All Statuses" or r.get("status") == status_filter)
    ]

    for review in reversed(filtered):
        review_id = review.get("id")
        status = review.get("status", "unknown")
        verdict = review.get("verdict") or "PENDING"
        thread_id = review.get("thread_id", "")
        
        # UI Styling based on status
        border_color = "orange" if status == "pending_hitl" else "gray"
        icon = "⏳" if status == "pending_hitl" else ("✅" if verdict == "APPROVE" else "🔴")
        
        with st.container(border=True):
            ca, cb, cc = st.columns([3, 2, 1.5])
            
            with ca:
                st.markdown(f"### {icon} {review.get('repo')} #{review.get('pr_number')}")
                st.caption(f"ID: `{review_id}` | Created: {review.get('created_at', '')[:16]}")
            
            with cb:
                st.write(f"**Current Phase:** `{status.replace('_', ' ').title()}`")
                st.write(f"**AI Verdict:** `{verdict}`")
            
            with cc:
                # IMPORTANT: Logic for Detail & Chat navigation
                st.link_button(
                    "🔎 Review Detail", 
                    url=f"/review_detail?review_id={review_id}", 
                    use_container_width=True,
                    type="secondary"
                )
                if thread_id:
                    st.link_button(
                        "💬 PR Chat", 
                        url=f"/review_chat?thread_id={thread_id}", 
                        use_container_width=True
                    )

            # ── QUICK HITL ACTIONS ──
            if status == "pending_hitl":
                st.divider()
                st.warning("Review paused. Human intervention required.")
                btn_col1, btn_col2, btn_spacer = st.columns([1, 1, 2])
                
                with btn_col1:
                    if st.button("✅ Approve", key=f"app_{review_id}", type="primary", use_container_width=True):
                        # FIX: Point to correct decision endpoint with JSON payload
                        r = requests.post(
                            f"{API_BASE}/reviews/id/{review_id}/decision", 
                            json={"decision": "approved"}
                        )
                        if r.status_code == 200:
                            st.toast(f"Review #{review_id} approved!", icon="✅")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Approval failed: {r.status_code}")

                with btn_col2:
                    if st.button("❌ Reject", key=f"rej_{review_id}", use_container_width=True):
                        # FIX: Point to correct decision endpoint with JSON payload
                        r = requests.post(
                            f"{API_BASE}/reviews/id/{review_id}/decision", 
                            json={"decision": "rejected"}
                        )
                        if r.status_code == 200:
                            st.toast(f"Review #{review_id} rejected.", icon="🚫")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Rejection failed: {r.status_code}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("GitHub Agentic Dashboard · Version 5.0 (Corrective RAG Enabled)")