import os
import requests
import streamlit as st
import time
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Review Detail — Code Reviewer",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Code Reviewer")
    st.caption("Advanced GitHub Code Reviewer")
    st.divider()
    # Use st.page_link for native multipage navigation
    st.page_link("dashboard.py", label="📋 Back to Dashboard", icon="📋")
    st.divider()
    
    try:
        api_check = requests.get(f"{API_BASE}/reviews/", timeout=2)
        if api_check.status_code == 200:
            st.success("API Status: Online")
        else:
            st.warning(f"API Status: {api_check.status_code}")
    except:
        st.error("API Status: Offline")

# ── Helper functions ──────────────────────────────────────────────────────────

def _api(method: str, path: str, **kwargs):
    """Unified API caller with error handling."""
    url = f"{API_BASE}{path}"
    try:
        resp = getattr(requests, method)(url, timeout=15, **kwargs)
        if resp.status_code == 404:
            return None, f"❌ 404 Not Found: `{url}`"
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return None, f"Communication Error: {e}"

def _status_badge(status: str) -> str:
    badges = {
        "running": "🔵 Running",
        "pending_hitl": "🟡 Awaiting Approval",
        "completed": "🟢 Completed",
        "rejected": "🔴 Rejected",
        "failed": "❌ Failed",
    }
    return badges.get(status, f"⚪ {status}")

# ── Review ID Management ──────────────────────────────────────────────────────
# Pull from query params (?review_id=123)
params = st.query_params
raw_id = params.get("review_id", "")

# Ensure we have a valid ID
if not raw_id:
    st.warning("### No Review Selected")
    st.info("Please select a review from the main Dashboard to view details.")
    if st.button("Go to Dashboard"):
        st.switch_page("dashboard.py")
    st.stop()

try:
    review_id = int(raw_id)
except ValueError:
    st.error(f"Invalid Review ID: {raw_id}")
    st.stop()

# ── Data Fetching ────────────────────────────────────────────────────────────────
with st.spinner(f"Loading Review #{review_id}..."):
    # Matches FastAPI: @router.get("/id/{review_id}")
    review, error = _api("get", f"/reviews/id/{review_id}")

if error:
    st.error(error)
    st.stop()

# ── Header Section ────────────────────────────────────────────────────────────
status_str = review.get("status", "unknown")
thread_id = review.get("thread_id")
# Note: Ensure the API returns these fields; fallback to generic labels if not
repo_name = review.get("repo", "Repository") 
pr_num = review.get("pr_number", "PR")

st.title(f"🔎 {repo_name} #{pr_num}")
st.caption(f"Review ID: `{review_id}` | Created: {review.get('created_at')}")

# Action Bar
top_c1, top_c2, top_c3 = st.columns([2, 2, 1.5])
top_c1.markdown(f"**Status:** {_status_badge(status_str)}")
top_c2.markdown(f"**Verdict:** `{review.get('verdict') or 'PENDING'}`")

with top_c3:
    if thread_id:
        # Passes the thread_id to the chat page query params
        st.link_button(
            "💬 Open PR Chat", 
            url=f"/review_chat?thread_id={thread_id}", 
            use_container_width=True,
            type="secondary"
        )
    else:
        st.button("💬 Chat Unavailable", disabled=True, use_container_width=True)

st.divider()

# ── HITL Decision Panel ───────────────────────────────────────────────────────
if status_str == "pending_hitl":
    with st.container(border=True):
        st.subheader("🟡 Action Required: Human-in-the-Loop")
        st.write("The AI has finished its analysis. Review the findings below and provide a decision.")
        
        c_app, c_rej = st.columns(2)
        with c_app:
            if st.button("✅ Approve & Post to GitHub", type="primary", use_container_width=True):
                with st.spinner("Submitting approval..."):
                    _, err = _api("post", f"/reviews/id/{review_id}/decision", json={"decision": "approved"})
                    if not err: 
                        st.success("Review Approved! Syncing with GitHub...")
                        time.sleep(1.5)
                        st.rerun()
                    else: st.error(err)
        with c_rej:
            if st.button("❌ Reject (Internal Only)", use_container_width=True):
                with st.spinner("Rejecting..."):
                    _, err = _api("post", f"/reviews/id/{review_id}/decision", json={"decision": "rejected"})
                    if not err: 
                        st.warning("Review Rejected.")
                        time.sleep(1.5)
                        st.rerun()
                    else: st.error(err)

# ── Content Tabs ──────────────────────────────────────────────────────────────
tab_findings, tab_diff, tab_steps = st.tabs(["🔴 AI Findings", "📄 Diff", "📑 Audit Trail"])

with tab_findings:
    st.markdown(f"### Summary")
    st.write(review.get("summary") or "No summary available yet.")
    
    col_iss, col_sug = st.columns(2)
    with col_iss:
        st.markdown("### ⚠️ Critical Issues")
        issues = review.get("issues") or []
        if not issues: 
            st.success("No critical issues found.")
        else:
            for issue in issues: 
                st.error(issue)
        
    with col_sug:
        st.markdown("### 💡 Suggestions")
        suggestions = review.get("suggestions") or []
        if not suggestions: 
            st.caption("No logic suggestions provided.")
        else:
            for sug in suggestions: 
                st.info(sug)

with tab_diff:
    diff_content = review.get("diff", "No diff available.")
    st.code(diff_content, language="diff")

with tab_steps:
    st.write("### Workflow Execution Steps")
    steps = review.get("steps", [])
    if not steps:
        st.info("No execution steps recorded.")
    else:
        for step in steps:
            status = step.get('status', 'unknown')
            step_icon = "✅" if status == "completed" else "⏳"
            with st.expander(f"{step_icon} {step.get('step_name')} — {status.upper()}"):
                st.write("**Model/Tool Output:**")
                st.json(step.get("output_data", {}))

st.divider()
st.caption("Powered by Gemini Agentic Platform · P3 Release")