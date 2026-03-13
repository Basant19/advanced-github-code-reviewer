"""
streamlit_app/dashboard.py

P3 — Streamlit Admin Dashboard.

Pages:
    Main page (this file): PR list with status badges + HITL approve/reject.
    pages/review_detail.py: Full review — diff, findings, sandbox output, HITL.

Run with:
    streamlit run streamlit_app/dashboard.py

Environment:
    API_BASE_URL — FastAPI base URL (default: http://localhost:8000)
    Set in .env or as env var before running.
"""

import os
import time
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
POLL_INTERVAL = int(os.getenv("DASHBOARD_POLL_SECONDS", "10"))

st.set_page_config(
    page_title="Code Reviewer — Admin Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Code Reviewer")
    st.caption("Advanced GitHub Code Reviewer — Admin Dashboard")
    st.divider()
    st.markdown("**Pages**")
    st.page_link("dashboard.py", label="📋 PR Dashboard", icon="📋")
    st.page_link("pages/review_detail.py", label="🔎 Review Detail", icon="🔎")
    st.divider()
    st.markdown(f"**API:** `{API_BASE}`")
    auto_refresh = st.toggle("Auto-refresh (10s)", value=False)
    if st.button("🔄 Refresh now"):
        st.rerun()

# ── Helper functions ──────────────────────────────────────────────────────────

def _api(method: str, path: str, **kwargs):
    """Make a request to the FastAPI backend. Returns (data, error_str)."""
    url = f"{API_BASE}{path}"
    try:
        resp = getattr(requests, method)(url, timeout=10, **kwargs)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"❌ Cannot connect to API at `{API_BASE}`. Is FastAPI running?"
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, f"API error {e.response.status_code}: {detail}"
    except Exception as e:
        return None, f"Unexpected error: {e}"


def _status_badge(status: str) -> str:
    """Return a coloured emoji badge for a review status."""
    return {
        "running":      "🔵 Running",
        "pending_hitl": "🟡 Awaiting Approval",
        "completed":    "🟢 Completed",
        "rejected":     "🔴 Rejected",
        "failed":       "❌ Failed",
    }.get(status, f"⚪ {status}")


def _verdict_badge(verdict: str | None) -> str:
    if not verdict:
        return "—"
    return {
        "APPROVE":          "✅ Approve",
        "REQUEST_CHANGES":  "⚠️ Request Changes",
        "HUMAN_REJECTED":   "🚫 Human Rejected",
    }.get(verdict, verdict)


# ── Main page content ─────────────────────────────────────────────────────────
st.title("📋 Pull Request Review Dashboard")
st.caption("All recent PR reviews — approve or reject pending AI verdicts below.")

# ── Fetch reviews ─────────────────────────────────────────────────────────────
data, error = _api("get", "/reviews/")

if error:
    st.error(error)
    st.stop()

reviews = data if isinstance(data, list) else data.get("reviews", [])

if not reviews:
    st.info("No reviews yet. Open a PR on a connected repository to trigger one.")
    st.stop()

# ── Summary metrics ───────────────────────────────────────────────────────────
total       = len(reviews)
pending     = sum(1 for r in reviews if r.get("status") == "pending_hitl")
completed   = sum(1 for r in reviews if r.get("status") == "completed")
rejected    = sum(1 for r in reviews if r.get("status") == "rejected")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews", total)
col2.metric("🟡 Awaiting Approval", pending)
col3.metric("🟢 Completed", completed)
col4.metric("🔴 Rejected", rejected)

st.divider()

# ── Pending HITL section (shown prominently if any) ───────────────────────────
pending_reviews = [r for r in reviews if r.get("status") == "pending_hitl"]
if pending_reviews:
    st.subheader("🟡 Awaiting Your Approval")
    st.caption("These reviews are paused — the AI has finished analysis and is waiting for your decision.")

    for review in pending_reviews:
        rid = review["id"]
        pr_num = review.get("pr_number", "?")
        repo = review.get("repo_name", "unknown/repo")
        pr_title = review.get("pr_title", f"PR #{pr_num}")

        with st.container(border=True):
            col_info, col_actions = st.columns([3, 1])

            with col_info:
                st.markdown(f"### PR #{pr_num} — {pr_title}")
                st.markdown(f"**Repo:** `{repo}`")
                st.markdown(f"**Review ID:** {rid}")
                st.markdown(f"**Started:** {review.get('created_at', '—')}")

                # Quick summary of findings
                issues_count = review.get("issues_count", 0)
                st.markdown(
                    f"**AI found:** {issues_count} issue(s). "
                    f"[View full details →](/?review_id={rid})"
                )

            with col_actions:
                st.markdown("**Decision**")

                # Approve button
                if st.button(
                    "✅ Approve",
                    key=f"approve_{rid}",
                    type="primary",
                    use_container_width=True,
                ):
                    with st.spinner("Approving and resuming review..."):
                        resp, err = _api("post", f"/reviews/{rid}/approve")
                    if err:
                        st.error(err)
                    else:
                        st.success(
                            f"✅ Approved! Verdict: **{resp.get('verdict', '—')}**"
                        )
                        time.sleep(1)
                        st.rerun()

                # Reject button
                if st.button(
                    "❌ Reject",
                    key=f"reject_{rid}",
                    use_container_width=True,
                ):
                    with st.spinner("Rejecting review..."):
                        resp, err = _api("post", f"/reviews/{rid}/reject")
                    if err:
                        st.error(err)
                    else:
                        st.warning("🚫 Review rejected. No comment posted to GitHub.")
                        time.sleep(1)
                        st.rerun()

                # Detail link
                st.page_link(
                    "pages/review_detail.py",
                    label="🔎 View Detail",
                    use_container_width=True,
                )

    st.divider()

# ── Full review table ─────────────────────────────────────────────────────────
st.subheader("📄 All Reviews")

# Filter controls
col_f1, col_f2, _ = st.columns([1, 1, 2])
with col_f1:
    status_filter = st.selectbox(
        "Filter by status",
        ["All", "pending_hitl", "completed", "rejected", "running", "failed"],
        index=0,
    )
with col_f2:
    sort_order = st.selectbox("Sort by", ["Newest first", "Oldest first"])

filtered = reviews if status_filter == "All" else [
    r for r in reviews if r.get("status") == status_filter
]

if sort_order == "Oldest first":
    filtered = list(reversed(filtered))

if not filtered:
    st.info(f"No reviews with status '{status_filter}'.")
else:
    for review in filtered:
        rid = review["id"]
        pr_num = review.get("pr_number", "?")
        repo = review.get("repo_name", "—")
        pr_title = review.get("pr_title", f"PR #{pr_num}")
        status_str = review.get("status", "unknown")
        verdict_str = review.get("verdict")
        created = review.get("created_at", "—")

        with st.expander(
            f"[{_status_badge(status_str)}] PR #{pr_num} — {pr_title}  |  {repo}",
            expanded=False,
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"**Review ID:** {rid}")
            c2.markdown(f"**Status:** {_status_badge(status_str)}")
            c3.markdown(f"**Verdict:** {_verdict_badge(verdict_str)}")
            c4.markdown(f"**Created:** {created}")

            if status_str == "pending_hitl":
                col_a, col_r = st.columns(2)
                with col_a:
                    if st.button(f"✅ Approve", key=f"tbl_approve_{rid}", type="primary"):
                        with st.spinner("Approving..."):
                            resp, err = _api("post", f"/reviews/{rid}/approve")
                        if err:
                            st.error(err)
                        else:
                            st.success(f"Approved! Verdict: {resp.get('verdict')}")
                            time.sleep(1)
                            st.rerun()
                with col_r:
                    if st.button(f"❌ Reject", key=f"tbl_reject_{rid}"):
                        with st.spinner("Rejecting..."):
                            resp, err = _api("post", f"/reviews/{rid}/reject")
                        if err:
                            st.error(err)
                        else:
                            st.warning("Rejected.")
                            time.sleep(1)
                            st.rerun()

            st.page_link(
                "pages/review_detail.py",
                label=f"🔎 Open full review detail →",
            )

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(POLL_INTERVAL)
    st.rerun()