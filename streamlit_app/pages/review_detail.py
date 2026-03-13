"""
streamlit_app/pages/review_detail.py

P3 — Full Review Detail page.

Shows for a selected review:
    - PR metadata (title, author, repo, branch)
    - Unified diff (syntax highlighted)
    - AI findings: issues and suggestions
    - Sandbox output: lint + validation results
    - HITL approve / reject buttons (only if status == pending_hitl)
    - Final verdict and GitHub comment summary

Navigation:
    Accessible from the main dashboard via "View Detail" link.
    Can also be opened directly with ?review_id=<id> in the URL.
"""

import os
import requests
import streamlit as st
import time

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Review Detail — Code Reviewer",
    page_icon="🔎",
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


# ── Helper functions ──────────────────────────────────────────────────────────

def _api(method: str, path: str, **kwargs):
    """Make a request to the FastAPI backend. Returns (data, error_str)."""
    url = f"{API_BASE}{path}"
    try:
        resp = getattr(requests, method)(url, timeout=15, **kwargs)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"❌ Cannot connect to API at `{API_BASE}`."
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, f"API error {e.response.status_code}: {detail}"
    except Exception as e:
        return None, f"Unexpected error: {e}"


def _status_badge(status: str) -> str:
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


def _render_diff(diff_text: str):
    """Render a unified diff with basic colour hints using st.code."""
    if not diff_text:
        st.info("No diff available.")
        return
    # Show raw diff in a code block (Streamlit doesn't natively syntax-highlight diffs)
    st.code(diff_text, language="diff")


def _render_sandbox_result(result: dict | None, label: str):
    """Render a SandboxResult dict."""
    if not result:
        st.info(f"No {label} result available.")
        return

    passed = result.get("passed", False)
    icon = "✅" if passed else "❌"
    duration = result.get("duration_ms")
    exit_code = result.get("exit_code")

    col1, col2, col3 = st.columns(3)
    col1.metric(f"{label} Result", f"{icon} {'Passed' if passed else 'Failed'}")
    if duration is not None:
        col2.metric("Duration", f"{duration:.0f} ms")
    if exit_code is not None:
        col3.metric("Exit Code", exit_code)

    output = result.get("output", "")
    error = result.get("error", "")

    if output:
        with st.expander("Output", expanded=not passed):
            st.code(output, language="bash")
    if error:
        with st.expander("Error", expanded=True):
            st.code(error, language="bash")


# ── Review selector ───────────────────────────────────────────────────────────
st.title("🔎 Review Detail")

# Allow review_id from URL param or manual input
params = st.query_params
default_id = params.get("review_id", "")

review_id_input = st.text_input(
    "Enter Review ID",
    value=str(default_id),
    placeholder="e.g. 42",
)

if not review_id_input.strip():
    st.info("Enter a Review ID above or navigate here from the PR Dashboard.")
    st.stop()

try:
    review_id = int(review_id_input.strip())
except ValueError:
    st.error("Review ID must be an integer.")
    st.stop()

# ── Fetch review data ─────────────────────────────────────────────────────────
with st.spinner("Loading review..."):
    data, error = _api("get", f"/reviews/{review_id}")

if error:
    st.error(error)
    st.stop()

review = data

# ── Header ────────────────────────────────────────────────────────────────────
status_str = review.get("status", "unknown")
verdict_str = review.get("verdict")
pr_num      = review.get("pr_number", "?")
pr_title    = review.get("pr_title", f"PR #{pr_num}")
repo        = review.get("repo_name", "—")
pr_author   = review.get("pr_author", "—")
pr_branch   = review.get("branch", "—")
created_at  = review.get("created_at", "—")

st.markdown(f"## PR #{pr_num} — {pr_title}")

col1, col2, col3, col4 = st.columns(4)
col1.markdown(f"**Repo:** `{repo}`")
col2.markdown(f"**Author:** @{pr_author}")
col3.markdown(f"**Branch:** `{pr_branch}`")
col4.markdown(f"**Created:** {created_at}")

col5, col6 = st.columns(2)
col5.markdown(f"**Status:** {_status_badge(status_str)}")
col6.markdown(f"**Verdict:** {_verdict_badge(verdict_str)}")

st.divider()

# ── HITL decision panel (shown prominently at top when pending) ───────────────
if status_str == "pending_hitl":
    st.subheader("🟡 Awaiting Your Decision")
    st.info(
        "The AI has completed its analysis. Review the findings below, "
        "then approve to post the review to GitHub, or reject to discard it."
    )

    col_approve, col_reject = st.columns(2)

    with col_approve:
        if st.button(
            "✅ Approve & Post to GitHub",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Approving review and posting to GitHub..."):
                resp, err = _api("post", f"/reviews/{review_id}/approve")
            if err:
                st.error(err)
            else:
                st.success(
                    f"✅ Approved! Verdict: **{resp.get('verdict', '—')}**. "
                    "GitHub comment posted."
                )
                time.sleep(1.5)
                st.rerun()

    with col_reject:
        if st.button(
            "❌ Reject (Do Not Post)",
            use_container_width=True,
        ):
            with st.spinner("Rejecting review..."):
                resp, err = _api("post", f"/reviews/{review_id}/reject")
            if err:
                st.error(err)
            else:
                st.warning("🚫 Review rejected. No comment has been posted to GitHub.")
                time.sleep(1.5)
                st.rerun()

    st.divider()

# ── Tabs: Diff | Findings | Sandbox | Summary ─────────────────────────────────
tab_diff, tab_findings, tab_sandbox, tab_summary = st.tabs(
    ["📄 Diff", "🔴 Findings", "🐳 Sandbox", "📝 Summary"]
)

# ── Tab: Diff ─────────────────────────────────────────────────────────────────
with tab_diff:
    st.subheader("Unified Diff")
    diff_text = review.get("diff", "")
    files = review.get("files", [])

    if files:
        st.markdown(f"**Changed files ({len(files)}):**")
        for f in files:
            st.markdown(f"- `{f}`")
        st.divider()

    _render_diff(diff_text)

# ── Tab: Findings ─────────────────────────────────────────────────────────────
with tab_findings:
    issues      = review.get("issues", [])
    suggestions = review.get("suggestions", [])

    col_i, col_s = st.columns(2)

    with col_i:
        st.subheader(f"🔴 Issues ({len(issues)})")
        if issues:
            for i, issue in enumerate(issues, 1):
                st.markdown(f"{i}. {issue}")
        else:
            st.success("No issues found.")

    with col_s:
        st.subheader(f"💡 Suggestions ({len(suggestions)})")
        if suggestions:
            for i, sug in enumerate(suggestions, 1):
                st.markdown(f"{i}. {sug}")
        else:
            st.info("No suggestions.")

    # Patch
    patch = review.get("patch", "")
    if patch:
        st.divider()
        st.subheader("🔧 Generated Patch")
        st.code(patch, language="diff")

# ── Tab: Sandbox ──────────────────────────────────────────────────────────────
with tab_sandbox:
    lint_result       = review.get("lint_result")
    validation_result = review.get("validation_result")

    st.subheader("Lint Result")
    _render_sandbox_result(lint_result, "Lint")

    st.divider()

    st.subheader("Patch Validation Result")
    _render_sandbox_result(validation_result, "Validation")

    st.markdown(
        "_Lint runs ruff check on the PR diff. "
        "Validation runs ruff + pytest on the generated patch — "
        "both execute inside an isolated Docker sandbox._"
    )

# ── Tab: Summary ──────────────────────────────────────────────────────────────
with tab_summary:
    summary = review.get("summary", "")
    if summary:
        st.subheader("Final Review Comment")
        st.markdown(summary)
        st.divider()
        st.markdown("_This is the markdown that was (or would be) posted as a GitHub PR comment._")
    else:
        if status_str in ("running", "pending_hitl"):
            st.info("Summary will appear here once the review is completed.")
        else:
            st.info("No summary available for this review.")

# ── Audit trail ───────────────────────────────────────────────────────────────
st.divider()
with st.expander("🗂 Audit Trail (ReviewSteps)", expanded=False):
    steps, err = _api("get", f"/reviews/{review_id}/steps")
    if err:
        st.warning(f"Could not load audit trail: {err}")
    elif steps:
        for step in steps:
            node  = step.get("node_name", "?")
            dur   = step.get("duration_ms", 0)
            out   = step.get("output_summary", "")
            s_at  = step.get("started_at", "")
            st.markdown(f"**{node}** — {dur:.0f} ms — `{s_at}`")
            if out:
                st.caption(out)
    else:
        st.info("No audit steps recorded.")