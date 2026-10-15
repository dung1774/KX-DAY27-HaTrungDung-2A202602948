"""Streamlit UI for the durable Human-in-the-Loop PR review agent.

Run with: ``uv run streamlit run app.py``.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_path, list_review_sessions
from common.github import parse_pr_url
from exercises.exercise_4_audit import build_graph


load_dotenv()


def _interrupt_from_snapshot(snapshot) -> Any | None:
    for task in snapshot.tasks:
        if task.interrupts:
            return task.interrupts[0]
    return None


async def run_graph(
    pr_url: str,
    thread_id: str,
    resume_value: Any = None,
    *,
    load_only: bool = False,
) -> dict[str, Any]:
    """Start, resume, or load one durable graph session."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as checkpointer:
        await checkpointer.setup()
        graph = build_graph(checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        if load_only:
            snapshot = await graph.aget_state(config)
            if not snapshot.values:
                raise ValueError(f"Không tìm thấy checkpoint cho thread {thread_id}")
            pending = _interrupt_from_snapshot(snapshot)
            if pending is not None:
                return {"__interrupt__": (pending,), "__values__": dict(snapshot.values)}
            return dict(snapshot.values)

        if resume_value is None:
            return await graph.ainvoke(
                {
                    "pr_url": pr_url,
                    "thread_id": thread_id,
                    "require_final_confirmation": True,
                },
                config,
            )
        return await graph.ainvoke(Command(resume=resume_value), config)


def _record_result(result: dict[str, Any]) -> None:
    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
        st.session_state.final = None
        values = result.get("__values__", {})
        if values.get("pr_url"):
            st.session_state.pr_url = values["pr_url"]
    else:
        st.session_state.interrupt_payload = None
        st.session_state.final = result


def _comment_rows(payload: dict[str, Any]) -> None:
    for comment in payload.get("comments", []):
        st.markdown(
            f"- **[{comment['severity']}]** "
            f"`{comment['file']}:{comment.get('line') or '?'}` — {comment['body']}"
        )


def render_approval_card(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Render the normal or post-escalation final approval card."""
    refined = payload["kind"] == "refined_approval"
    title = "Xác nhận refined review" if refined else "Yêu cầu phê duyệt"
    st.subheader(f"{title} — confidence {payload['confidence']:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])
    if payload.get("risk_factors"):
        st.warning("Risk factors: " + "; ".join(payload["risk_factors"]))
    _comment_rows(payload)

    with st.expander("Diff", expanded=False):
        st.code(payload.get("diff_preview", ""), language="diff")

    key_prefix = f"{st.session_state.thread_id}_{payload['kind']}"
    feedback = st.text_input("Feedback (không bắt buộc)", key=f"feedback_{key_prefix}")
    edit_key = f"edit_mode_{key_prefix}"
    col_approve, col_reject, col_edit = st.columns(3)
    if col_approve.button("Approve", type="primary", key=f"approve_{key_prefix}"):
        return {"choice": "approve", "feedback": feedback}
    if col_reject.button("Reject", key=f"reject_{key_prefix}"):
        return {"choice": "reject", "feedback": feedback}
    if col_edit.button("Edit", key=f"edit_{key_prefix}"):
        st.session_state[edit_key] = True

    if st.session_state.get(edit_key):
        edited_review = st.text_area(
            "Chỉnh sửa review Markdown trước khi đăng",
            value=payload.get("draft_review", ""),
            height=280,
            key=f"edited_review_{key_prefix}",
        )
        if st.button("Submit edited review", type="primary", key=f"submit_edit_{key_prefix}"):
            if not edited_review.strip():
                st.error("Nội dung review sau khi sửa không được để trống.")
            else:
                st.session_state[edit_key] = False
                return {
                    "choice": "edit",
                    "feedback": feedback,
                    "edited_review": edited_review,
                }
    return None


def render_escalation_card(payload: dict[str, Any]) -> dict[str, str] | None:
    """Render one answer box per concrete escalation question."""
    st.subheader(f"Strong escalation — confidence {payload['confidence']:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])
    if payload.get("risk_factors"):
        st.error("Risk factors: " + "; ".join(payload["risk_factors"]))
    with st.expander("Diff", expanded=False):
        st.code(payload.get("diff_preview", ""), language="diff")

    answers: dict[str, str] = {}
    with st.form(f"escalation_{st.session_state.thread_id}"):
        for index, question in enumerate(payload["questions"], start=1):
            answers[question] = st.text_area(
                f"{index}. {question}", key=f"answer_{st.session_state.thread_id}_{index}"
            )
        submitted_answers = st.form_submit_button("Gửi câu trả lời", type="primary")
    if submitted_answers:
        missing = [question for question, answer in answers.items() if not answer.strip()]
        if missing:
            st.error("Vui lòng trả lời đầy đủ tất cả câu hỏi trước khi tiếp tục.")
            return None
        return answers
    return None


def _load_recent_sessions() -> list[dict[str, Any]]:
    try:
        return asyncio.run(list_review_sessions())
    except Exception as exc:
        st.sidebar.warning(f"Không thể đọc recent sessions: {exc}")
        return []


def _reset_session_for_new_review(pr_url: str) -> None:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    st.session_state.error = None


st.set_page_config(page_title="HITL PR Review Agent", layout="wide")
st.title("HITL PR Review Agent")
st.caption("Day 27 – Track 03 · Confidence routing, human approval và SQLite audit trail")

for key, default in (
    ("thread_id", None),
    ("pr_url", ""),
    ("interrupt_payload", None),
    ("final", None),
    ("error", None),
):
    if key not in st.session_state:
        st.session_state[key] = default


with st.sidebar:
    st.header("Recent sessions")
    sessions = _load_recent_sessions()
    if sessions:
        labels = {
            f"{row['thread_id'][:8]} · {row['worst_risk']} · {row['pr_url']}": row
            for row in sessions
        }
        selected_label = st.selectbox("Chọn session", list(labels), label_visibility="collapsed")
        if st.button("Mở session", use_container_width=True):
            selected = labels[selected_label]
            st.session_state.thread_id = selected["thread_id"]
            st.session_state.pr_url = selected["pr_url"]
            st.session_state.error = None
            try:
                loaded = asyncio.run(run_graph(
                    selected["pr_url"], selected["thread_id"], load_only=True
                ))
                _record_result(loaded)
                st.rerun()
            except Exception as exc:
                st.session_state.error = f"Không thể mở session: {exc}"
    else:
        st.caption("Chưa có audit session.")


with st.form("start"):
    entered_pr_url = st.text_input(
        "GitHub PR URL",
        value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    start_submitted = st.form_submit_button("Run Review", type="primary")

if start_submitted:
    try:
        parse_pr_url(entered_pr_url)
        _reset_session_for_new_review(entered_pr_url.strip())
        with st.spinner("Đang tải PR và phân tích bằng AI..."):
            first_result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id
            ))
        _record_result(first_result)
    except Exception as exc:
        st.session_state.error = f"Không thể chạy review: {exc}"

if st.session_state.error:
    st.error(st.session_state.error)

payload = st.session_state.interrupt_payload
if payload is not None:
    if payload["kind"] in {"approval_request", "refined_approval"}:
        answer = render_approval_card(payload)
    elif payload["kind"] == "escalation":
        answer = render_escalation_card(payload)
    else:
        answer = None
        st.error(f"Loại interrupt không được hỗ trợ: {payload['kind']}")

    if answer is not None:
        try:
            with st.spinner("Đang tiếp tục workflow..."):
                resumed = asyncio.run(run_graph(
                    st.session_state.pr_url,
                    st.session_state.thread_id,
                    resume_value=answer,
                ))
            _record_result(resumed)
            st.session_state.error = None
            st.rerun()
        except Exception as exc:
            st.session_state.error = f"Không thể tiếp tục session: {exc}"
            st.error(st.session_state.error)

final = st.session_state.final
if final is not None:
    action = final.get("final_action", "unknown")
    analysis = final.get("analysis")
    if analysis is not None:
        st.metric("Confidence", f"{analysis.confidence:.0%}")
        st.markdown(f"**Summary:** {analysis.summary}")

    if action in {"auto_committed", "committed"}:
        st.success("GitHub đã xác nhận review comment được đăng thành công.")
        target = final.get("comment_url") or st.session_state.pr_url
        st.link_button("Mở comment / PR trên GitHub", target)
    elif action == "rejected":
        st.warning("Review đã bị từ chối — không có comment nào được đăng.")
        st.link_button("Mở PR trên GitHub", st.session_state.pr_url)
    elif action == "dry_run":
        st.info("Dry-run hoàn tất — comment không được đăng lên GitHub.")
        st.link_button("Mở PR trên GitHub", st.session_state.pr_url)
    elif action in {"commit_failed", "auto_commit_failed"}:
        st.error("GitHub không xác nhận đăng comment. Kiểm tra token/quyền và audit log.")
    else:
        st.info(f"final_action = {action}")

    st.caption(
        f"thread_id = {st.session_state.thread_id} · replay: "
        f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`"
    )
