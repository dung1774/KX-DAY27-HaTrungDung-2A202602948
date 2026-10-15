"""Exercise 4 — durable HITL PR review graph with a structured audit trail."""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid
from typing import Any

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.db import db_path, write_audit_event
from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.prompts import PR_REVIEW_SYSTEM_PROMPT
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    AuditEntry,
    PRAnalysis,
    ReviewState,
    risk_level_for,
)


console = Console()
AGENT_ID = "pr-review-agent@v0.1"


async def audit(
    state: ReviewState,
    entry: AuditEntry,
    *,
    event_key: str | None = None,
) -> None:
    """Persist one queryable audit event, optionally idempotently."""
    await write_audit_event(
        thread_id=state["thread_id"],
        pr_url=state["pr_url"],
        entry=entry,
        event_key=event_key,
    )


async def node_fetch_pr(state: ReviewState) -> dict[str, Any]:
    console.print("[cyan]-> fetch_pr[/cyan]")
    t0 = time.monotonic()
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]OK[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    await audit(
        state,
        AuditEntry(
            agent_id=AGENT_ID,
            action="fetch_pr",
            confidence=0.0,
            risk_level="med",
            decision="pending",
            reason=f"Fetched {len(pr.files_changed)} files, head={pr.head_sha[:7]}",
            execution_time_ms=int((time.monotonic() - t0) * 1000),
        ),
        event_key="fetch_pr",
    )
    return {
        "pr_title": pr.title,
        "pr_author": pr.author,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }


async def node_analyze(state: ReviewState) -> dict[str, PRAnalysis]:
    console.print("[cyan]-> analyze[/cyan]")
    t0 = time.monotonic()
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        analysis: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": PR_REVIEW_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"PR title: {state['pr_title']}\n\nUnified diff:\n{state['pr_diff']}",
            },
        ])
    console.print(
        f"  [green]OK[/green] confidence={analysis.confidence:.0%}, "
        f"{len(analysis.comments)} comment(s)"
    )
    await audit(
        state,
        AuditEntry(
            agent_id=AGENT_ID,
            action="analyze",
            confidence=analysis.confidence,
            risk_level=risk_level_for(analysis.confidence),
            decision="pending",
            reason=analysis.confidence_reasoning,
            execution_time_ms=int((time.monotonic() - t0) * 1000),
        ),
        event_key="analyze",
    )
    return {"analysis": analysis}


async def node_route(state: ReviewState) -> dict[str, str]:
    console.print("[cyan]-> route[/cyan]")
    t0 = time.monotonic()
    confidence = state["analysis"].confidence
    if confidence >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif confidence < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(
        f"  [green]OK[/green] decision=[bold]{decision}[/bold] "
        f"(confidence={confidence:.0%})"
    )
    await audit(
        state,
        AuditEntry(
            agent_id=AGENT_ID,
            action="route",
            confidence=confidence,
            risk_level=risk_level_for(confidence),
            decision=decision,
            reason=f"Thresholds: auto>={AUTO_APPROVE_THRESHOLD}, escalate<{ESCALATE_THRESHOLD}",
            execution_time_ms=int((time.monotonic() - t0) * 1000),
        ),
        event_key="route",
    )
    return {"decision": decision}


def _render_comment_body(state: ReviewState) -> str:
    if state.get("edited_review"):
        return state["edited_review"].strip()
    analysis = state["analysis"]
    lines = [
        f"### Automated review (confidence {analysis.confidence:.0%})",
        "",
        analysis.summary,
        "",
    ]
    for comment in analysis.comments:
        lines.append(
            f"- **[{comment.severity}]** `{comment.file}:{comment.line or '?'}` — {comment.body}"
        )
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    if state.get("escalation_answers"):
        lines.append("\n_Reviewer answered escalation questions:_")
        for question, answer in state["escalation_answers"].items():
            lines.append(f"> **{question}** {answer}")
    return "\n".join(lines)


async def node_human_approval(state: ReviewState) -> dict[str, Any]:
    t0 = time.monotonic()
    analysis = state["analysis"]
    await audit(
        state,
        AuditEntry(
            agent_id=AGENT_ID,
            action="human_approval",
            confidence=analysis.confidence,
            risk_level=risk_level_for(analysis.confidence),
            decision="pending",
            reason="Waiting for reviewer decision",
            execution_time_ms=int((time.monotonic() - t0) * 1000),
        ),
        event_key="human_approval:pending",
    )
    response = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "confidence": analysis.confidence,
        "confidence_reasoning": analysis.confidence_reasoning,
        "summary": analysis.summary,
        "comments": [comment.model_dump() for comment in analysis.comments],
        "diff_preview": state["pr_diff"][:4000],
        "draft_review": _render_comment_body(state),
    })
    choice = response.get("choice")
    if choice not in {"approve", "reject", "edit"}:
        raise ValueError(f"Invalid reviewer choice: {choice!r}")
    edited_review = response.get("edited_review") if choice == "edit" else None
    if choice == "edit" and not (edited_review or "").strip():
        raise ValueError("Edit requires a non-empty edited_review")
    feedback = response.get("feedback")
    await audit(
        state,
        AuditEntry(
            agent_id=AGENT_ID,
            action="human_approval",
            confidence=analysis.confidence,
            risk_level=risk_level_for(analysis.confidence),
            reviewer_id=os.environ.get("GITHUB_USER"),
            decision=choice,
            reason=feedback or ("Reviewer edited the draft" if choice == "edit" else None),
            execution_time_ms=int((time.monotonic() - t0) * 1000),
        ),
        event_key="human_approval:response",
    )
    return {
        "human_choice": choice,
        "human_feedback": feedback,
        "edited_review": edited_review,
    }


async def _post(state: ReviewState) -> dict[str, Any]:
    body = _render_comment_body(state)
    try:
        result = post_review_comment(
            state["pr_url"], body, idempotency_key=state["thread_id"]
        )
    except Exception as exc:
        console.print(f"  [red]ERROR[/red] post failed: {exc}")
        return {"final_action": "commit_failed", "human_feedback": str(exc)}
    if result.status == "dry_run":
        console.print("  [yellow]-[/yellow] dry-run: comment was not posted")
        return {"final_action": "dry_run", "posted_comment_body": body}
    console.print(f"  [green]OK[/green] {result.status} at {state['pr_url']}")
    return {
        "final_action": "committed",
        "posted_comment_body": body,
        "comment_url": result.url,
    }


async def node_commit(state: ReviewState) -> dict[str, Any]:
    console.print("[cyan]-> commit[/cyan]")
    t0 = time.monotonic()
    confidence = state["analysis"].confidence
    from_escalation = bool(state.get("escalation_answers"))
    if from_escalation:
        final_choice = state.get("final_confirmation")
        should_post = not state.get("require_final_confirmation") or final_choice in {"approve", "edit"}
        audit_decision = final_choice if state.get("require_final_confirmation") else "escalate"
    else:
        should_post = state.get("human_choice") in {"approve", "edit"}
        audit_decision = state.get("human_choice") or "reject"

    if should_post:
        outcome = await _post(state)
    else:
        console.print("  [yellow]-[/yellow] rejected - no comment posted")
        outcome = {"final_action": "rejected"}

    await audit(
        state,
        AuditEntry(
            agent_id=AGENT_ID,
            action="commit",
            confidence=confidence,
            risk_level=risk_level_for(confidence),
            reviewer_id=os.environ.get("GITHUB_USER") if not from_escalation or state.get("require_final_confirmation") else None,
            decision=str(audit_decision),
            reason=f"Final action: {outcome['final_action']}",
            execution_time_ms=int((time.monotonic() - t0) * 1000),
        ),
        event_key="commit",
    )
    return outcome


async def node_auto_approve(state: ReviewState) -> dict[str, Any]:
    console.print("[cyan]-> auto_approve[/cyan]  [dim]high confidence[/dim]")
    t0 = time.monotonic()
    confidence = state["analysis"].confidence
    outcome = await _post(state)
    final_action = outcome["final_action"]
    if final_action == "committed":
        outcome["final_action"] = "auto_committed"
    elif final_action == "commit_failed":
        outcome["final_action"] = "auto_commit_failed"
    await audit(
        state,
        AuditEntry(
            agent_id=AGENT_ID,
            action="auto_approve",
            confidence=confidence,
            risk_level=risk_level_for(confidence),
            decision="auto",
            reason=f"Final action: {outcome['final_action']}",
            execution_time_ms=int((time.monotonic() - t0) * 1000),
        ),
        event_key="auto_approve",
    )
    return outcome


def _fallback_questions(state: ReviewState) -> list[str]:
    analysis = state["analysis"]
    context = ", ".join(analysis.risk_factors[:2]) or ", ".join(state.get("pr_files", [])[:2])
    return [
        f"What control or requirement addresses this identified risk: {context}?",
        "Which tests or rollout checks verify the changed behavior in this diff?",
    ]


async def node_escalate(state: ReviewState) -> dict[str, dict[str, str]]:
    t0 = time.monotonic()
    analysis = state["analysis"]
    questions = (analysis.escalation_questions or _fallback_questions(state))[:4]
    await audit(
        state,
        AuditEntry(
            agent_id=AGENT_ID,
            action="escalate",
            confidence=analysis.confidence,
            risk_level=risk_level_for(analysis.confidence),
            decision="pending",
            reason=f"Waiting for answers to {len(questions)} escalation questions",
            execution_time_ms=int((time.monotonic() - t0) * 1000),
        ),
        event_key="escalate:pending",
    )
    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": analysis.confidence,
        "confidence_reasoning": analysis.confidence_reasoning,
        "summary": analysis.summary,
        "risk_factors": analysis.risk_factors,
        "questions": questions,
        "diff_preview": state["pr_diff"][:4000],
    })
    if not isinstance(answers, dict):
        raise ValueError("Escalation answers must be a question-to-answer mapping")
    await audit(
        state,
        AuditEntry(
            agent_id=AGENT_ID,
            action="escalate",
            confidence=analysis.confidence,
            risk_level=risk_level_for(analysis.confidence),
            reviewer_id=os.environ.get("GITHUB_USER"),
            decision="escalate",
            reason=f"Reviewer answered {len(answers)} escalation questions",
            execution_time_ms=int((time.monotonic() - t0) * 1000),
        ),
        event_key="escalate:response",
    )
    return {"escalation_answers": answers}


async def node_synthesize(state: ReviewState) -> dict[str, PRAnalysis]:
    console.print("[cyan]-> synthesize[/cyan]")
    t0 = time.monotonic()
    initial = state["analysis"]
    qa = "\n\n".join(
        f"Question: {question}\nReviewer answer: {answer}"
        for question, answer in (state.get("escalation_answers") or {}).items()
    )
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined: PRAnalysis = await llm.ainvoke([
            {
                "role": "system",
                "content": (
                    "Refine the complete PRAnalysis using the human answers and original diff. "
                    "Verify every finding against evidence. Do not assume confidence must rise; "
                    "choose the score justified by the remaining uncertainty."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original diff:\n{state['pr_diff']}\n\n"
                    f"Initial analysis:\n{initial.model_dump_json(indent=2)}\n\n"
                    f"Reviewer Q&A:\n{qa}"
                ),
            },
        ])
    console.print(f"  [green]OK[/green] refined confidence={refined.confidence:.0%}")
    await audit(
        state,
        AuditEntry(
            agent_id=AGENT_ID,
            action="synthesize",
            confidence=refined.confidence,
            risk_level=risk_level_for(refined.confidence),
            reviewer_id=os.environ.get("GITHUB_USER"),
            decision="pending" if state.get("require_final_confirmation") else "escalate",
            reason=refined.confidence_reasoning,
            execution_time_ms=int((time.monotonic() - t0) * 1000),
        ),
        event_key="synthesize",
    )
    return {"analysis": refined}


async def node_final_confirmation(state: ReviewState) -> dict[str, Any]:
    """UI-only final gate for refined escalation reviews."""
    if not state.get("require_final_confirmation"):
        return {"final_confirmation": "approve"}
    t0 = time.monotonic()
    analysis = state["analysis"]
    await audit(
        state,
        AuditEntry(
            agent_id=AGENT_ID,
            action="human_approval",
            confidence=analysis.confidence,
            risk_level=risk_level_for(analysis.confidence),
            decision="pending",
            reason="Waiting for final confirmation of the refined review",
            execution_time_ms=int((time.monotonic() - t0) * 1000),
        ),
        event_key="final_confirmation:pending",
    )
    response = interrupt({
        "kind": "refined_approval",
        "pr_url": state["pr_url"],
        "confidence": analysis.confidence,
        "confidence_reasoning": analysis.confidence_reasoning,
        "summary": analysis.summary,
        "comments": [comment.model_dump() for comment in analysis.comments],
        "risk_factors": analysis.risk_factors,
        "diff_preview": state["pr_diff"][:4000],
        "draft_review": _render_comment_body(state),
    })
    choice = response.get("choice")
    if choice not in {"approve", "reject", "edit"}:
        raise ValueError(f"Invalid final reviewer choice: {choice!r}")
    edited_review = response.get("edited_review") if choice == "edit" else None
    if choice == "edit" and not (edited_review or "").strip():
        raise ValueError("Edit requires a non-empty edited_review")
    feedback = response.get("feedback")
    await audit(
        state,
        AuditEntry(
            agent_id=AGENT_ID,
            action="human_approval",
            confidence=analysis.confidence,
            risk_level=risk_level_for(analysis.confidence),
            reviewer_id=os.environ.get("GITHUB_USER"),
            decision=choice,
            reason=feedback or "Final decision for refined review",
            execution_time_ms=int((time.monotonic() - t0) * 1000),
        ),
        event_key="final_confirmation:response",
    )
    return {
        "final_confirmation": choice,
        "human_feedback": feedback,
        "edited_review": edited_review,
    }


def build_graph(checkpointer):
    graph = StateGraph(ReviewState)
    for name, function in (
        ("fetch_pr", node_fetch_pr),
        ("analyze", node_analyze),
        ("route", node_route),
        ("auto_approve", node_auto_approve),
        ("human_approval", node_human_approval),
        ("commit", node_commit),
        ("escalate", node_escalate),
        ("synthesize", node_synthesize),
        ("final_confirmation", node_final_confirmation),
    ):
        graph.add_node(name, function)
    graph.add_edge(START, "fetch_pr")
    graph.add_edge("fetch_pr", "analyze")
    graph.add_edge("analyze", "route")
    graph.add_conditional_edges(
        "route",
        lambda state: state["decision"],
        {
            "auto_approve": "auto_approve",
            "human_approval": "human_approval",
            "escalate": "escalate",
        },
    )
    graph.add_edge("auto_approve", END)
    graph.add_edge("human_approval", "commit")
    graph.add_edge("escalate", "synthesize")
    graph.add_edge("synthesize", "final_confirmation")
    graph.add_edge("final_confirmation", "commit")
    graph.add_edge("commit", END)
    return graph.compile(checkpointer=checkpointer)


def handle_interrupt(payload: dict[str, Any]) -> dict[str, Any]:
    kind = payload["kind"]
    if kind in {"approval_request", "refined_approval"}:
        console.print(Panel.fit(
            payload["summary"],
            title=f"Review confirmation — conf={payload['confidence']:.0%}",
            border_style="green",
        ))
        choice = ""
        while choice not in {"approve", "reject", "edit"}:
            choice = console.input("approve/reject/edit? ").strip().lower()
        feedback = console.input("Feedback: ").strip() if choice != "approve" else ""
        answer: dict[str, Any] = {"choice": choice, "feedback": feedback}
        if choice == "edit":
            console.print(payload["draft_review"])
            edited = console.input("Edited review Markdown: ").strip()
            while not edited:
                edited = console.input("Edited review cannot be empty: ").strip()
            answer["edited_review"] = edited
        return answer
    if kind == "escalation":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Escalation — conf={payload['confidence']:.0%}",
            border_style="yellow",
        ))
        return {
            question: console.input(f"Q: {question}\nA: ").strip()
            for question in payload["questions"]
        }
    raise ValueError(f"Unknown interrupt kind: {kind}")


def _snapshot_interrupt(snapshot):
    for task in snapshot.tasks:
        if task.interrupts:
            return task.interrupts[0]
    return None


async def run(pr_url: str | None, thread_id: str | None) -> dict[str, Any]:
    thread_id = thread_id or str(uuid.uuid4())
    console.rule("[bold]Exercise 4 — SQLite audit trail[/bold]")
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    async with AsyncSqliteSaver.from_conn_string(db_path()) as checkpointer:
        await checkpointer.setup()
        app = build_graph(checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await app.aget_state(config)

        if snapshot.values:
            saved_url = snapshot.values.get("pr_url")
            if pr_url and saved_url and pr_url != saved_url:
                raise ValueError("--pr does not match the PR stored for this thread")
            pending_interrupt = _snapshot_interrupt(snapshot)
            if pending_interrupt is not None:
                result = {"__interrupt__": (pending_interrupt,)}
            elif not snapshot.next:
                result = dict(snapshot.values)
            else:
                result = await app.ainvoke(None, config)
        else:
            if not pr_url:
                raise ValueError("A new thread requires --pr")
            console.print(f"[dim]PR: {pr_url}[/dim]")
            result = await app.ainvoke(
                {
                    "pr_url": pr_url,
                    "thread_id": thread_id,
                    "require_final_confirmation": False,
                },
                config,
            )

        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            result = await app.ainvoke(Command(resume=handle_interrupt(payload)), config)

        console.rule("Final")
        console.print(f"final_action = {result.get('final_action')}")
        console.print(f"\n[dim]Replay:[/dim] uv run python -m audit.replay --thread {thread_id}")
        return result


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", help="GitHub pull-request URL (required for a new thread)")
    parser.add_argument("--thread", help="Resume or inspect an existing durable thread")
    args = parser.parse_args()
    if not args.pr and not args.thread:
        parser.error("provide --pr for a new review or --thread to resume one")
    asyncio.run(run(args.pr, args.thread))


if __name__ == "__main__":
    main()
