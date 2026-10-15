"""Exercise 3 — Escalation branch with reviewer Q&A.

When confidence < 58%, the agent doesn't ask approve/reject — it asks specific
clarifying questions and then synthesizes a refined review from the answers.
"""

from __future__ import annotations

import argparse
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.prompts import PR_REVIEW_SYSTEM_PROMPT
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
)


console = Console()


def node_fetch_pr(state):
    console.print("[cyan]-> fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]OK[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {"pr_title": pr.title, "pr_diff": pr.diff, "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha}


def node_analyze(state):
    console.print("[cyan]-> analyze[/cyan]")
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        analysis = llm.invoke([
            {"role": "system", "content": PR_REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": f"Title: {state['pr_title']}\nDiff:\n{state['pr_diff']}"},
        ])
    console.print(f"  [green]OK[/green] confidence={analysis.confidence:.0%}, {len(analysis.escalation_questions)} question(s)")
    return {"analysis": analysis}


def node_route(state):
    console.print("[cyan]-> route[/cyan]")
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD: decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:    decision = "escalate"
    else:                           decision = "human_approval"
    console.print(f"  [green]OK[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    return {"decision": decision}


def node_escalate(state: ReviewState) -> dict:
    """Ask the reviewer specific questions; return their answers in state."""
    a = state["analysis"]
    questions = a.escalation_questions
    if not questions:
        context = ", ".join(a.risk_factors[:2]) or ", ".join(state.get("pr_files", [])[:2])
        questions = [
            f"What requirement or control addresses this identified risk: {context}?",
            "What tests or rollout checks verify the changed behavior in this diff?",
        ]
    questions = questions[:4]
    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
    })
    if not isinstance(answers, dict):
        raise ValueError("Escalation answers must be a question-to-answer mapping")
    return {"escalation_answers": answers}


def node_synthesize(state: ReviewState) -> dict:
    """Re-prompt LLM with the reviewer's answers and produce a refined review."""
    initial = state["analysis"]
    answers = state.get("escalation_answers") or {}
    qa = "\n\n".join(f"Question: {q}\nReviewer answer: {answer}" for q, answer in answers.items())
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining the review...[/dim]"):
        refined: PRAnalysis = llm.invoke([
            {
                "role": "system",
                "content": (
                    "You are a senior code reviewer refining an earlier PR analysis with "
                    "human context. Return a complete PRAnalysis. Re-evaluate every finding "
                    "against the original diff and reviewer answers. Do not force confidence "
                    "to increase; use the score warranted by the evidence."
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
    return {"analysis": refined}


def node_human_approval(state):
    a = state["analysis"]
    response = interrupt({
        "kind": "approval_request", "pr_url": state["pr_url"],
        "confidence": a.confidence, "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
        "draft_review": _render_comment_body(state),
    })
    choice = response.get("choice")
    if choice not in {"approve", "reject", "edit"}:
        raise ValueError(f"Invalid reviewer choice: {choice!r}")
    edited = response.get("edited_review") if choice == "edit" else None
    if choice == "edit" and not (edited or "").strip():
        raise ValueError("Edit requires a non-empty edited_review")
    return {
        "human_choice": choice,
        "human_feedback": response.get("feedback"),
        "edited_review": edited,
    }


def _render_comment_body(state) -> str:
    if state.get("edited_review"):
        return state["edited_review"].strip()
    a = state["analysis"]
    lines = [f"### Automated review (confidence {a.confidence:.0%})", "", a.summary, ""]
    for c in a.comments:
        lines.append(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    if state.get("escalation_answers"):
        lines.append("\n_Reviewer answered escalation questions:_")
        for q, ans in state["escalation_answers"].items():
            lines.append(f"> **{q}** {ans}")
    return "\n".join(lines)


def _post(state, label: str) -> dict:
    body = _render_comment_body(state)
    try:
        result = post_review_comment(
            state["pr_url"], body, idempotency_key=state.get("thread_id")
        )
        if result.status == "dry_run":
            console.print("  [yellow]-[/yellow] dry-run: comment was not posted")
            return {"final_action": "dry_run", "posted_comment_body": body}
        console.print(f"  [green]OK[/green] {result.status} at {state['pr_url']}")
        return {
            "final_action": label,
            "posted_comment_body": body,
            "comment_url": result.url,
        }
    except Exception as e:
        console.print(f"  [red]ERROR[/red] post failed: {e}")
        return {"final_action": "commit_failed"}


def node_commit(state):
    console.print("[cyan]-> commit[/cyan]")
    # Two paths converge here:
    #   1. human_approval → commit (only post if approved)
    #   2. escalate → synthesize → commit (always post the refined review)
    if state.get("escalation_answers"):
        return _post(state, "committed_after_escalation")
    if state.get("human_choice") in {"approve", "edit"}:
        return _post(state, "committed")
    console.print(f"  [yellow]-[/yellow] skipping comment (choice={state.get('human_choice')})")
    return {"final_action": "rejected"}


def node_auto_approve(state):
    console.print("[cyan]-> auto_approve[/cyan]  [dim]high confidence - posting directly[/dim]")
    return _post(state, "auto_approved")


def build_graph():
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("commit", node_commit), ("escalate", node_escalate), ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("commit", END)
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "commit")
    return g.compile(checkpointer=MemorySaver())


def handle_interrupt(payload):
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Approve? conf={payload['confidence']:.0%}",
            border_style="green",
        ))
        choice = console.input("approve/reject/edit? ").strip().lower()
        answer = {"choice": choice, "feedback": console.input("Feedback: ").strip()}
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
            title=f"Escalation conf={payload['confidence']:.0%}",
            border_style="yellow",
        ))
        return {q: console.input(f"Q: {q}\nA: ").strip() for q in payload["questions"]}
    raise ValueError(kind)


def main():
    load_dotenv()
    p = argparse.ArgumentParser(); p.add_argument("--pr", required=True)
    args = p.parse_args()

    console.rule("[bold]Exercise 3 — escalation with reviewer Q&A[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    result = app.invoke({"pr_url": args.pr, "thread_id": thread_id}, cfg)
    while "__interrupt__" in result:
        result = app.invoke(Command(resume=handle_interrupt(result["__interrupt__"][0].value)), cfg)

    console.rule("Final")
    console.print(f"final_action = {result.get('final_action')}")
    if "analysis" in result:
        console.print(f"final confidence = {result['analysis'].confidence:.0%}")


if __name__ == "__main__":
    main()
