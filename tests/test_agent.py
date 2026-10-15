from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import ValidationError

from common.github import CommentResult, PullRequest, parse_pr_url
from common.schemas import AuditEntry, PRAnalysis, ReviewComment


PR_URL = "https://github.com/VinUni-AI20k/PR-Demo/pull/1"


def analysis(confidence: float, *, questions: list[str] | None = None) -> PRAnalysis:
    return PRAnalysis(
        summary="Adds a reviewed feature.",
        risk_factors=["migration may be required"],
        comments=[
            ReviewComment(
                file="app.py", line=12, severity="issue", body="Handle the failure path."
            )
        ],
        confidence=confidence,
        confidence_reasoning="Confidence reflects the available diff and missing tests.",
        escalation_questions=questions or [],
    )


def fake_pr(url: str = PR_URL) -> PullRequest:
    return PullRequest(
        url=url,
        owner="VinUni-AI20k",
        repo="PR-Demo",
        number=1,
        title="Test PR",
        author="student",
        base_ref="main",
        head_ref="feature",
        head_sha="abcdef123456",
        diff="diff --git a/app.py b/app.py\n+new code",
        files_changed=["app.py"],
    )


class FakeLLM:
    def __init__(self, outputs: list[PRAnalysis]):
        self.outputs = list(outputs)

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        return self.outputs.pop(0)

    async def ainvoke(self, _messages):
        return self.outputs.pop(0)


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (0.57, "escalate"),
        (0.58, "human_approval"),
        (0.72, "human_approval"),
        (0.73, "auto_approve"),
        (0.90, "auto_approve"),
    ],
)
def test_confidence_routing_boundaries(confidence, expected):
    from exercises.exercise_1_confidence import node_route

    assert node_route({"analysis": analysis(confidence)})["decision"] == expected


@pytest.mark.parametrize(
    "url",
    [PR_URL, PR_URL + "/"],
)
def test_parse_pr_url_valid(url):
    assert parse_pr_url(url) == ("VinUni-AI20k", "PR-Demo", 1)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/a/b/pull/1",
        "https://github.com/a/b/issues/1",
        "https://github.com/a/b/pull/not-a-number",
        "prefix https://github.com/a/b/pull/1 suffix",
    ],
)
def test_parse_pr_url_invalid(url):
    with pytest.raises(ValueError):
        parse_pr_url(url)


def test_pr_analysis_validation():
    valid = analysis(0.58)
    assert valid.comments[0].severity == "issue"
    with pytest.raises(ValidationError):
        PRAnalysis(
            summary="bad",
            confidence=1.1,
            confidence_reasoning="outside valid range",
        )


def test_all_exercises_share_risk_calibration_prompt():
    from common.prompts import PR_REVIEW_SYSTEM_PROMPT
    from exercises import (
        exercise_1_confidence,
        exercise_2_hitl,
        exercise_3_escalation,
        exercise_4_audit,
    )

    for module in (
        exercise_1_confidence,
        exercise_2_hitl,
        exercise_3_escalation,
        exercise_4_audit,
    ):
        assert module.PR_REVIEW_SYSTEM_PROMPT == PR_REVIEW_SYSTEM_PROMPT
    assert "0.73-1.00" in PR_REVIEW_SYSTEM_PROMPT
    assert "0.58-0.72" in PR_REVIEW_SYSTEM_PROMPT
    assert "0.00-0.57" in PR_REVIEW_SYSTEM_PROMPT
    assert "It is NOT confidence that an individual bug exists" in PR_REVIEW_SYSTEM_PROMPT
    assert "github.com" not in PR_REVIEW_SYSTEM_PROMPT


@pytest.mark.parametrize(
    ("choice", "expected_action", "expected_posts"),
    [
        ("approve", "committed", 1),
        ("reject", "rejected", 0),
        ("edit", "committed", 1),
    ],
)
def test_hitl_interrupt_resume_choices(monkeypatch, choice, expected_action, expected_posts):
    import exercises.exercise_2_hitl as exercise

    posts: list[str] = []
    monkeypatch.setattr(exercise, "fetch_pr", fake_pr)
    monkeypatch.setattr(exercise, "get_llm", lambda: FakeLLM([analysis(0.65)]))

    def post(_url, body, **_kwargs):
        posts.append(body)
        return CommentResult("posted", "https://github.com/comment/1")

    monkeypatch.setattr(exercise, "post_review_comment", post)
    graph = exercise.build_graph()
    config = {"configurable": {"thread_id": f"hitl-{choice}"}}
    result = graph.invoke({"pr_url": PR_URL, "thread_id": f"hitl-{choice}"}, config)
    assert result["__interrupt__"][0].value["kind"] == "approval_request"
    response = {"choice": choice, "feedback": "reviewer feedback"}
    if choice == "edit":
        response["edited_review"] = "### Edited by the reviewer"
    result = graph.invoke(Command(resume=response), config)
    assert result["final_action"] == expected_action
    assert len(posts) == expected_posts
    if choice == "edit":
        assert posts == ["### Edited by the reviewer"]


def test_escalation_answers_and_synthesis(monkeypatch):
    import exercises.exercise_3_escalation as exercise

    initial = analysis(0.40, questions=["How are passwords hashed?", "How is cloud sync secured?"])
    refined = analysis(0.55)
    llm = FakeLLM([initial, refined])
    posted: list[str] = []
    monkeypatch.setattr(exercise, "fetch_pr", fake_pr)
    monkeypatch.setattr(exercise, "get_llm", lambda: llm)
    monkeypatch.setattr(
        exercise,
        "post_review_comment",
        lambda _url, body, **_kwargs: posted.append(body) or CommentResult("posted"),
    )
    graph = exercise.build_graph()
    config = {"configurable": {"thread_id": "escalation-flow"}}
    result = graph.invoke({"pr_url": PR_URL, "thread_id": "escalation-flow"}, config)
    payload = result["__interrupt__"][0].value
    assert payload["kind"] == "escalation"
    answers = {question: "Documented reviewer answer" for question in payload["questions"]}
    result = graph.invoke(Command(resume=answers), config)
    assert result["analysis"] == refined
    assert result["escalation_answers"] == answers
    assert result["final_action"] == "committed_after_escalation"
    assert len(posted) == 1


def test_audit_entry_storage_replay_and_worst_risk(monkeypatch, tmp_path):
    from common.db import list_review_sessions, replay_events, write_audit_event

    database = tmp_path / "audit.db"
    monkeypatch.setenv("HITL_DB_PATH", str(database))

    async def scenario():
        await write_audit_event(
            thread_id="audit-thread",
            pr_url=PR_URL,
            entry=AuditEntry(
                agent_id="test",
                action="analyze",
                confidence=0.9,
                risk_level="low",
                decision="pending",
                execution_time_ms=3,
            ),
            event_key="analyze",
        )
        await write_audit_event(
            thread_id="audit-thread",
            pr_url=PR_URL,
            entry=AuditEntry(
                agent_id="test",
                action="route",
                confidence=0.4,
                risk_level="high",
                decision="escalate",
                execution_time_ms=4,
            ),
            event_key="route",
        )
        # Same event key is intentionally ignored.
        await write_audit_event(
            thread_id="audit-thread",
            pr_url=PR_URL,
            entry=AuditEntry(
                agent_id="test",
                action="route",
                confidence=0.4,
                risk_level="high",
                decision="escalate",
                execution_time_ms=4,
            ),
            event_key="route",
        )
        return await replay_events("audit-thread"), await list_review_sessions()

    events, sessions = asyncio.run(scenario())
    assert [event["action"] for event in events] == ["analyze", "route"]
    assert sessions[0]["thread_id"] == "audit-thread"
    assert sessions[0]["worst_risk"] == "high"


def test_durable_checkpoint_resume_and_final_confirmation(monkeypatch, tmp_path):
    import exercises.exercise_4_audit as exercise

    database = tmp_path / "durable.db"
    monkeypatch.setenv("HITL_DB_PATH", str(database))
    questions = ["How is login protected?", "How are cloud conflicts resolved?"]
    llm = FakeLLM([analysis(0.4, questions=questions), analysis(0.52)])
    posts: list[str] = []
    monkeypatch.setattr(exercise, "fetch_pr", fake_pr)
    monkeypatch.setattr(exercise, "get_llm", lambda: llm)
    monkeypatch.setattr(
        exercise,
        "post_review_comment",
        lambda _url, body, **_kwargs: posts.append(body) or CommentResult("posted"),
    )
    config = {"configurable": {"thread_id": "durable-thread"}}

    async def first_process():
        async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
            await saver.setup()
            graph = exercise.build_graph(saver)
            return await graph.ainvoke(
                {
                    "pr_url": PR_URL,
                    "thread_id": "durable-thread",
                    "require_final_confirmation": True,
                },
                config,
            )

    first = asyncio.run(first_process())
    assert first["__interrupt__"][0].value["kind"] == "escalation"

    async def second_process():
        async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
            await saver.setup()
            graph = exercise.build_graph(saver)
            answers = {question: "Reviewer context" for question in questions}
            refined_interrupt = await graph.ainvoke(Command(resume=answers), config)
            assert refined_interrupt["__interrupt__"][0].value["kind"] == "refined_approval"
            final = await graph.ainvoke(Command(resume={"choice": "approve"}), config)
            # Calling an already-complete thread does not replay the commit side effect.
            same_final = await graph.ainvoke(None, config)
            return final, same_final

    final, same_final = asyncio.run(second_process())
    assert final["final_action"] == "committed"
    assert same_final["final_action"] == "committed"
    assert len(posts) == 1

    from common.db import replay_events

    events = asyncio.run(replay_events("durable-thread"))
    assert sum(e["action"] == "escalate" and e["decision"] == "pending" for e in events) == 1
    assert {e["action"] for e in events} >= {
        "fetch_pr", "analyze", "route", "escalate", "synthesize", "commit"
    }


def test_post_comment_idempotency_marker(monkeypatch):
    import common.github as github

    calls = {"post": 0}

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, *_args, **_kwargs):
            return Response([
                {
                    "body": "done\n<!-- hitl-pr-review:same-thread -->",
                    "html_url": "https://github.com/comment/existing",
                }
            ])

        def post(self, *_args, **_kwargs):
            calls["post"] += 1
            return Response({"html_url": "https://github.com/comment/new"})

    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setattr(github.httpx, "Client", Client)
    result = github.post_review_comment(PR_URL, "body", idempotency_key="same-thread")
    assert result.status == "already_posted"
    assert calls["post"] == 0


def test_streamlit_app_initializes(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("HITL_DB_PATH", str(tmp_path / "ui.db"))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "HITL PR Review Agent"
