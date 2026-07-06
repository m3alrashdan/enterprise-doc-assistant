"""API tests for chat: query, citations, conversation memory, streaming."""

from __future__ import annotations

import json

import httpx

HR_DOC = (
    b"# Vacation Policy\n"
    b"Full-time employees accrue twenty vacation days per calendar year.\n\n"
    b"# Expense Policy\n"
    b"Meal expenses are reimbursed up to fifty dollars per day.\n"
)
IT_DOC = b"# VPN Policy\n" b"The corporate vpn is mandatory for remote database access.\n"


async def upload(client: httpx.AsyncClient, name: str, content: bytes) -> str:
    resp = await client.post(
        "/api/v1/documents/upload",
        files=[("files", (name, content, "application/octet-stream"))],
    )
    result = resp.json()["results"][0]
    assert result["accepted"], result
    return result["document_id"]


async def test_query_returns_cited_answer(client: httpx.AsyncClient) -> None:
    await upload(client, "hr.md", HR_DOC)

    resp = await client.post(
        "/api/v1/chat/query", json={"question": "How many vacation days do employees accrue?"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "[1]" in body["answer"]
    assert body["citations"], "expected at least one citation"
    citation = body["citations"][0]
    assert citation["document_name"] == "hr.md"
    assert citation["index"] == 1
    assert citation["snippet"]
    assert body["model_used"] == "fake:fake-model"
    assert body["latency_ms"] >= 0
    assert body["conversation_id"]


async def test_empty_corpus_yields_not_found_answer(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/chat/query", json={"question": "What is the meaning of life?"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "could not find an answer" in body["answer"]
    assert body["citations"] == []


async def test_document_filter_restricts_sources(client: httpx.AsyncClient) -> None:
    await upload(client, "hr.md", HR_DOC)
    it_id = await upload(client, "it.md", IT_DOC)

    resp = await client.post(
        "/api/v1/chat/query",
        json={"question": "What is mandatory for remote access?", "document_ids": [it_id]},
    )
    body = resp.json()
    assert body["citations"]
    assert all(c["document_id"] == it_id for c in body["citations"])


async def test_conversation_memory_condenses_follow_up(client: httpx.AsyncClient, test_app) -> None:
    await upload(client, "hr.md", HR_DOC)

    first = await client.post(
        "/api/v1/chat/query", json={"question": "How many vacation days do employees get?"}
    )
    conversation_id = first.json()["conversation_id"]

    second = await client.post(
        "/api/v1/chat/query",
        json={"question": "What about meal expenses?", "conversation_id": conversation_id},
    )
    assert second.status_code == 200
    assert second.json()["conversation_id"] == conversation_id

    # the condense step ran: FakeLLM saw a prompt containing the history
    fake_llm = test_app.state.container.llm
    condense_calls = [call for call in fake_llm.calls if "Follow-up question:" in call[-1].content]
    assert condense_calls, "expected a condense prompt for the follow-up"
    assert "vacation days" in condense_calls[-1][-1].content

    # both turns are persisted
    history = await test_app.state.container.conversation_repo.get_messages(conversation_id)
    roles = [role for role, _ in history]
    assert roles == ["user", "assistant", "user", "assistant"]


async def test_validation_rejects_empty_question(client: httpx.AsyncClient) -> None:
    resp = await client.post("/api/v1/chat/query", json={"question": ""})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def parse_sse(payload: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in payload.strip().split("\n\n"):
        event_name = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if event_name is not None and data is not None:
            events.append((event_name, data))
    return events


async def test_stream_emits_sources_tokens_done(client: httpx.AsyncClient) -> None:
    await upload(client, "hr.md", HR_DOC)

    async with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={"question": "How many vacation days do employees accrue?"},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        payload = (await resp.aread()).decode()

    events = parse_sse(payload)
    names = [name for name, _ in events]
    assert names[0] == "sources"
    assert "token" in names
    assert names[-1] == "done"

    sources = events[0][1]["sources"]
    assert sources and sources[0]["document_name"] == "hr.md"

    done = events[-1][1]
    tokens = "".join(data["text"] for name, data in events if name == "token")
    assert done["answer"] == tokens.strip()
    assert done["citations"]
    assert done["model_used"] == "fake:fake-model"
    assert done["conversation_id"]
