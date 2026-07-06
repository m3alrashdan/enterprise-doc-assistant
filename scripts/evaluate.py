"""Evaluation harness: retrieval hit-rate and answer faithfulness checks.

Runs the QA pairs in sample_data/eval_set.json against the configured
pipeline (uses the same Settings/env as the app - no server required):

- retrieval hit@k ... expected document appears in the retrieved chunks
- faithfulness   ... the answer cites sources, every citation points at a
                     retrieved chunk, and an expected keyword appears

Usage:
    .venv/bin/python scripts/evaluate.py               # real providers (.env)
    EMBEDDING_PROVIDER=fake LLM_PROVIDER=fake \\
        .venv/bin/python scripts/evaluate.py           # offline smoke run

Sample documents are ingested automatically if they are not in the store yet.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import Settings  # noqa: E402
from app.core.container import Container, build_container  # noqa: E402
from app.services.documents import DuplicateDocumentUpload  # noqa: E402
from scripts.generate_sample_pdf import generate  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
UPLOADABLE = {".pdf", ".md", ".html", ".htm", ".txt", ".docx"}


@dataclass
class CaseResult:
    question: str
    hit: bool
    cited: bool
    citations_grounded: bool
    keyword_found: bool

    @property
    def faithful(self) -> bool:
        return self.cited and self.citations_grounded and self.keyword_found


async def ensure_samples_ingested(container: Container) -> None:
    pdf_path = SAMPLE_DIR / "employee_handbook.pdf"
    if not pdf_path.exists():
        generate(pdf_path)
    for path in sorted(SAMPLE_DIR.iterdir()):
        if path.suffix.lower() not in UPLOADABLE:
            continue
        try:
            record = await container.document_service.register_upload(
                path.name, path.read_bytes(), uploader="evaluate.py", tag="sample"
            )
        except DuplicateDocumentUpload:
            continue
        await container.ingestion_service.ingest_document(record.id)
        refreshed = await container.document_service.get_document(record.id)
        print(f"  ingested {path.name}: {refreshed.status} ({refreshed.chunk_count} chunks)")


async def evaluate_case(container: Container, case: dict) -> CaseResult:
    question: str = case["question"]
    expected_document: str = case["expected_document"]
    keywords: list[str] = case.get("expected_keywords", [])

    chunks = await container.retrieval_service.retrieve(question)
    retrieved_documents = {chunk.metadata.get("document_name") for chunk in chunks}
    hit = expected_document in retrieved_documents

    result = await container.chat_service.query(question)
    cited = bool(result.citations)
    citations_grounded = all(
        citation.document_name in retrieved_documents for citation in result.citations
    )
    answer_lower = result.answer.lower()
    keyword_found = any(keyword.lower() in answer_lower for keyword in keywords)

    return CaseResult(
        question=question,
        hit=hit,
        cited=cited,
        citations_grounded=citations_grounded,
        keyword_found=keyword_found,
    )


async def main() -> int:
    settings = Settings()
    print(
        f"providers: embeddings={settings.embedding_provider} "
        f"llm={settings.llm_provider} | top_k={settings.top_k} "
        f"mmr={settings.use_mmr} rerank={settings.rerank_enabled} "
        f"hybrid={settings.hybrid_search_enabled}\n"
    )
    container = await build_container(settings)
    try:
        await ensure_samples_ingested(container)
        cases = json.loads((SAMPLE_DIR / "eval_set.json").read_text())

        results: list[CaseResult] = []
        print(f"\n{'hit':<5} {'faithful':<9} question")
        print("-" * 72)
        for case in cases:
            outcome = await evaluate_case(container, case)
            results.append(outcome)
            print(
                f"{'y' if outcome.hit else 'N':<5} {'y' if outcome.faithful else 'N':<9} "
                f"{outcome.question}"
            )
            if not outcome.faithful:
                flags = []
                if not outcome.cited:
                    flags.append("no citations")
                if not outcome.citations_grounded:
                    flags.append("citation outside retrieved set")
                if not outcome.keyword_found:
                    flags.append("expected keyword missing from answer")
                print(f"      -> {', '.join(flags)}")

        total = len(results)
        hits = sum(r.hit for r in results)
        faithful = sum(r.faithful for r in results)
        print("-" * 72)
        print(f"retrieval hit rate : {hits}/{total} ({hits / total:.0%})")
        print(f"faithfulness       : {faithful}/{total} ({faithful / total:.0%})")
        return 0 if hits == total else 1
    finally:
        await container.shutdown()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
