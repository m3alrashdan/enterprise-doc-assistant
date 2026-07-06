"""Unit tests for prompt construction and citation assembly."""

from __future__ import annotations

from app.models.document import RetrievedChunk
from app.rag import pipeline


def make_chunk(i: int, content: str = "", **meta) -> RetrievedChunk:
    metadata = {
        "document_id": f"doc-{i}",
        "document_name": f"doc{i}.pdf",
        "chunk_index": 0,
        **meta,
    }
    return RetrievedChunk(
        id=f"doc-{i}:0",
        content=content or f"Content of source number {i}.",
        metadata=metadata,
        score=0.9 - i * 0.1,
    )


class TestFormatContext:
    def test_sources_are_numbered_with_provenance(self) -> None:
        chunks = [make_chunk(1, page=4), make_chunk(2, section="Benefits")]
        context = pipeline.format_context(chunks)
        assert "[1] (document: doc1.pdf, page: 4)" in context
        assert "[2] (document: doc2.pdf, section: Benefits)" in context
        assert "Content of source number 1." in context


class TestBuildMessages:
    def test_answer_messages_carry_rules_and_context(self) -> None:
        messages = pipeline.build_answer_messages("What is X?", [make_chunk(1)])
        assert messages[0].role == "system"
        assert "ONLY the numbered context sources" in messages[0].content
        assert "Context sources:" in messages[-1].content
        assert "What is X?" in messages[-1].content

    def test_condense_messages_include_history(self) -> None:
        history = [("user", "What is the vacation policy?"), ("assistant", "20 days [1].")]
        messages = pipeline.build_condense_messages(history, "What about sick leave?")
        assert "user: What is the vacation policy?" in messages[-1].content
        assert "Follow-up question: What about sick leave?" in messages[-1].content


class TestAssembleCitations:
    def test_maps_markers_to_chunks_in_order(self) -> None:
        chunks = [make_chunk(1), make_chunk(2), make_chunk(3)]
        citations = pipeline.assemble_citations("Fact one [2]. Fact two [1].", chunks)
        assert [c.index for c in citations] == [1, 2]
        assert citations[0].document_name == "doc1.pdf"
        assert citations[1].document_name == "doc2.pdf"

    def test_ignores_out_of_range_and_duplicate_markers(self) -> None:
        chunks = [make_chunk(1)]
        citations = pipeline.assemble_citations("See [1] and [1] and [7].", chunks)
        assert [c.index for c in citations] == [1]

    def test_not_found_answer_has_no_citations(self) -> None:
        chunks = [make_chunk(1)]
        assert pipeline.assemble_citations(pipeline.NOT_FOUND_ANSWER, chunks) == []

    def test_answer_without_markers_cites_all_retrieved(self) -> None:
        chunks = [make_chunk(1), make_chunk(2)]
        citations = pipeline.assemble_citations("An uncited answer.", chunks)
        assert [c.index for c in citations] == [1, 2]

    def test_no_chunks_means_no_citations(self) -> None:
        assert pipeline.assemble_citations("Anything [1].", []) == []

    def test_snippet_is_truncated_at_word_boundary(self) -> None:
        long_content = "word " * 200
        citation = pipeline.chunk_to_citation(1, make_chunk(1, content=long_content))
        assert len(citation.snippet) <= 242
        assert citation.snippet.endswith("…")

    def test_citation_carries_page_section_and_score(self) -> None:
        chunk = make_chunk(1, page=7, section="Travel")
        citation = pipeline.chunk_to_citation(1, chunk)
        assert citation.page == 7
        assert citation.section == "Travel"
        assert citation.score == round(chunk.score, 4)
