"""Grounded answer pipeline — retrieval-augmented generation with citation enforcement."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.chat.index import CitationValidation, IndexEntry, RetrievalIndex
from src.chat.llm import LLMClient
from src.config import ChatConfig

# Matches [PID:source:pN:element_id] or [delta:idx]
_CITATION_RE = re.compile(r"\[(?:PID:[^\]]+|delta:\d+)\]")

_MAX_CITATION_RETRIES = 1

_SYSTEM_PROMPT = """\
You are an expert Applied AI assistant for P&ID engineering revisions.
You answer questions about two revisions of a P&ID document and the differences between them.

CRITICAL RESPONSE RULES:
1. You MUST cite your sources using the format [PID:source:pN:element_id] or [delta:entry_index].
2. Every factual claim MUST have at least one citation.
3. NEVER display raw system internal IDs (for example, "Element with ID 7") in narrative text. Convert to engineering context such as element type, tag, or grid location.
4. DO NOT list items that had no change. Ignore entries where old and new values are equivalent.
5. If an element type is unclear, describe the revision using drawing location context (for example, "In Grid C-8, value updated from 9057 to 9015").
6. Group related changes when possible (for example: Equipment Tags, Line Specs, Setpoints, Notes).
7. If the provided context does not contain enough information to answer, say "I don't have enough information to answer this question based on the available documents."
8. Never fabricate citations or information.
9. Be concise and precise. Use engineering terminology appropriate for P&ID documents.
"""

_HEDGE_SUFFIX = """

IMPORTANT: Some citations in your response could not be validated against the index.
Please rewrite your answer. For any claims that cannot be supported by valid citations \
from the context, hedge by saying "I am not certain, but..." or omit the claim entirely. \
Only keep claims backed by verified citations."""

_CONTEXT_HEADER = "The following context was retrieved from the PID documents and delta report:\n\n"

_ANSWER_TEMPLATE = """\
{context_header}

---

Question: {question}

Answer the question using ONLY the information above. Cite every claim."""


@dataclass
class CitationReport:
    """Validation result for a single citation."""
    raw: str
    valid: bool
    reason: str = ""


@dataclass
class Answer:
    """A grounded answer with provenance."""
    text: str
    citations: list[str]
    validated_citations: list[CitationReport]
    retrieved_entries: list[IndexEntry]
    citation_rate: float  # fraction of citations that validated


class GroundedQA:
    """Retrieval-augmented Q&A over PID documents."""

    def __init__(
        self,
        index: RetrievalIndex,
        llm: LLMClient,
        config: ChatConfig | None = None,
    ) -> None:
        self._index = index
        self._llm = llm
        self._cfg = config or ChatConfig()

    def answer(self, question: str, top_k: int | None = None) -> Answer:
        """Answer a question using retrieval + grounded generation.

        Validates citations against the index. If validation fails,
        re-prompts the LLM once to hedge/drop unsupported claims.
        """
        k = top_k or self._cfg.top_k

        # Embed the question
        q_emb = self._llm.embed_single(question)

        # Retrieve
        results = self._index.search(q_emb, top_k=k)
        entries = [entry for entry, _ in results]

        # Build context
        context_parts = []
        for entry in entries:
            context_parts.append(entry.text)
        context = "\n".join(context_parts)

        # Generate
        user_msg = _ANSWER_TEMPLATE.format(
            context_header=_CONTEXT_HEADER + context,
            question=question,
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        response = self._llm.chat(messages, temperature=0.0)

        # Extract and validate citations
        citations = _CITATION_RE.findall(response)
        validations = self._index.validate_citations(citations)
        rate = _citation_rate(validations)

        # If citations failed validation, re-prompt once
        if rate < 1.0 and _MAX_CITATION_RETRIES > 0:
            retry_msg = user_msg + _HEDGE_SUFFIX
            retry_messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": retry_msg},
                {"role": "assistant", "content": response},
                {"role": "user", "content": (
                    "Some of your citations could not be verified. "
                    "Please revise your answer to only include claims "
                    "supported by valid citations, or hedge where uncertain."
                )},
            ]
            response = self._llm.chat(retry_messages, temperature=0.0)
            citations = _CITATION_RE.findall(response)
            validations = self._index.validate_citations(citations)
            rate = _citation_rate(validations)

        reports = [
            CitationReport(raw=v.raw, valid=v.valid, reason=v.reason)
            for v in validations
        ]

        return Answer(
            text=response,
            citations=citations,
            validated_citations=reports,
            retrieved_entries=entries,
            citation_rate=rate,
        )


def _citation_rate(validations: list[CitationValidation]) -> float:
    if not validations:
        return 1.0
    return sum(1 for v in validations if v.valid) / len(validations)
