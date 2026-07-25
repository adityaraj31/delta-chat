"""Grounded answer pipeline — retrieval-augmented generation with citation enforcement."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.chat.index import IndexEntry, RetrievalIndex
from src.chat.llm import LLMClient
from src.config import ChatConfig

# Matches [PID:source:page:element_id] or [delta:idx]
_CITATION_RE = re.compile(r"\[(?:PID:[^\]]+|delta:\d+)\]")

_SYSTEM_PROMPT = """\
You are a PID (P&ID) document analysis assistant. You answer questions about \
two revisions of a P&ID document and the differences between them.

RULES:
1. You MUST cite your sources using the format [PID:source:page:element_id] or [delta:entry_index].
2. Every factual claim MUST have at least one citation.
3. If the provided context does not contain enough information to answer, say \
"I don't have enough information to answer this question based on the available documents."
4. Never fabricate citations or information.
5. Be concise and precise. Use engineering terminology appropriate for P&ID documents.
"""

_CONTEXT_HEADER = "The following context was retrieved from the PID documents and delta report:\n\n"

_ANSWER_TEMPLATE = """\
{context_header}

---

Question: {question}

Answer the question using ONLY the information above. Cite every claim."""


@dataclass
class Answer:
    """A grounded answer with provenance."""
    text: str
    citations: list[str]
    retrieved_entries: list[IndexEntry]


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
        """Answer a question using retrieval + grounded generation."""
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
        user_msg = _ANSWER_TEMPLATE.format(context_header=_CONTEXT_HEADER + context, question=question)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        response = self._llm.chat(messages, temperature=0.0)

        # Extract citations from the response
        citations = _CITATION_RE.findall(response)

        return Answer(text=response, citations=citations, retrieved_entries=entries)
