"""Qwen3-Reranker-0.6B scoring for system turns.

Uses the reranker to score how relevant each system turn is to the current
task direction (derived from recent user messages).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .parser import Turn, extract_text

MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MAX_RERANKER_TOKENS = 8192

INSTRUCTION = (
    "Given an AI coding assistant conversation, determine if this assistant response "
    "contains information needed to continue the task. Prioritize: code changes, "
    "decisions, errors, file paths, architectural context, unfinished work."
)

PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query "
    "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    "<|im_end|>\n"
    "<|im_start|>user\n"
)

SUFFIX = (
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
)


@dataclass
class ScoredTurn:
    """A turn with its relevance score."""

    turn: Turn
    score: float
    tokens: int


def build_query(user_turns: list[Turn], max_chars: int = 4000) -> str:
    """Build a query from the last 2-3 user messages."""
    recent = user_turns[-3:] if len(user_turns) >= 3 else user_turns
    parts = [extract_text(t) for t in recent]
    query = "\n---\n".join(parts)
    if len(query) > max_chars:
        query = query[-max_chars:]
    return query


class Scorer:
    """Scores system turns using Qwen3-Reranker-0.6B."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
        dtype = torch.float32 if device == "cpu" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=dtype,
        ).to(device).eval()
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")

    def _format_input(self, query: str, document: str) -> str:
        """Format a single query-document pair for the reranker."""
        return (
            f"{PREFIX}"
            f"<Instruct>: {INSTRUCTION}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
            f"{SUFFIX}"
        )

    def _truncate_document(self, document: str, query: str) -> str:
        """Truncate document to fit within the reranker's context window.

        Leaves room for the query, instruction, and template tokens.
        """
        # Estimate overhead tokens from template + query + instruction
        overhead = self._format_input(query, "")
        overhead_tokens = len(self.tokenizer.encode(overhead, add_special_tokens=False))
        max_doc_tokens = MAX_RERANKER_TOKENS - overhead_tokens - 64  # safety margin

        if max_doc_tokens <= 0:
            return document[:500]

        doc_tokens = self.tokenizer.encode(document, add_special_tokens=False)
        if len(doc_tokens) <= max_doc_tokens:
            return document

        # Truncate and decode back
        truncated = doc_tokens[:max_doc_tokens]
        return self.tokenizer.decode(truncated, skip_special_tokens=True)

    def _score_one(self, text: str) -> float:
        """Score a single formatted input and return P(yes)."""
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_RERANKER_TOKENS,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)
            # Last token logits only
            logits = outputs.logits[0, -1, :]
            yes_no = logits[[self.no_id, self.yes_id]]
            probs = torch.softmax(yes_no, dim=0)
            return probs[1].item()  # P(yes)

    def score_turns(
        self,
        system_turns: list[Turn],
        query: str,
        token_counts: dict[int, int],
        batch_size: int = 1,  # unused, kept for API compat
    ) -> list[ScoredTurn]:
        """Score a list of system turns against the query.

        Processes one turn at a time to stay within GPU memory.
        Returns ScoredTurn objects with scores in [0, 1].
        """
        results: list[ScoredTurn] = []

        for i, turn in enumerate(system_turns):
            doc = extract_text(turn)
            doc = self._truncate_document(doc, query)
            text = self._format_input(query, doc)
            score = self._score_one(text)

            results.append(ScoredTurn(
                turn=turn,
                score=score,
                tokens=token_counts.get(turn.index, 0),
            ))

            if (i + 1) % 10 == 0:
                print(f"  scored {i + 1}/{len(system_turns)}", flush=True)

        return results


def random_scores(
    system_turns: list[Turn],
    token_counts: dict[int, int],
) -> list[ScoredTurn]:
    """Generate random scores for dry-run testing."""
    return [
        ScoredTurn(
            turn=turn,
            score=random.random(),
            tokens=token_counts.get(turn.index, 0),
        )
        for turn in system_turns
    ]
