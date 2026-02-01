#!/usr/bin/env python3
"""Reranker-based conversation compaction for Claude Code JSONL histories.

Uses Qwen3-Reranker-0.6B to score and select system turns within a token budget.
All user turns are preserved. Short system turns are always kept. Long system turns
are scored by the reranker and greedily selected by adjusted relevance score.

Usage:
    uv run compact.py JSONL_FILE [--budget 80000] [--dry-run] [--verbose]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from lib.parser import parse_jsonl, extract_text
from lib.tokenizer import turn_tokens
from lib.scorer import Scorer, build_query, random_scores
from lib.selector import select_turns
from lib.formatter import print_stats, write_compacted_jsonl, write_scores_csv

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compact Claude Code conversation histories using reranker scoring."
    )
    parser.add_argument("jsonl_file", type=Path, help="Path to the JSONL conversation file")
    parser.add_argument("--budget", type=int, default=80_000, help="Target token budget (default: 80000)")
    parser.add_argument("--short-threshold", type=int, default=300, help="System turns <= this many tokens are always kept (default: 300)")
    parser.add_argument("--device", type=str, default="cuda:0", help="PyTorch device (default: cuda:0)")
    parser.add_argument("--output", type=Path, default=None, help="Write compacted JSONL to this file")
    parser.add_argument("--scores-file", type=Path, default=None, help="Write scores CSV to this file")
    parser.add_argument("--batch-size", type=int, default=8, help="Reranker batch size (default: 8)")
    parser.add_argument("--dry-run", action="store_true", help="Skip model loading, use random scores")
    parser.add_argument("--verbose", action="store_true", help="Show detailed score breakdown")

    args = parser.parse_args()

    if not args.jsonl_file.exists():
        console.print(f"[red]Error: {args.jsonl_file} not found[/red]")
        return 1

    # Parse
    console.print(f"Parsing {args.jsonl_file.name}...")
    turns = parse_jsonl(args.jsonl_file)

    user_turns = [t for t in turns if t.kind == "user"]
    system_turns = [t for t in turns if t.kind == "system"]

    console.print(f"  {len(turns)} turns total: {len(user_turns)} user, {len(system_turns)} system")

    # Token estimation
    console.print("Estimating tokens...")
    token_counts: dict[int, int] = {}
    for turn in turns:
        token_counts[turn.index] = turn_tokens(turn)

    total_tokens = sum(token_counts.values())
    console.print(f"  {total_tokens:,} tokens total")

    if total_tokens <= args.budget:
        console.print(f"[green]Already within budget ({total_tokens:,} <= {args.budget:,}), nothing to compact.[/green]")
        return 0

    # Identify long system turns that need scoring
    long_system = [
        t for t in system_turns
        if token_counts.get(t.index, 0) > args.short_threshold
    ]
    short_system = [
        t for t in system_turns
        if token_counts.get(t.index, 0) <= args.short_threshold
    ]

    console.print(f"  {len(short_system)} short system turns (always kept)")
    console.print(f"  {len(long_system)} long system turns (to be scored)")

    if not long_system:
        console.print("[yellow]No long system turns to score.[/yellow]")
        return 0

    # Score
    if args.dry_run:
        console.print("[yellow]Dry run: using random scores[/yellow]")
        scored = random_scores(long_system, token_counts)
    else:
        console.print(f"Loading reranker on {args.device}...")
        scorer = Scorer(device=args.device)

        query = build_query(user_turns)
        if args.verbose:
            console.print(f"  Query ({len(query)} chars): {query[:200]}...")

        console.print(f"Scoring {len(long_system)} turns (batch_size={args.batch_size})...")
        scored = scorer.score_turns(long_system, query, token_counts, batch_size=args.batch_size)

    # Select
    result = select_turns(
        turns=turns,
        scored=scored,
        token_counts=token_counts,
        budget=args.budget,
        short_threshold=args.short_threshold,
    )

    # Output
    print_stats(result, verbose=args.verbose)

    if args.output:
        write_compacted_jsonl(result, args.output)

    if args.scores_file:
        kept_indices = {t.index for t in result.kept_turns}
        write_scores_csv(scored, kept_indices, args.scores_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
