#!/usr/bin/env python3
"""Conversation compaction for Claude Code JSONL histories.

Two methods:
  embed  — Qwen3-Embedding-0.6B cosine similarity scoring (needs GPU/CPU + model)
  dedup  — Suffix automaton dedup, scores by unique content ratio (no model, instant)

Usage:
    uv run compact.py JSONL_FILE --method embed [--budget 80000] [--device cuda:0]
    uv run compact.py JSONL_FILE --method dedup [--budget 80000] [--min-repeat-len 64]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console

from lib.parser import parse_jsonl, extract_text
from lib.tokenizer import turn_tokens
from lib.scorer import ScoredTurn, build_query, random_scores
from lib.selector import select_turns
from lib.formatter import print_stats, write_compacted_jsonl, write_scores_csv

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compact Claude Code conversation histories."
    )
    parser.add_argument("jsonl_file", type=Path, help="Path to the JSONL conversation file")
    parser.add_argument("--method", choices=["embed", "dedup"], default="embed", help="Scoring method (default: embed)")
    parser.add_argument("--budget", type=int, default=80_000, help="Target token budget (default: 80000)")
    parser.add_argument("--short-threshold", type=int, default=300, help="System turns <= this many tokens are always kept (default: 300)")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device for embed method (default: cpu)")
    parser.add_argument("--output", type=Path, default=None, help="Write compacted JSONL to this file")
    parser.add_argument("--scores-file", type=Path, default=None, help="Write scores CSV to this file")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size (default: 16)")
    parser.add_argument("--min-repeat-len", type=int, default=64, help="Min repeated substring length for dedup (default: 64)")
    parser.add_argument("--dry-run", action="store_true", help="Skip model loading, use random scores")
    parser.add_argument("--verbose", action="store_true", help="Show detailed score breakdown")

    args = parser.parse_args()

    if not args.jsonl_file.exists():
        console.print(f"[red]Error: {args.jsonl_file} not found[/red]")
        return 1

    t_start = time.monotonic()

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
    scored: list[ScoredTurn]

    if args.dry_run:
        console.print("[yellow]Dry run: using random scores[/yellow]")
        scored = random_scores(long_system, token_counts)

    elif args.method == "dedup":
        console.print(f"Dedup scoring (min_repeat_len={args.min_repeat_len})...")
        from lib.dedup import dedup_scores
        scored = dedup_scores(turns, long_system, token_counts, min_repeat_len=args.min_repeat_len)

    elif args.method == "embed":
        console.print(f"Loading embedding model on {args.device}...")
        from lib.scorer import Scorer
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

    t_elapsed = time.monotonic() - t_start

    # Output
    print_stats(result, verbose=args.verbose)
    console.print(f"\nMethod: {args.method} | Wall time: {t_elapsed:.1f}s")

    if args.output:
        write_compacted_jsonl(result, args.output)

    if args.scores_file:
        kept_indices = {t.index for t in result.kept_turns}
        write_scores_csv(scored, kept_indices, args.scores_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
