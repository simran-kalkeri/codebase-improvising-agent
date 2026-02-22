#!/usr/bin/env python3
"""
main.py — CLI entry point for the Self-Improving Codebase Modernization Agent.

Usage:
    python main.py --repo <path_to_repo> --goal "Add type hints and remove deprecated syntax"
"""

import argparse
import sys
from pathlib import Path

from modernizer_agent.config import (
    DATABASE_PATH,
    LOG_LEVEL,
    MAX_RETRIES_PER_ITEM,
    MAX_TOTAL_ITERATIONS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    SYSTEM_PROMPT_PATH,
)
from modernizer_agent.utils.logger import get_logger

log = get_logger("modernizer_agent.main")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="modernizer-agent",
        description="Self-improving codebase modernization agent powered by local LLMs.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Path to the target repository to modernize.",
    )
    parser.add_argument(
        "--goal",
        type=str,
        required=True,
        help='Modernization goal, e.g. "Add type hints and remove deprecated syntax".',
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=MAX_TOTAL_ITERATIONS,
        help=f"Maximum total agent iterations (default: {MAX_TOTAL_ITERATIONS}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Plan only — do not apply any changes.",
    )

    args = parser.parse_args(argv)

    # Validate repo path.
    repo_path = Path(args.repo).resolve()
    if not repo_path.is_dir():
        parser.error(f"Repository path does not exist or is not a directory: {repo_path}")
    args.repo = str(repo_path)

    return args


def print_banner(args: argparse.Namespace) -> None:
    """Print a startup summary so the user knows what is configured."""
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║           Codebase Modernization Agent  v0.1.0              ║
╠══════════════════════════════════════════════════════════════╣
║  Repository   : {args.repo:<43}║
║  Goal         : {args.goal[:43]:<43}║
║  LLM          : {OLLAMA_MODEL:<43}║
║  Ollama URL   : {OLLAMA_BASE_URL:<43}║
║  Max iters    : {str(args.max_iterations):<43}║
║  Max retries  : {str(MAX_RETRIES_PER_ITEM):<43}║
║  Dry run      : {str(args.dry_run):<43}║
║  Log level    : {LOG_LEVEL:<43}║
║  DB path      : {str(DATABASE_PATH):<43}║
║  Prompt file  : {str(SYSTEM_PROMPT_PATH):<43}║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def main(argv: list[str] | None = None) -> None:
    """Entry point — parse args, show config, and (in later phases) start the agent loop."""
    args = parse_args(argv)

    log.info(
        "Agent starting",
        extra={
            "action": "startup",
            "iteration": 0,
            "tool": "cli",
            "result": "ok",
        },
    )

    print_banner(args)

    # --- Start the agent loop ---
    from modernizer_agent.agent.controller import Controller

    controller = Controller(
        repo_path=args.repo,
        goal=args.goal,
        dry_run=args.dry_run,
        max_iterations=args.max_iterations,
    )

    try:
        controller.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting.")
        sys.exit(1)
    except Exception as exc:
        log.error(f"Agent failed: {exc}", extra={
            "action": "crash", "error": str(exc),
        })
        print(f"\n❌ Agent error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
