"""CLI entry points for ops-agent: review-pr and scaffold subcommands."""

import argparse
import sys


def _require_settings() -> None:
    """Fail fast with a clear message if required env vars are missing."""
    try:
        from ops_agent.config import get_settings

        get_settings()
    except Exception as exc:
        print(f"Error: missing or invalid configuration — {exc}", file=sys.stderr)
        print("Copy .env.example to .env and fill in the required values.", file=sys.stderr)
        sys.exit(1)


def _cmd_review_pr(args: argparse.Namespace) -> None:
    """Run Graph A: review a Renovate PR and post a comment."""
    _require_settings()

    from ops_agent.graphs.renovate_review import run
    from ops_agent.state import Verdict

    verdict: Verdict = run(pr_index=args.pr, owner=args.owner, repo=args.repo)

    print(f"\nDecision: {verdict.decision}")
    print(f"Summary:  {verdict.summary}")

    if verdict.findings:
        print("\nFindings:")
        for finding in verdict.findings:
            print(f"  [{finding.source}] {finding.claim}")
            print(f"    Quote: {finding.quote!r}")


def _cmd_scaffold(args: argparse.Namespace) -> None:
    """Run Graph B: scaffold a deployment from a prompt and open a PR."""
    _require_settings()

    if args.prompt_file:
        try:
            with open(args.prompt_file) as fh:
                prompt = fh.read()
        except OSError as exc:
            print(f"Error reading prompt file: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        prompt = args.prompt

    from ops_agent.graphs.scaffold_deploy import run

    pr_url: str | None = run(prompt=prompt)

    if pr_url:
        print(f"\nPR opened: {pr_url}")
    else:
        print("\nScaffolding complete (no PR URL returned).")


def main() -> None:
    """Entry point for the ops-agent CLI."""
    parser = argparse.ArgumentParser(
        prog="ops-agent",
        description="Agentic ops pipeline: Renovate PR review and deployment scaffolding.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # --- review-pr ---
    review_p = sub.add_parser(
        "review-pr",
        help="Review a Renovate dependency-update PR and post a verdict comment.",
    )
    review_p.add_argument("--owner", required=True, metavar="OWNER", help="Gitea repo owner")
    review_p.add_argument("--repo", required=True, metavar="REPO", help="Gitea repo name")
    review_p.add_argument(
        "--pr", required=True, type=int, metavar="N", help="PR index (integer)"
    )
    review_p.set_defaults(func=_cmd_review_pr)

    # --- scaffold ---
    scaffold_p = sub.add_parser(
        "scaffold",
        help="Scaffold a deployment from a natural-language prompt and open a Gitea PR.",
    )
    prompt_grp = scaffold_p.add_mutually_exclusive_group(required=True)
    prompt_grp.add_argument("--prompt", metavar="TEXT", help="Deployment request as inline text")
    prompt_grp.add_argument(
        "--prompt-file", metavar="PATH", help="Path to a file containing the deployment request"
    )
    scaffold_p.set_defaults(func=_cmd_scaffold)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
