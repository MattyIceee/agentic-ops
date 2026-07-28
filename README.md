# ops-agent

An agentic ops pipeline for a self-hosted homelab. Two LangGraph graphs share a common set of tools and personas to automate two recurring ops tasks:

1. **Renovate PR reviewer** — annotates dependency-bump PRs opened by Renovate with breaking-change findings extracted from changelogs and release notes, then posts a verdict comment on the Gitea PR.
2. **Deployment scaffolder** — takes a natural-language service request, researches the service, generates Flux `HelmRelease` or Kustomize manifests following your existing repo conventions, self-reviews them, and opens a Gitea PR.

Renovate's Merge Confidence rating remains the primary auto-merge gate. This agent is a **veto and annotation layer on top** — it cannot merge PRs, only comment on them.

---

## Architecture

```
ops-agent/
  llm/personas.py       # One model, four sampling-param bundles (research / coding / extract / instruct)
  tools/
    search.py           # SearXNG web search
    fetch.py            # URL fetcher → markdown text
    git_ops.py          # Local git: diff, branch, commit, push
    gitea.py            # Gitea REST client + LangChain tools
  graphs/
    renovate_review.py  # Graph A — PR annotation pipeline
    scaffold_deploy.py  # Graph B — manifest generation pipeline
  cli.py                # `ops-agent` entry point
```

**Personas are sampling parameters, not model aliases.** There is one llama.cpp model (`qwen3.6-a3b`). A persona is a bundle of `temperature`, `top_p`, `top_k`, `presence_penalty`, and `enable_thinking` passed per-request:

| Persona   | `enable_thinking` | Use                                 |
|-----------|-------------------|-------------------------------------|
| `research`  | True              | Web search, evidence gathering      |
| `coding`    | True              | Manifest generation                 |
| `extract`   | False             | Structured extraction, self-review  |
| `instruct`  | False             | Parsing, formatting, posting        |

Extraction nodes never fabricate — each `Finding` requires a verbatim quote tied to a source URL. There are no confidence scores.

---

## Prerequisites

All external services must be running before you start the agent.

| Service | Role | Notes |
|---------|------|-------|
| **llama.cpp** | Local LLM inference | Must serve `qwen3.6-a3b` on an OpenAI-compatible endpoint (default `http://localhost:8080/v1`) |
| **SearXNG** | Web search | Self-hosted, reachable over HTTP |
| **Gitea** | Git forge | Self-hosted; agent needs a PAT with repo read/write scope |
| **Renovate** | Dependency update bot | Must be configured to open PRs against your Gitea instance |
| **Flux** | GitOps runtime | Required if you use Graph B for HelmRelease-based deployments |

Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/) are required on the machine running the agent.

---

## Setup

```bash
# 1. Clone and enter the repo
git clone <your-gitea-url>/ops-agent.git
cd ops-agent

# 2. Install dependencies
uv sync

# 3. Configure
cp .env.example .env
$EDITOR .env   # fill in the required fields (see Environment variables below)
```

### Environment variables

Copy `.env.example` to `.env` and set:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLAMACPP_BASE_URL` | No | `http://localhost:8080/v1` | llama.cpp OpenAI-compatible endpoint |
| `MODEL_ALIAS` | No | `qwen3.6-a3b` | Model name registered in llama.cpp |
| `LLAMACPP_API_KEY` | No | `sk-no-auth` | Placeholder — llama.cpp ignores auth |
| `SEARXNG_URL` | **Yes** | — | Base URL of your SearXNG instance |
| `GITEA_BASE_URL` | **Yes** | — | Base URL of your Gitea instance (no trailing slash) |
| `GITEA_TOKEN` | **Yes** | — | Gitea PAT with repo read/write scope |
| `GIT_AUTHOR_NAME` | No | `ops-agent` | Git commit author name |
| `GIT_AUTHOR_EMAIL` | No | `ops-agent@homelab.local` | Git commit author email |
| `EXAMPLE_REPO_PATH` | No | — | Absolute path to a local repo Graph B reads for conventions |
| `REQUEST_TIMEOUT_SECONDS` | No | `120` | HTTP timeout for all outbound requests |

---

## Running

### Graph A — Review a Renovate PR

```bash
uv run ops-agent review-pr --owner <gitea-owner> --repo <repo-name> --pr <pr-number>
```

The agent will:
1. Fetch the PR diff and metadata from Gitea.
2. Search the web for changelogs and release notes.
3. Extract breaking-change findings (verbatim quotes only).
4. Assemble a verdict (`clear` / `breaking` / `needs_human`).
5. Post a markdown comment on the Gitea PR.

### Graph B — Scaffold a deployment

```bash
# Inline prompt
uv run ops-agent scaffold --prompt "Deploy Paperless-NGX with a PostgreSQL backend, port 8000"

# Or from a file
uv run ops-agent scaffold --prompt-file request.txt
```

The agent will:
1. Parse the request into a service spec.
2. Research the service (image, ports, env, volumes).
3. Check for an upstream Helm chart on Artifact Hub.
4. Read your `EXAMPLE_REPO_PATH` repo for Kustomize/Flux conventions.
5. Generate a `HelmRelease` + `values.yaml` (if a chart exists) or Kustomize manifests.
6. Self-review against your conventions; retry up to 2 times if issues are found.
7. Create a branch, commit the manifests, push, and open a Gitea PR.

---

## Tests

```bash
uv run pytest -q
```

Tests run without a live model or network — LLM calls and HTTP are mocked.

---

## Evals

Evals measure breaking-change extraction precision/recall against labeled past Renovate PRs.

```bash
uv run python -m evals.runner
```

On a fresh checkout this prints `no eval data yet` and exits 0. To add labeled examples, drop JSON files into `evals/data/` following the schema documented in [evals/README.md](evals/README.md).

---

## Development

```bash
# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/
```
