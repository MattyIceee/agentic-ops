# ops-agent

An agentic ops pipeline for a self-hosted homelab. Two LangGraph graphs share a common set of tools and personas to automate two recurring ops tasks:

1. **Renovate PR reviewer** — annotates dependency-bump PRs opened by Renovate with breaking-change findings extracted from changelogs and release notes, then posts a verdict comment on the GitHub PR.
2. **Deployment scaffolder** — takes a natural-language service request, researches the service, generates Flux `HelmRelease` or Kustomize manifests following your existing repo conventions, self-reviews them, and opens a GitHub PR.

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
    github.py           # GitHub REST client + LangChain tools
  graphs/
    renovate_review.py  # Graph A — PR annotation pipeline
    scaffold_deploy.py  # Graph B — manifest generation pipeline
  cli.py                # `ops-agent` entry point
```

**Personas are sampling parameters, not model aliases.** There is one model, configured via `LLM_MODEL` (or the legacy `MODEL_ALIAS`). A persona is a bundle of `temperature`, `top_p`, `top_k`, `presence_penalty`, and `enable_thinking` passed per-request:

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
| **LLM endpoint** | Model inference | Any OpenAI-compatible `/v1` server (llama.cpp, vLLM, Ollama, OpenRouter, Groq, ...); configured via `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` |
| **SearXNG** | Web search | Self-hosted, reachable over HTTP |
| **GitHub** | Git forge | github.com; agent needs a fine-grained PAT |
| **Renovate** | Dependency update bot | Must be configured to open PRs against your GitHub repos |
| **Flux** | GitOps runtime | Required if you use Graph B for HelmRelease-based deployments |

Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/) are required on the machine running the agent.

---

## Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/MattyIceee/agentic-ops.git
cd agentic-ops

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
| `LLM_PROVIDER` | No | `llamacpp` | Provider profile: `llamacpp` (sends top_k / thinking extensions) or `openai-compatible` (standard params only) |
| `LLM_BASE_URL` | No | `http://localhost:8080/v1`¹ | Base URL of the OpenAI-compatible endpoint (llama.cpp, vLLM, Ollama, OpenRouter, ...) |
| `LLM_API_KEY` | No | `sk-no-auth`¹ | API key — llama.cpp ignores auth, cloud providers need a real key |
| `LLM_MODEL` | No | `qwen3.6-a3b`¹ | Model name as served by the endpoint |

¹ Fall back to the legacy `LLAMACPP_BASE_URL` / `LLAMACPP_API_KEY` / `MODEL_ALIAS` variables when unset.
| `SEARXNG_URL` | **Yes** | `http://localhost:8081` | Base URL of your SearXNG instance (matches the docker-compose stack) |
| `GITHUB_API_BASE_URL` | No | `https://api.github.com/` | Base URL of the GitHub REST API |
| `GITHUB_BASE_URL` | No | `https://github.com` | Git clone host for pushing branches and opening PRs |
| `GITHUB_TOKEN` | **Yes** | — | GitHub fine-grained PAT with repo read/write scope |
| `GIT_AUTHOR_NAME` | No | `ops-agent` | Git commit author name |
| `GIT_AUTHOR_EMAIL` | No | `ops-agent@homelab.local` | Git commit author email |
| `EXAMPLE_REPO_PATH` | No | — | Absolute path to a local repo Graph B reads for conventions |
| `REQUEST_TIMEOUT_SECONDS` | No | `120` | HTTP timeout for all outbound requests |

### GitHub token

Create a **fine-grained personal access token** scoped to the repos the agent works on,
with these repository permissions:

| Permission | Access |
|---|---|
| Contents | Read and write |
| Pull requests | Read and write |
| Issues | Read and write |
| Metadata | Read |

Note that GitHub rejects (422) any attempt to approve a PR opened by the token's own
account. Graph A's auto-approve therefore works on Renovate PRs (authored by
`renovate[bot]`) but will warn and skip if the agent reviews a PR it opened itself.

---

## Running

### Graph A — Review a Renovate PR

```bash
uv run ops-agent review-pr --owner <github-owner> --repo <repo-name> --pr <pr-number>
```

The agent will:
1. Fetch the PR diff and metadata from GitHub.
2. Search the web for changelogs and release notes.
3. Extract breaking-change findings (verbatim quotes only).
4. Assemble a verdict (`clear` / `breaking` / `needs_human`).
5. Post a markdown comment on the GitHub PR.

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
7. Create a branch, commit the manifests, push, and open a GitHub PR.

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

## Langfuse Tracing (Local Development)

Langfuse v3 provides distributed tracing for graph executions. A full Docker stack is included for local testing.

### Start the stack

```bash
docker compose down -v  # Clean start
docker compose up -d
```

All services will be healthy in ~30 seconds:
- **Langfuse Web UI**: http://localhost:3000
- **MinIO S3 console**: http://localhost:9091
- **SearXNG**: http://localhost:8081

Point `SEARXNG_URL` at `http://localhost:8081` to use the bundled SearXNG
instance (the JSON API needed by the `web_search` tool is enabled).

### Default credentials

| Service | Username | Password |
|---------|----------|----------|
| Langfuse | `dev@localhost.com` | `changeme` |
| MinIO | `minioadmin` | `minioadmin` |

### Enable tracing in ops-agent

Update `.env`:
```
LANGFUSE_ENABLED=true
LANGFUSE_SECRET_KEY=sk-lf-dev-test-key-12345678901234567890ab
LANGFUSE_PUBLIC_KEY=pk-lf-dev-test-key-12345678901234567890ab
LANGFUSE_BASE_URL=http://localhost:3000
```

### Run a traced execution

```bash
uv run ops-agent review-pr --owner test --repo test --pr 1
```

Traces will appear in Langfuse Web UI with:
- Trace name: `update-review/test/test#1`
- Tags: `graph:update-review`, `has-findings` (if applicable)
- Metadata: dependency, version bump, verdict, finding count
- Spans: one per node, with LLM latency visible

---

## Development

```bash
# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/
```
