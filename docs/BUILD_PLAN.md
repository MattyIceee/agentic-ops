# Build Plan — Local-LLM Agentic Ops Pipeline

This is a step-by-step build plan meant to be executed by a cheaper coding model
(e.g. Sonnet/Haiku), one step at a time. Each step is a **self-contained prompt**:
paste the "Shared context" block once at the top of a session, then paste the
step's **Prompt** and let the model produce the files. Do the steps in order —
later steps import earlier ones.

Verify each step with its **Acceptance check** before moving on.

---

## Shared context (prepend to every step's prompt)

> You are implementing part of a Python project called **ops-agent**: an agent
> system that talks to a **locally hosted Qwen3.6-35B-A3B** model served by
> llama.cpp behind an OpenAI-compatible endpoint (default
> `http://localhost:8080/v1`, model alias `qwen3.6-a3b`). The model server is
> external and already running — we only build the orchestration layer.
>
> **Stack:** Python ≥3.11, dependency management with **uv** (never pip/poetry),
> `src/` layout under `src/ops_agent/`, **LangGraph** for orchestration,
> **LangChain** tool-calling agents for open-ended tool steps,
> `langchain_openai.ChatOpenAI` as the model client, **httpx** for HTTP.
>
> **Hard design constraints (do not violate):**
> 1. **No self-reported confidence scores from the model.** Model output is
>    either "grounded in cited evidence" or "needs human review" — nothing in
>    between. Never ask the model for a 0–100 confidence number.
> 2. **Personas are code, not server aliases.** There is ONE model alias. A
>    "persona" is just a bundle of sampling params (`temperature`, `top_p`,
>    `top_k`, `presence_penalty`) plus `chat_template_kwargs={"enable_thinking": bool}`
>    passed per-request. Never switch model aliases.
> 3. **Extraction steps** run `enable_thinking: False`, temperature 0.
>    **Research/planning steps** run `enable_thinking: True`, higher temperature.
> 4. **Tool-calling is required** — the agent must actually call tools, not just
>    emit text.
> 5. The self-hosted forge is **Gitea** (GitHub-shaped REST API but NOT
>    identical), authenticated with a personal access token. There is no GitHub.
> 6. Web search is **SearXNG**, self-hosted, reached over HTTP. No MCP server.
>
> Keep functions small and typed. Add docstrings. Do not invent extra
> dependencies beyond what the step names.

---

## Target repo layout (end state)

```
homelab-agentic-agent/
  pyproject.toml
  uv.lock
  .env.example
  README.md
  docs/BUILD_PLAN.md            # this file
  src/ops_agent/
    __init__.py
    config.py                   # pydantic-settings, loads .env
    state.py                    # shared TypedDict/pydantic state + Evidence/Finding
    llm/
      __init__.py
      personas.py               # ChatOpenAI factory + persona registry
    tools/
      __init__.py
      search.py                 # SearXNG search tool
      fetch.py                  # URL / changelog / issue fetch tool
      git_ops.py                # local git: diff, branch, commit, push
      gitea.py                  # Gitea REST client + PR tools
    graphs/
      __init__.py
      renovate_review.py        # Graph A
      scaffold_deploy.py        # Graph B
    cli.py                      # entry points for both graphs
  tests/
    __init__.py
    test_personas.py
    test_tools.py
    test_graphs.py
  evals/
    __init__.py
    README.md
    loader.py
    runner.py
    data/.gitkeep               # real labeled PRs go here later (empty for now)
```

---

## Step 1 — Project init + dependencies (uv)

**Prompt:**

> Initialize the uv project in the current directory (it already contains
> `README.md`, `.gitignore`, and `docs/`). Create `pyproject.toml` for a package
> named `ops-agent` (import package `ops_agent`, `src/` layout,
> `requires-python = ">=3.11"`). Add these runtime dependencies with `uv add`:
> `langgraph`, `langchain`, `langchain-openai`, `langchain-community`, `httpx`,
> `pydantic`, `pydantic-settings`, `python-dotenv`, `gitpython`, `tenacity`,
> `html2text`. Add dev dependencies with `uv add --dev`: `pytest`, `ruff`.
> Create empty `src/ops_agent/__init__.py`. Generate and commit `uv.lock`
> (`uv lock`). Configure a `[project.scripts]` entry `ops-agent = "ops_agent.cli:main"`.
> Add a minimal `[tool.ruff]` section (line-length 100).

**Acceptance check:** `uv sync` succeeds; `uv run python -c "import ops_agent"`
prints nothing and exits 0; `uv.lock` exists.

---

## Step 2 — Config + `.env.example`

**Prompt:**

> Create `src/ops_agent/config.py` using `pydantic-settings`. Define a `Settings`
> class (`BaseSettings`, reads from `.env`, env prefix none, case-insensitive)
> with fields:
> - `llamacpp_base_url: str = "http://localhost:8080/v1"`
> - `model_alias: str = "qwen3.6-a3b"`
> - `llamacpp_api_key: str = "sk-no-auth"`  (llama.cpp ignores it but the client needs a value)
> - `searxng_url: str` (required)
> - `gitea_base_url: str` (required, e.g. `https://gitea.homelab.local`)
> - `gitea_token: str` (required)
> - `git_author_name: str = "ops-agent"`
> - `git_author_email: str = "ops-agent@homelab.local"`
> - `example_repo_path: str | None = None`  (local path to the repo whose conventions Graph B studies)
> - `request_timeout_seconds: int = 120`
>
> Expose a cached `get_settings()` (`functools.lru_cache`). Then create
> `.env.example` documenting every field with placeholder values and a one-line
> comment each. Do NOT create a real `.env`.

**Acceptance check:** `uv run python -c "from ops_agent.config import get_settings"`
imports without error (it's fine if instantiating raises on missing required
vars — don't instantiate at import time).

---

## Step 3 — Persona / model client (`llm/personas.py`)

**Prompt:**

> Create `src/ops_agent/llm/personas.py`. Build a single base `ChatOpenAI` client
> pointed at the llama.cpp endpoint (`base_url=settings.llamacpp_base_url`,
> `model=settings.model_alias`, `api_key=settings.llamacpp_api_key`,
> `timeout=settings.request_timeout_seconds`).
>
> Define a `PERSONAS` registry (dict) with four entries — `research`, `coding`,
> `extract`, `instruct` — each a dataclass/`TypedDict` holding `temperature`,
> `top_p`, `top_k`, `presence_penalty`, and `enable_thinking: bool`:
> - `research`: temp 0.7, top_p 0.95, top_k 40, presence_penalty 0.0, thinking **True**
> - `coding`:   temp 0.25, top_p 0.9, top_k 40, presence_penalty 0.0, thinking **True**
> - `extract`:  temp 0.0, top_p 1.0, top_k 0, presence_penalty 0.0, thinking **False**
> - `instruct`: temp 0.3, top_p 0.9, top_k 40, presence_penalty 0.0, thinking **False**
>
> Expose `get_llm(persona: str) -> ChatOpenAI` that returns the base client bound
> with that persona's params. Pass `temperature`/`top_p` as normal kwargs, and
> pass `top_k`, `presence_penalty`, and
> `chat_template_kwargs={"enable_thinking": ...}` via `extra_body` (llama.cpp
> reads non-OpenAI params from the request body). Adding a new persona must be a
> one-line dict addition — keep it that trivial. Include a module docstring
> explaining the "personas are params, not aliases" rule.

**Acceptance check:** `uv run python -c "from ops_agent.llm.personas import get_llm, PERSONAS; assert set(PERSONAS)=={'research','coding','extract','instruct'}; get_llm('extract')"`
runs without error (no network call is made just constructing the client).

---

## Step 4 — Shared state (`state.py`)

**Prompt:**

> Create `src/ops_agent/state.py`. Define pydantic models:
> - `EvidenceItem`: `source: str` (human label, e.g. "changelog"), `url: str | None`, `text: str`
> - `Finding`: `claim: str`, `source: str`, `quote: str`  (verbatim quote from evidence; no confidence field)
> - `Verdict`: `decision: Literal["clear","breaking","needs_human"]`, `findings: list[Finding]`, `summary: str`
>
> Then define two LangGraph state TypedDicts (use `typing_extensions.TypedDict`
> with `Annotated` list reducers via `operator.add` for accumulating lists):
> - `RenovateReviewState`: `pr_index: int`, `dependency: str`, `current_version: str`,
>   `new_version: str`, `diff: str`, `renovate_rating: str | None`,
>   `evidence: Annotated[list[EvidenceItem], add]`, `verdict: Verdict | None`,
>   `posted: bool`
> - `DeployScaffoldState`: `request: str`, `provided_links: list[str]`,
>   `spec: dict`, `service_evidence: Annotated[list[EvidenceItem], add]`,
>   `helm_chart_found: bool`, `helm_chart_ref: str | None`, `conventions: str`,
>   `manifests: dict[str,str]` (filename→content), `review_passed: bool`,
>   `review_issues: list[str]`, `retry_count: int`, `pr_url: str | None`
>
> Everything importable from `ops_agent.state`.

**Acceptance check:** `uv run python -c "from ops_agent.state import RenovateReviewState, DeployScaffoldState, Finding, Verdict, EvidenceItem"` exits 0.

---

## Step 5 — SearXNG search tool (`tools/search.py`)

**Prompt:**

> Create `src/ops_agent/tools/search.py`. Using
> `langchain_community.utilities.SearxSearchWrapper` (host from
> `settings.searxng_url`), expose a LangChain tool `web_search` (via the `@tool`
> decorator or `Tool`) that takes a query string and returns the top N result
> snippets as a formatted string including each result's title, url, and snippet.
> Provide a factory `get_search_tool()` that reads settings lazily (do not create
> the wrapper at import time, so missing config doesn't break imports). Handle the
> case where SearXNG returns no results (return a clear "no results" string, not
> an exception).

**Acceptance check:** `uv run python -c "from ops_agent.tools.search import get_search_tool"`
imports without error. (Live query is a manual test once SearXNG is reachable.)

---

## Step 6 — Fetch tool (`tools/fetch.py`)

**Prompt:**

> Create `src/ops_agent/tools/fetch.py`. Expose a LangChain tool `fetch_url` that
> GETs a URL with `httpx` (follow redirects, timeout from settings, sane
> User-Agent) and returns readable text: if the response is HTML, convert to
> markdown/plain text with `html2text` and truncate to ~8000 chars; if it's
> already text/markdown/json, return as-is (truncated). Return errors as strings,
> not exceptions. This tool is used to pull changelogs, release notes, issue
> pages, READMEs, and docker-compose files. Provide `get_fetch_tool()` factory.

**Acceptance check:** import succeeds; `uv run python -c "from ops_agent.tools.fetch import get_fetch_tool"` exits 0.

---

## Step 7 — Local git tools (`tools/git_ops.py`)

**Prompt:**

> Create `src/ops_agent/tools/git_ops.py` using `GitPython`. Implement functions
> (and wrap each as a LangChain `@tool`) operating on a repo path:
> - `read_diff(repo_path: str, ref_a: str = "HEAD~1", ref_b: str = "HEAD") -> str`
> - `create_branch(repo_path: str, branch: str) -> str`
> - `commit_all(repo_path: str, message: str) -> str`  (stage all, commit with configured author)
> - `push_branch(repo_path: str, branch: str, remote: str = "origin") -> str`
>
> Use `git_author_name`/`git_author_email` from settings for commits. Return
> human-readable status strings; catch git errors and return them as strings.
> Provide a `get_git_tools() -> list` factory returning the tool objects.

**Acceptance check:** import succeeds; unit-testable against a temp repo (see Step 12).

---

## Step 8 — Gitea client + PR tools (`tools/gitea.py`)

**Prompt:**

> Create `src/ops_agent/tools/gitea.py`. Implement a `GiteaClient` class wrapping
> `httpx.Client` with base URL `settings.gitea_base_url + "/api/v1"` and header
> `Authorization: token <gitea_token>`. **Gitea's API is GitHub-shaped but not
> identical — use Gitea's own endpoints:**
> - `get_pr(owner, repo, index) -> dict`  → `GET /repos/{owner}/{repo}/pulls/{index}`
> - `get_pr_diff(owner, repo, index) -> str`  → `GET /repos/{owner}/{repo}/pulls/{index}.diff`
> - `post_issue_comment(owner, repo, index, body) -> dict`  → `POST /repos/{owner}/{repo}/issues/{index}/comments`
>   (Gitea PRs are issues; use the issue-comment endpoint for the review note.)
> - `create_pr(owner, repo, title, body, head, base) -> dict`  → `POST /repos/{owner}/{repo}/pulls`
> - `get_file(owner, repo, path, ref="main") -> str`  → `GET /repos/{owner}/{repo}/contents/{path}` (decode base64 `content`)
>
> Use `tenacity` retry on 5xx/timeout. Then expose LangChain tools wrapping the
> comment/create-PR/get-PR operations, plus a `get_gitea_tools()` factory. Keep
> the raw client usable directly by graph nodes (not everything needs to be a tool).

**Acceptance check:** import succeeds; `uv run python -c "from ops_agent.tools.gitea import GiteaClient, get_gitea_tools"` exits 0.

---

## Step 9 — Graph A: Renovate PR reviewer (`graphs/renovate_review.py`)

**Prompt:**

> Create `src/ops_agent/graphs/renovate_review.py` — a LangGraph `StateGraph`
> over `RenovateReviewState`. Nodes:
> 1. `ingest_pr` (persona `instruct`, direct GiteaClient call — not an agent):
>    read PR metadata + `.diff`, parse dependency name and old→new version, read
>    any Renovate "Merge Confidence" rating from the PR body if present. Fill state.
> 2. `research` (persona `research`, **LangChain tool-calling agent** built with
>    `create_tool_calling_agent` + `AgentExecutor`, tools = `[web_search, fetch_url]`):
>    gather evidence about the version bump (changelog, release notes, issues,
>    migration guides). Append `EvidenceItem`s to `state["evidence"]`, each with a
>    real `url`/`source`.
> 3. `extract_breaking_changes` (persona `extract`, `enable_thinking False`,
>    `with_structured_output(list[Finding])`, NO tools): over the gathered
>    evidence ONLY, emit findings where each has a verbatim `quote` tied to a
>    `source`. Drop any claim that has no supporting quote. Never fabricate.
> 4. `assemble_verdict` (plain Python, no LLM): if any finding exists →
>    `decision="breaking"`; if evidence list is empty/ambiguous → `"needs_human"`;
>    else `"clear"`. Build a `Verdict`.
> 5. `post_review` (direct GiteaClient call): post a markdown comment on the PR —
>    for `breaking`, list each finding with its source + quote; for `clear`, a
>    short "no breaking changes found, good to update" note; for `needs_human`, a
>    "insufficient evidence, please review" note. Set `posted=True`.
>
> Edges: `ingest_pr → research → extract_breaking_changes → assemble_verdict →
> post_review → END`, linear. Expose `build_renovate_graph()` returning the
> compiled graph, and a `run(pr_index, owner, repo)` helper. **Node bodies may be
> thin/stubbed for LLM prompt wording, but all tool/graph wiring must be real and
> the graph must compile and be invokable end-to-end with mocked LLM/tools.**

**Acceptance check:** `uv run python -c "from ops_agent.graphs.renovate_review import build_renovate_graph; build_renovate_graph()"`
compiles the graph without error.

---

## Step 10 — Graph B: deployment scaffolder (`graphs/scaffold_deploy.py`)

**Prompt:**

> Create `src/ops_agent/graphs/scaffold_deploy.py` — a LangGraph `StateGraph` over
> `DeployScaffoldState`. Nodes:
> 1. `parse_request` (persona `instruct`, structured output): turn the prompt into
>    a `spec` dict (name, image?, namespace, ports, env, volumes) and extract
>    `provided_links` (service site / repo URLs).
> 2. `research_service` (persona `research`, **tool-calling agent**, tools =
>    `[web_search, fetch_url]`): the docker command / compose yaml usually is NOT
>    in the prompt — follow the provided links and search to find the image, ports,
>    env, volumes, and any compose/install docs. Append `service_evidence`.
> 3. `assess_helm` (persona `research`, **tool-calling agent**, tools =
>    `[web_search, fetch_url]`): determine whether a decent upstream Helm chart
>    exists (search Artifact Hub / the project repo). Set `helm_chart_found: bool`
>    and `helm_chart_ref` (+ record source in evidence).
> 4. `load_conventions` (persona `research`; read `settings.example_repo_path` from
>    disk via file reads / git tools): summarize this repo's Kustomize/Flux layout
>    conventions into `state["conventions"]`.
> 5. **conditional edge on `helm_chart_found`:**
>    - True  → `generate_helmrelease` (persona `coding`): write a Flux
>      `HelmRelease` + `values.yaml` following conventions into `manifests`.
>    - False → `generate_kustomize` (persona `coding`): write Kustomize
>      Deployment/Service/etc into `manifests`.
> 6. `self_review` (persona `extract`, structured output `{passed: bool, issues: list[str]}`):
>    check the generated manifests against conventions + a checklist.
> 7. **conditional edge on `self_review`:** `passed False` and
>    `retry_count < MAX_RETRIES (2)` → loop back to the matching generate node with
>    `review_issues` as feedback, incrementing `retry_count`; else → `commit_and_pr`.
> 8. `commit_and_pr` (git tools + GiteaClient): create a branch, write manifest
>    files to the repo working tree, commit, push, open a Gitea PR; set `pr_url`.
>
> Edges: `parse_request → research_service → assess_helm → load_conventions →
> (route) → generate_* → self_review → (route) → commit_and_pr → END`. Expose
> `build_scaffold_graph()` and a `run(prompt)` helper. Same rule as Graph A: node
> LLM bodies can be thin, but all wiring/routing must be real and compile.

**Acceptance check:** `uv run python -c "from ops_agent.graphs.scaffold_deploy import build_scaffold_graph; build_scaffold_graph()"`
compiles without error; the conditional/loop edges are present.

---

## Step 11 — CLI entry points (`cli.py`)

**Prompt:**

> Create `src/ops_agent/cli.py` with an `argparse` CLI exposing two subcommands:
> - `review-pr --owner O --repo R --pr N` → runs Graph A's `run(...)`.
> - `scaffold --prompt "..."` (or `--prompt-file path`) → runs Graph B's `run(...)`.
> Provide `main()` wired to the `ops-agent` script entry point from Step 1. Print
> the resulting verdict / PR url as readable output. Fail with a clear message if
> required settings are missing.

**Acceptance check:** `uv run ops-agent --help` shows both subcommands.

---

## Step 12 — Tests scaffold (`tests/`)

**Prompt:**

> Create a `tests/` package with pytest tests that run without network or a live
> model:
> - `test_personas.py`: assert the four personas exist, that `extract` has
>   `enable_thinking False` + temperature 0, that `research` has thinking True, and
>   that `get_llm` returns a client (no network).
> - `test_tools.py`: test `git_ops` against a temp git repo (create, commit, read
>   diff) using `tmp_path`; test that tool factories import and return tools; mock
>   `httpx` for one `GiteaClient` method (e.g. `get_pr`) and assert the right URL +
>   auth header are used.
> - `test_graphs.py`: build both graphs and assert they compile; invoke each with
>   monkeypatched node functions / mocked `get_llm` and tools to confirm the graph
>   traverses end-to-end (including Graph B's helm-true and helm-false branches and
>   the self-review retry loop).
> Use `monkeypatch`/`unittest.mock`; do NOT hit the network or the model.

**Acceptance check:** `uv run pytest -q` passes.

---

## Step 13 — Evals scaffold (`evals/`) — NO fake data

**Prompt:**

> Create an `evals/` package for measuring breaking-change extraction precision/
> recall on **real, labeled past Renovate PRs** (to be added later — do NOT invent
> data now):
> - `evals/data/.gitkeep` and an empty `evals/data/` directory. Labeled examples
>   will be JSON files here later.
> - `evals/loader.py`: define an `EvalExample` schema (pr reference, dependency,
>   old/new version, `has_breaking_change: bool` ground truth, notes) and a
>   `load_examples(path="evals/data") -> list[EvalExample]` that reads the JSON
>   files (returns empty list if none — no crash).
> - `evals/runner.py`: a runner that, for each example, runs the extraction step
>   (Graph A's `extract_breaking_changes` in isolation, or the whole graph with
>   fetch/search mocked to the example's stored evidence) and computes
>   precision/recall/F1 of `breaking vs not-breaking` against ground truth. Print a
>   summary table. If there are zero examples, print a clear "no eval data yet"
>   message and exit 0.
> - `evals/README.md`: explain the expected JSON schema of a labeled example, that
>   real past PRs must be added by hand, and how to run `uv run python -m evals.runner`.

**Acceptance check:** `uv run python -m evals.runner` prints "no eval data yet"
and exits 0; `uv run python -c "from evals.loader import load_examples; assert load_examples()==[]"` passes.

---

## Step 14 — README

**Prompt:**

> Rewrite `README.md` to cover: what the project is (the two responsibilities),
> the architecture (two LangGraph graphs sharing personas + tools), prerequisites
> (external llama.cpp with alias `qwen3.6-a3b`, a reachable SearXNG instance, a
> Gitea instance + PAT, Renovate configured against Gitea, Flux for deployment),
> setup (`uv sync`, copy `.env.example` → `.env`), how to run each flow via the
> CLI, how to run tests and evals, and a note that Renovate's Merge Confidence
> rating is the primary auto-merge gate while this agent is a veto/annotation
> layer on top. Keep it practical.

**Acceptance check:** README documents both flows, the env vars, and the CLI
commands.

---

## Step 15 (LAST) — Dockerfile

Only after Steps 1–14 run locally.

**Prompt:**

> Create a multi-stage, uv-based `Dockerfile`: a builder stage that installs
> dependencies with `uv sync --frozen --no-dev` into a venv, and a slim runtime
> stage that copies the venv + `src/` and sets the `ops-agent` entrypoint. llama.cpp,
> SearXNG, and Gitea are all EXTERNAL — do not add them. Add a `docker-compose.yml`
> that runs ONLY the agent service, reading config from `.env`, with the three
> external services documented as network prerequisites (not services). Add
> `.dockerignore`.

**Acceptance check:** `docker build` succeeds; `docker run --rm IMAGE --help`
prints the CLI help.

---

## Notes for the driver (you)

- Run steps in order; each acceptance check gates the next.
- If a cheaper model drifts from the design constraints (e.g. adds a confidence
  score, switches model aliases, or reaches for a GitHub library), reject and
  re-paste the Shared context block — those are the load-bearing rules.
- Steps 5–8 (tools) are independent of each other and can be done in any order
  after Step 4.
- Keep `uv.lock` committed after Step 1; re-run `uv lock` only if deps change.
