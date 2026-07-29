# Checkpointing Implementation Summary

## Overview

Added resilience to ops-agent graphs via PostgreSQL-backed checkpointing, allowing graphs to resume from the last completed node on failure.

## Files Changed

### Infrastructure

1. **docker-compose.yml** (new)
   - PostgreSQL 16 Alpine container for checkpoint storage
   - Volume for data persistence
   - Health checks for readiness verification
   - Port mapping: 5432 (internal) → exposed as 6432 (external)

2. **src/ops_agent/checkpointing.py** (new)
   - `get_checkpoint_saver()`: Cached connection to PostgreSQL via `langgraph.checkpoint.postgres.PostgresSaver`
   - Connection string built from environment variables
   - Graceful fallback if `CHECKPOINT_ENABLED=false`

3. **.env.example** (updated)
   - Added checkpoint configuration section:
     - `CHECKPOINT_ENABLED` (default: true)
     - `CHECKPOINT_DB_HOST`, `CHECKPOINT_DB_PORT`, `CHECKPOINT_DB_USER`, `CHECKPOINT_DB_PASSWORD`, `CHECKPOINT_DB_NAME`

4. **src/ops_agent/config.py** (updated)
   - Added checkpoint settings fields to `Settings` class
   - Allows configuration via environment variables

### State Management

5. **src/ops_agent/state.py** (updated)
   - `RenovateReviewState`: Added `thread_id` field (checkpoint key)
   - `DeployScaffoldState`: Added `thread_id`, `_owner`, `_repo`, `issue_number` fields

### Graph A: Renovate Review

6. **src/ops_agent/graphs/renovate_review.py** (updated)

   **Changes to `ingest_pr` node:**
   - Extracts last commit ID from PR's commits array
   - Falls back to `owner/repo#pr_index` if commits unavailable
   - Returns `thread_id` in state update
   - Uses commit ID as checkpoint key for resumption

   **Changes to `research` node:**
   - Wrapped in try/except to catch LLM/tool-calling failures
   - On failure: logs to stderr, returns empty evidence (allows graph to continue)
   - Prevents single failed tool call from crashing entire graph

   **Changes to `extract_breaking_changes` node:**
   - Wrapped in try/except to catch LLM extraction failures
   - On failure: logs to stderr, returns empty findings dict (allows graph to continue)

   **Changes to `build_renovate_graph` function:**
   - Imports `get_checkpoint_saver()`
   - Passes checkpointer to `graph.compile()` if available
   - Gracefully compiles without checkpointer if disabled

   **Changes to `run` function:**
   - Accepts optional `thread_id` parameter
   - Initializes state with `thread_id` (computed by ingest_pr or passed in)
   - Uses `config={"configurable": {"thread_id": ...}}` when invoking if thread_id is provided
   - Enables checkpoint resumption when same thread_id is re-run

### Graph B: Deployment Scaffolder

7. **src/ops_agent/graphs/scaffold_deploy.py** (updated)

   **Changes to `parse_request` node:**
   - Wrapped in try/except to catch parsing/LLM failures
   - On failure: logs to stderr, returns dummy spec (allows graph to continue)

   **Changes to `research_service` node:**
   - Wrapped in try/except to catch research/tool-calling failures
   - On failure: logs to stderr, returns empty evidence (allows graph to continue)

   **Changes to `assess_helm` node:**
   - Wrapped in try/except to catch helm assessment failures
   - On failure: logs to stderr, assumes no helm chart found (allows graph to continue)

   **Changes to `build_scaffold_graph` function:**
   - Imports `get_checkpoint_saver()`
   - Passes checkpointer to `graph.compile()` if available
   - Gracefully compiles without checkpointer if disabled

   **Changes to `run` function:**
   - Accepts `id`, `thread_id` parameters in addition to existing params
   - Determines checkpoint key from: `thread_id` > `issue_number` > `id` > fallback
   - Initializes state with `thread_id` and optional `_owner`, `_repo`, `issue_number`
   - Uses `config={"configurable": {"thread_id": ...}}` when invoking if thread_id is provided
   - Enables checkpoint resumption when same thread_id is re-run

### CLI

8. **src/ops_agent/cli.py** (updated)

   **Changes to `_cmd_scaffold` function:**
   - Extracts checkpoint `id` parameter from args
   - Passes `id`, `owner`, `repo` to `run()` function

   **Changes to scaffold argument parser:**
   - Added `--id` optional argument for explicit checkpoint ID
   - Used with `--prompt` or `--prompt-file` to enable resumption
   - Not required when using `--issue` (issue number is used as checkpoint key)

### Documentation

9. **docs/checkpointing.md** (new)
   - Comprehensive user guide for checkpointing setup and usage
   - Usage examples for both graphs
   - Failure handling explanation
   - Troubleshooting guide
   - Future enhancement ideas

## Checkpoint Keys Strategy

### Renovate PR Review
- **Key source**: Last commit ID from PR (extracted in `ingest_pr`)
- **Why**: Uniquely identifies a specific version of the PR; re-running the same PR uses the same key
- **Fallback**: `owner/repo#pr_index` if commit data unavailable

### Deployment Scaffolder
- **Key sources** (priority):
  1. Explicit `--id` parameter (most control)
  2. `--issue` number (automatic: `issue_{N}`)
  3. Explicit `thread_id` parameter
  4. Fallback: `scaffold_deploy_no_id` (not checkpointed)
- **Why**: Allows multiple independent scaffold runs alongside issue-based runs

## Failure Handling Strategy

Each node that performs I/O or LLM calls is wrapped in try/except:

| Node | Failure Mode | Recovery |
|------|--------------|----------|
| ingest_pr | Gitea API error | Raises RuntimeError (graph fails; checkpointed before this node) |
| research | Tool call / LLM timeout | Empty evidence; downstream proceeds conservatively |
| extract_breaking_changes | LLM extraction failure | Empty findings; verdict uses only reasoning and deterministic checks |
| assess_risk | LLM reasoning failure | No risk opinion; verdict doesn't apply reasoning layer |
| post_review | Gitea API error | Logs error to stderr; returns `posted: False` but state still checkpointed |
| parse_request | LLM parsing failure | Dummy spec; graph continues to research phase |
| research_service | Tool call / LLM timeout | Empty evidence; helm assessment still runs |
| assess_helm | LLM assessment failure | Assumes no helm chart; Kustomize path taken |
| load_conventions | Missing example repo | Returns generic conventions; manifest generation proceeds |
| generate_helmrelease / generate_kustomize | LLM generation failure | Returns empty manifests; graph continues to review |
| self_review | LLM review failure | Returns review passed=false; generator retries or commits |
| commit_and_pr | Git / Gitea API error | Returns error message in pr_url field; state is checkpointed |

**Key principle**: Failures are logged but don't crash the graph. Intermediate state is checkpointed, allowing manual inspection and retry.

## Resume Behavior

### When a Checkpoint Exists
- LangGraph detects the thread_id matches a saved checkpoint
- Loads the full state from the last completed node
- Resumes execution from the next node

### When No Checkpoint Exists
- Graph runs fresh from entry point with initialized state

### Manual Resumption
```bash
# After a failed run:
ops-agent review-pr --owner X --repo Y --pr 42
# (thread_id auto-computed as last commit ID; checkpoint found; resumes)

ops-agent scaffold --prompt "Deploy X" --id my-run-id
# (thread_id is "my-run-id"; checkpoint found; resumes)
```

## Backward Compatibility

- Graphs work unchanged if PostgreSQL is unavailable (checkpointing disabled)
- Existing code using `run()` functions still works (thread_id is optional)
- State TypedDict additions are backward compatible (new fields can be omitted on invoke)

## Testing Checklist

Before using checkpointing in production:

- [ ] PostgreSQL container starts and stays healthy
- [ ] First graph run completes and saves checkpoint
- [ ] Verify checkpoint stored in PostgreSQL:
  ```bash
  psql -h localhost -U ops_agent -d ops_agent_checkpoints \
    -c "SELECT thread_id, checkpoint FROM checkpoints;"
  ```
- [ ] Kill graph mid-run, re-run with same thread_id → resumes from checkpoint
- [ ] Modify env to disable checkpointing → graphs work without PostgreSQL
- [ ] Resume graph after fixing the issue that caused initial failure → completes successfully

## Environment Variables Summary

```env
# Enable/disable checkpointing
CHECKPOINT_ENABLED=true|false

# PostgreSQL connection
CHECKPOINT_DB_HOST=localhost
CHECKPOINT_DB_PORT=5432
CHECKPOINT_DB_USER=ops_agent
CHECKPOINT_DB_PASSWORD=ops_agent_dev_password
CHECKPOINT_DB_NAME=ops_agent_checkpoints
```

All are optional with sensible defaults (matching docker-compose.yml defaults).
