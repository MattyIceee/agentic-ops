# Graph Checkpointing and Resilience

This document describes how to set up and use checkpointing for the ops-agent graphs.

## Overview

Checkpointing allows graphs to resume from the last completed node if they fail mid-run. This is especially useful for long-running operations like the deployment scaffolder, which may fail during external API calls or LLM requests.

## Architecture

- **Checkpoint Storage**: PostgreSQL database (via `langgraph.checkpoint.postgres.PostgresSaver`)
- **Checkpoint Keys**: 
  - `renovate_review`: Last commit ID from the PR (or `owner/repo#pr` as fallback)
  - `scaffold_deploy`: Issue number (as `issue_{N}`), explicit `--id` parameter, or custom thread ID
- **Error Handling**: Nodes fail gracefully without crashing the entire graph; intermediate state is preserved for resumption

## Setup

### 1. Start PostgreSQL

```bash
docker-compose up -d postgres
```

This starts a PostgreSQL container at `localhost:5432` with:
- User: `ops_agent`
- Password: `ops_agent_dev_password`
- Database: `ops_agent_checkpoints`

The container will automatically create the database schema on first connection.

### 2. Configure Environment Variables

Copy the checkpoint settings from `.env.example` to `.env`:

```env
CHECKPOINT_ENABLED=true
CHECKPOINT_DB_HOST=localhost
CHECKPOINT_DB_PORT=5432
CHECKPOINT_DB_USER=ops_agent
CHECKPOINT_DB_PASSWORD=ops_agent_dev_password
CHECKPOINT_DB_NAME=ops_agent_checkpoints
```

If `CHECKPOINT_ENABLED=false`, checkpointing is disabled and the graphs will run without persistence.

### 3. Verify Connection

The checkpointing module will connect on first use. You can manually test the connection:

```python
from ops_agent.checkpointing import get_checkpoint_saver
saver = get_checkpoint_saver()
print(saver)  # Should print a PostgresSaver instance if connected
```

## Usage

### Renovate PR Review

**First run:**
```bash
ops-agent review-pr --owner myorg --repo myrepo --pr 42
```

The graph automatically extracts the last commit ID from the PR and uses it as the checkpoint key. If the graph fails partway through (e.g., during the research node), the checkpoint is saved.

**Resume from checkpoint:**
```bash
ops-agent review-pr --owner myorg --repo myrepo --pr 42
```

The CLI extracts the same commit ID, LangGraph detects an existing checkpoint, and resumes from the last completed node.

### Deployment Scaffold

**From an issue (automatic checkpoint key):**
```bash
ops-agent scaffold --issue 100 --owner myorg --repo myrepo --prompt "Deploy nginx"
```

The graph uses `issue_100` as the checkpoint key.

**From inline prompt with explicit checkpoint ID:**
```bash
ops-agent scaffold --prompt "Deploy nginx" --id my-deployment-v1
```

This allows resuming via the same ID later:
```bash
ops-agent scaffold --prompt "Deploy nginx" --id my-deployment-v1
```

**From prompt file with checkpoint ID:**
```bash
ops-agent scaffold --prompt-file deploy.txt --id my-deployment-v1
```

## Failure Handling

Nodes are designed to fail gracefully:

- **research**: LLM call fails → returns empty evidence; downstream nodes proceed with no evidence
- **extract_breaking_changes**: Extraction fails → returns no findings; downstream nodes proceed conservatively
- **assess_risk**: Risk judgment fails → returns no opinion; assemble_verdict doesn't use it
- **parse_request**: Parsing fails → returns dummy spec with "unknown" name; graph continues
- **research_service**: Research fails → returns empty evidence; helm assessment still runs

If a **terminal node** fails (like `post_review` or `commit_and_pr`), the error is logged to stderr but the graph state is still checkpointed. This allows manual intervention or retry.

## Resuming Failed Runs

If a graph run fails:

1. **Check logs**: stderr output indicates which node failed and why
2. **Fix the issue**: Address the root cause (e.g., network connectivity, API rate limits)
3. **Re-run with the same checkpoint key**: The graph resumes from the last completed node

### Example: Resuming a Failed Scaffold

First run fails during helm assessment:
```bash
$ ops-agent scaffold --prompt "Deploy myapp" --id my-app-v1
Warning: assess_helm failed: Connection timeout
^C
```

Fix the network issue, then re-run:
```bash
$ ops-agent scaffold --prompt "Deploy myapp" --id my-app-v1
# Graph resumes from load_conventions (the next node after parse_request, research_service, assess_helm)
```

## Checkpoint Data Structure

Checkpoints store the full state at each node:
- Input values passed to the node
- Output values returned by the node
- Timestamps and node metadata

This allows LangGraph to reconstruct the execution state and resume mid-run.

## Disabling Checkpointing

To run without checkpointing:

1. Set `CHECKPOINT_ENABLED=false` in `.env`
2. Re-run graphs as normal; no checkpoint data is read or written

This is useful for:
- Development/testing (avoid PostgreSQL dependency)
- One-shot runs where resumption is not needed
- Environments where PostgreSQL is not available

## Limitations and Notes

- **Schema migrations**: If LangGraph updates the checkpoint schema, you may need to reset the database. We recommend treating `ops_agent_checkpoints` as ephemeral during development.
- **Thread ID collision**: If you resume a graph with the same thread ID, it picks up from the last checkpoint. Ensure thread IDs are unique per logical run.
- **State size**: Very large state objects (e.g., multi-megabyte evidence blocks) will slow down checkpointing. Keep evidence summaries reasonable.

## Troubleshooting

### PostgreSQL connection refused

Check that the container is running:
```bash
docker-compose ps
```

If the container is not running:
```bash
docker-compose up -d postgres
```

### "No such table" errors

This typically means the PostgreSQL schema hasn't been created yet. The first graph run will auto-create it via `langgraph.checkpoint.postgres`. If the error persists:

1. Verify the PostgreSQL container is fully started (check `docker-compose logs postgres`)
2. Manually connect to verify the database exists:
   ```bash
   psql -h localhost -U ops_agent -d ops_agent_checkpoints -c "\dt"
   ```

### Checkpoint not being used

Verify that:
1. `CHECKPOINT_ENABLED=true` in your `.env`
2. The checkpoint key (thread ID) is the same on re-run
3. The PostgreSQL container is running and accessible

You can manually check if a checkpoint exists by querying PostgreSQL:
```bash
psql -h localhost -U ops_agent -d ops_agent_checkpoints \
  -c "SELECT thread_id, checkpoint FROM checkpoints LIMIT 5;"
```

## Future Enhancements

- **Automatic cleanup**: Expire old checkpoints after N days
- **Checkpoint inspection**: CLI command to list and inspect saved checkpoints
- **Partial replay**: Resume from specific nodes, not just the last completed one
