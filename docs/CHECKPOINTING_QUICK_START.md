# Checkpointing Quick Start

## 1. Start PostgreSQL

```bash
docker-compose up -d postgres
```

Wait a few seconds for the container to be ready:
```bash
docker-compose ps
# STATUS should show "healthy"
```

## 2. Configure .env

Copy these lines from `.env.example` to `.env` (or use defaults):

```env
CHECKPOINT_ENABLED=true
CHECKPOINT_DB_HOST=localhost
CHECKPOINT_DB_PORT=5432
CHECKPOINT_DB_USER=ops_agent
CHECKPOINT_DB_PASSWORD=ops_agent_dev_password
CHECKPOINT_DB_NAME=ops_agent_checkpoints
```

## 3. Test It Out

### Renovate PR Review (automatic checkpoint key)

```bash
# First run
ops-agent review-pr --owner myorg --repo myrepo --pr 42

# If it fails partway, re-run the exact same command
# It will resume from the last completed node
ops-agent review-pr --owner myorg --repo myrepo --pr 42
```

### Deployment Scaffolder with Issue (automatic checkpoint key)

```bash
# First run
ops-agent scaffold --issue 100 --owner myorg --repo myrepo

# Resume: same command automatically uses checkpoint
ops-agent scaffold --issue 100 --owner myorg --repo myrepo
```

### Deployment Scaffolder with Prompt (explicit checkpoint ID)

```bash
# First run with explicit ID
ops-agent scaffold --prompt "Deploy nginx" --id my-app-v1

# Resume: use the same --id
ops-agent scaffold --prompt "Deploy nginx" --id my-app-v1
```

## 4. Verify Checkpoints Are Saved

```bash
# Connect to PostgreSQL
psql -h localhost -U ops_agent -d ops_agent_checkpoints

# List all checkpoints
SELECT thread_id, COUNT(*) as checkpoint_count, MAX(ts) as latest FROM checkpoints GROUP BY thread_id;

# Inspect a specific checkpoint (replace thread_id)
SELECT * FROM checkpoints WHERE thread_id = 'your-thread-id' ORDER BY ts DESC LIMIT 1 \gx

# Exit
\q
```

## 5. Disable Checkpointing (Optional)

If you don't have PostgreSQL or want to run without checkpointing:

```env
CHECKPOINT_ENABLED=false
```

Graphs will work normally, just without resumption capability.

## Common Commands

| Task | Command |
|------|---------|
| Start PostgreSQL | `docker-compose up -d postgres` |
| Check PostgreSQL health | `docker-compose ps` |
| View PostgreSQL logs | `docker-compose logs postgres` |
| Stop PostgreSQL | `docker-compose down` |
| Reset checkpoint data | `docker-compose down -v` (deletes volume) |
| Review PR (resumable) | `ops-agent review-pr --owner X --repo Y --pr N` |
| Scaffold from issue (resumable) | `ops-agent scaffold --issue N --owner X --repo Y` |
| Scaffold from prompt (resumable) | `ops-agent scaffold --prompt "..." --id my-id` |
| Scaffold from file (resumable) | `ops-agent scaffold --prompt-file deploy.txt --id my-id` |

## What Gets Checkpointed?

- ✅ All node outputs and state at each step
- ✅ Evidence gathered from research
- ✅ LLM structured outputs (findings, verdicts, specs)
- ✅ Intermediate generated manifests
- ❌ Does NOT retry failed external calls automatically (you trigger the retry)

## On Failure

If a graph fails:

1. **Check stderr for error message** (printed to console)
2. **Fix the underlying issue** (network, API rate limits, etc.)
3. **Re-run with the same thread ID** → automatically resumes

Example:
```bash
$ ops-agent review-pr --owner X --repo Y --pr 42
# ... fails partway ...
$ # Fix network issue
$ ops-agent review-pr --owner X --repo Y --pr 42
# Resumes from where it left off!
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Connection refused` | PostgreSQL not running; run `docker-compose up -d postgres` |
| `No such table` | PostgreSQL just started; first graph run will auto-create schema |
| Checkpoint not being used | Check `CHECKPOINT_ENABLED=true` in `.env` |
| Want to reset everything | `docker-compose down -v && docker-compose up -d postgres` |

## For More Details

- Full setup guide: [checkpointing.md](checkpointing.md)
- Implementation details: [CHECKPOINTING_IMPLEMENTATION.md](CHECKPOINTING_IMPLEMENTATION.md)
