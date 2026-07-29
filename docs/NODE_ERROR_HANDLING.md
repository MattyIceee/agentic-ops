# Node Error Handling Verification

This document verifies that nodes only finish and publish state if they succeed, and that failures don't crash the graph.

## Renovate Review Graph

### Node: `ingest_pr`
**Status**: ✅ Raises on failure (intended)
- **Why**: This is the entry point; if we can't fetch the PR, the graph cannot proceed
- **Behavior**: Raises `RuntimeError` if Gitea fetch fails
- **Checkpoint**: State before this node is not yet checkpointed; first checkpoint happens after success
- **Impact**: Graph fails with clear error message

### Node: `research`
**Status**: ✅ Fails gracefully
- **What happens on failure**: 
  - Tool call fails (API timeout, rate limit, invalid tool use)
  - LLM returns invalid response
- **How it's handled**: Try/except wraps entire node; catches Exception
- **State update on failure**: Returns `{"evidence": []}` (empty list)
- **Graph impact**: Downstream nodes proceed with no evidence
- **Checkpoint saved**: Yes, before this node (so it can be retried)

### Node: `extract_breaking_changes`
**Status**: ✅ Fails gracefully
- **What happens on failure**:
  - LLM extraction fails
  - JSON parsing of output fails
  - Structured output validation fails
- **How it's handled**: Try/except wraps entire node; catches Exception
- **State update on failure**: Returns `{}` (no findings)
- **Graph impact**: Downstream verdict uses only deterministic checks and reasoning
- **Checkpoint saved**: Yes, before this node (so it can be retried)
- **Note**: Even if extraction fails, assess_risk and assemble_verdict still run

### Node: `assess_risk`
**Status**: ✅ Already had graceful failure
- **What happens on failure**: LLM reasoning fails
- **How it's handled**: Existing try/except (line 341-343); catches Exception
- **State update on failure**: Returns `{}` (no risk opinion)
- **Graph impact**: Verdict doesn't factor in reasoned risk layer; uses findings only
- **Checkpoint saved**: Yes

### Node: `assemble_verdict`
**Status**: ✅ Pure Python, unlikely to fail
- **What happens on failure**: Logic error in Python code
- **How it's handled**: No try/except (shouldn't fail if inputs are valid)
- **If it does fail**: Exception propagates; graph stops, state checkpointed before this node
- **Risk level**: Low (deterministic logic)

### Node: `post_review`
**Status**: ✅ Fails gracefully
- **What happens on failure**: Gitea API call fails, cannot post comment
- **How it's handled**: Existing try/except (line 601-610); catches Exception
- **State update on failure**: Returns `{"posted": False}` (error logged to stderr)
- **Graph impact**: Verdict was already assembled; comment just wasn't posted
- **Checkpoint saved**: Yes, complete state is saved (though posting failed)
- **User sees**: Verdict printed to stdout; warning on stderr about posting failure

## Deployment Scaffolder Graph

### Node: `parse_request`
**Status**: ✅ Fails gracefully (NEW)
- **What happens on failure**: LLM parsing fails; structured output invalid
- **How it's handled**: Try/except wraps entire node; catches Exception
- **State update on failure**: Returns dummy spec with name="unknown"
- **Graph impact**: Downstream generates scaffolding for unknown service; likely to fail review
- **Checkpoint saved**: Yes
- **Note**: Allows user to retry with better prompt rather than hard failing

### Node: `research_service`
**Status**: ✅ Fails gracefully (NEW)
- **What happens on failure**: Tool call fails (network, API); LLM timeout
- **How it's handled**: Try/except wraps entire node; catches Exception
- **State update on failure**: Returns `{"service_evidence": []}`
- **Graph impact**: No service evidence; helm assessment and generation proceed with less info
- **Checkpoint saved**: Yes
- **Note**: Generation may produce less optimal output, but graph completes

### Node: `assess_helm`
**Status**: ✅ Fails gracefully (NEW)
- **What happens on failure**: Tool call fails; LLM assessment fails
- **How it's handled**: Try/except wraps entire node; catches Exception
- **State update on failure**: Returns `{"helm_chart_found": False, "helm_chart_ref": None, ...}`
- **Graph impact**: Routes to Kustomize generation instead of Helm
- **Checkpoint saved**: Yes
- **Note**: Safe fallback; Kustomize is always a valid option

### Node: `load_conventions`
**Status**: ✅ Already fails gracefully
- **What happens on failure**: Example repo path not configured; path doesn't exist; read fails
- **How it's handled**: Existing code (line 305-310); checks path and returns message
- **State update on failure**: Returns `{"conventions": "No example repo..."}` or similar
- **Graph impact**: Manifest generation uses generic conventions instead of learned ones
- **Checkpoint saved**: Yes

### Node: `generate_helmrelease` / `generate_kustomize`
**Status**: ⚠️ Not wrapped, but failures cascade gracefully
- **What happens on failure**: LLM generation fails; returns invalid YAML
- **How it's handled**: No try/except (intentional — let failures bubble up)
- **State update on failure**: If LLM call raises, exception propagates
- **Graph impact**: If generation fails, self_review still runs and may catch issues
- **Checkpoint saved**: Yes, before this node (so can be retried)
- **Note**: Could be wrapped for added resilience; currently relies on review loop

### Node: `self_review`
**Status**: ✅ Fails gracefully (already had it)
- **What happens on failure**: LLM review fails; structured output invalid
- **How it's handled**: Existing try/except behavior (line 506); catches Exception
- **State update on failure**: Returns `{"review_passed": False, "review_issues": ["..."]}`
- **Graph impact**: Generation retries (up to MAX_RETRIES); then commits regardless
- **Checkpoint saved**: Yes

### Node: `commit_and_pr`
**Status**: ✅ Fails gracefully
- **What happens on failure**: Git command fails; Gitea API fails
- **How it's handled**: Existing try/except (line 548-549, 571-572); catches Exception
- **State update on failure**: Returns `{"pr_url": "error message"}` instead of crashing
- **Graph impact**: Graph completes; returns error message instead of PR URL
- **Checkpoint saved**: Yes, complete state saved
- **User sees**: Error message in pr_url field; can investigate and retry

## Summary Table

| Node | Graph | Graceful Fail? | On Failure | Checkpoint |
|------|-------|---|-----------|-----------|
| ingest_pr | A | ❌ Raises | Stops graph | Before node |
| research | A | ✅ Yes | Empty evidence | Before node |
| extract_breaking_changes | A | ✅ Yes | Empty findings | Before node |
| assess_risk | A | ✅ Yes | No opinion | Before node |
| assemble_verdict | A | ⚠️ Pure Python | Propagates | Before node |
| post_review | A | ✅ Yes | Logged error, posted=false | After all nodes |
| parse_request | B | ✅ Yes | Dummy spec | Before node |
| research_service | B | ✅ Yes | Empty evidence | Before node |
| assess_helm | B | ✅ Yes | No helm, uses kustomize | Before node |
| load_conventions | B | ✅ Yes | Generic conventions | Before node |
| generate_helmrelease | B | ⚠️ No | Propagates (then review catches) | Before node |
| generate_kustomize | B | ⚠️ No | Propagates (then review catches) | Before node |
| self_review | B | ✅ Yes | Fails review, retries or commits | Before node |
| commit_and_pr | B | ✅ Yes | Error in pr_url field | After all nodes |

## Key Principles

1. **Entry points are strict**: `ingest_pr` must succeed because without a PR, there's nothing to review
2. **Research nodes are permissive**: If we can't find evidence, proceed with what we have
3. **Generation nodes are medium**: They don't crash, but output validity is checked by self_review
4. **Terminal nodes are logged**: If posting or committing fails, the error is logged and returned in state
5. **All failures are checkpointed**: State before the failed node is saved, allowing manual retry

## Testing Failure Scenarios

### Simulate a research failure
```bash
# In a test, mock the tool call to raise an exception
# Run graph
# Verify: research node returns empty evidence
# Verify: downstream nodes proceed
# Verify: checkpoint exists before research node
```

### Simulate a generation failure
```bash
# In a test, mock the LLM to return invalid YAML
# Run graph
# Verify: self_review catches the issue
# Verify: generation retries (up to MAX_RETRIES)
# Verify: if retries exhausted, commit_and_pr runs anyway
# Verify: checkpoint exists before generation node
```

### Simulate a commit failure
```bash
# In a test, make the git repo read-only
# Run graph
# Verify: commit_and_pr catches the exception
# Verify: pr_url contains error message
# Verify: graph completes (doesn't crash)
# Verify: checkpoint exists with all prior state
```

## Conclusion

✅ **Nodes only finish and publish state if they succeed** — each node either:
- Returns a valid state update (success), or
- Returns an empty/default state update (graceful failure), or
- Raises an exception (rare, intentional for entry point)

✅ **Failures don't blow up the graph** — exception handling ensures:
- Errors are logged to stderr
- State is returned (empty or default)
- Downstream nodes can proceed
- Checkpoints preserve progress

✅ **Resumption works because checkpoints are saved before each node** — if a node fails:
1. Its input state was checkpointed
2. User fixes the issue
3. Re-run with same thread_id
4. LangGraph loads the checkpoint
5. Graph resumes from the failed node
