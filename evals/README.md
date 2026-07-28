# Evals — breaking-change extraction

Measures precision, recall, and F1 of Graph A's `extract_breaking_changes` node
against real, hand-labeled past Renovate PRs.

## Adding an example

Create a JSON file in `evals/data/` (one example per file, or a JSON array per file).
File names are arbitrary — use something descriptive, e.g. `lodash-4.17.21.json`.

### JSON schema

```json
{
  "pr_ref":              "owner/repo#42",
  "dependency":          "lodash",
  "old_version":         "4.17.20",
  "new_version":         "4.17.21",
  "has_breaking_change": false,
  "notes":               "Patch release; no API changes.",
  "evidence": [
    {
      "source": "changelog",
      "url":    "https://github.com/lodash/lodash/releases/tag/4.17.21",
      "text":   "Full text of the changelog entry you retrieved..."
    }
  ]
}
```

| Field                | Type            | Required | Description                                                  |
|----------------------|-----------------|----------|--------------------------------------------------------------|
| `pr_ref`             | string          | yes      | Human label for the PR, e.g. `"owner/repo#42"`               |
| `dependency`         | string          | yes      | Package name                                                 |
| `old_version`        | string          | yes      | Version before the bump                                      |
| `new_version`        | string          | yes      | Version after the bump                                       |
| `has_breaking_change`| bool            | yes      | Ground-truth label                                           |
| `notes`              | string          | no       | Commentary on why this label was assigned                    |
| `evidence`           | array of objects| no       | Pre-fetched evidence; without this the example is skipped    |

Each `evidence` object must have `source` (string) and `text` (string); `url` is optional.

**Important:** The evidence you store is what the model actually sees during eval —
copy the raw text you retrieved (changelog entry, release notes, issue body, etc.)
when you label the example. Without stored evidence, the runner skips the example
rather than making live network calls.

## Seed dataset

`evals/data/` ships with 7 hand-labeled examples drawn from real, well-documented
releases. The evidence text in each file is copied from the actual changelog /
advisory / migration guide (URLs included), so the set is a trustworthy baseline.

The split is deliberately balanced, and the non-breaking cases are chosen to be
false-positive traps (security patches and deprecation-warning releases whose text
is full of scary words like "deprecated" and "vulnerability" but which break no API):

| Example                      | Label        | Why it's here                                              |
|------------------------------|--------------|------------------------------------------------------------|
| `eslint-9.0.0`               | breaking     | Flat config default, Node bump, rules/formatters removed   |
| `express-5.0.0`              | breaking     | Many removed APIs + new path-matching syntax               |
| `axios-1.0.0`               | breaking     | Param serialization + interceptor config type changed      |
| `node-fetch-3.0.0`           | breaking     | Package is now ESM-only; CommonJS `require` breaks          |
| `lodash-4.17.21`             | non-breaking | Security patch (CVE-2021-23337 / CVE-2020-28500), no API change |
| `react-18.3.1`               | non-breaking | Adds deprecation *warnings* only; runtime unchanged        |
| `requests-2.31.0`            | non-breaking | Security fix (CVE-2023-32681); transparent for normal use  |

Grow the set by adding more files (see below). Good additions: minor releases that
*do* sneak in a breaking change, and majors that are actually safe for most users.

## Running evals

```sh
uv run python -m evals.runner
```

With no data files the runner prints `"no eval data yet"` and exits 0.

## Interpretation

- **Precision** — of all PRs the model flagged as breaking, how many actually were?
- **Recall** — of all truly breaking PRs, how many did the model catch?
- **F1** — harmonic mean; balance between precision and recall.

The positive class is `breaking`. A high recall is usually more important than high
precision for a safety-critical veto layer.
