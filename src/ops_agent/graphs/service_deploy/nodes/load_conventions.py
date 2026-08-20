"""Node: load_conventions - analyzes target repo and extracts its conventions."""

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from ops_agent.config import get_settings
from ops_agent.graphs.service_deploy.nodes.commit_and_pr import _get_or_clone_repo
from ops_agent.llm.personas import get_llm
from ops_agent.prompting import build_agent_messages
from ops_agent.state import ServiceDeployState

logger = logging.getLogger(__name__)

_CONVENTIONS_SYSTEM = """\
You are a Kubernetes/GitOps conventions analyst. Given the contents of a Flux/Kustomize
repository, summarize the conventions used so a code generator can follow them. Focus on:
- Directory structure (apps/, infrastructure/, etc.)
- Namespace conventions
- Common labels and annotations
- HelmRelease structure if Flux is used
- Kustomization file patterns
- Naming patterns for secrets, configmaps, etc.

Be concise — this summary will be passed to a code generator.
"""

# Namespaces that exist but are almost never the right target for a new app.
_SYSTEM_NAMESPACES = {"default", "kube-system", "kube-public", "kube-node-lease"}

_NAMESPACE_FIELD_RE = re.compile(r"^\s*namespace:\s*[\"']?([a-z0-9][a-z0-9-]*)[\"']?\s*$", re.MULTILINE)


def _scan_namespaces(repo_dir: Path) -> list[str]:
    """Collect namespaces referenced across the repo's YAML manifests.

    Gathers names from `kind: Namespace` resources and from `namespace:` fields.
    Files with template syntax (Helm/Flux) that fail to parse fall back to a
    regex scan so we still catch plain `namespace:` references.
    """
    found: set[str] = set()

    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules",)]
        for fname in files:
            if not fname.endswith((".yaml", ".yml")):
                continue
            try:
                text = (Path(root) / fname).read_text(encoding="utf-8")
            except Exception:
                logger.debug("skipping unreadable file %s", Path(root) / fname)
                continue

            parsed = False
            try:
                for doc in yaml.safe_load_all(text):
                    parsed = True
                    if not isinstance(doc, dict):
                        continue
                    if doc.get("kind") == "Namespace":
                        name = (doc.get("metadata") or {}).get("name")
                        if isinstance(name, str):
                            found.add(name)
                    meta_ns = (doc.get("metadata") or {}).get("namespace")
                    if isinstance(meta_ns, str):
                        found.add(meta_ns)
            except yaml.YAMLError:
                parsed = False

            if not parsed:
                found.update(_NAMESPACE_FIELD_RE.findall(text))

    return sorted(ns for ns in found if ns and ns not in _SYSTEM_NAMESPACES)


def load_conventions(state: ServiceDeployState) -> dict[str, Any]:
    """Clone target repo and analyze its conventions."""
    settings = get_settings()

    owner: str = state.get("_owner", "")  # type: ignore[typeddict-item]
    repo_name: str = state.get("_repo", "")  # type: ignore[typeddict-item]

    if not owner or not repo_name:
        logger.warning("No _owner/_repo in state; using generic conventions")
        return {
            "conventions": "Use standard Kubernetes/Flux GitOps conventions: apps directory structure, HelmRelease CRDs, Kustomize overlays, and standard labeling patterns.",
            "existing_namespaces": [],
        }

    try:
        repo_dir = _get_or_clone_repo(
            owner,
            repo_name,
            Path(settings.repo_location),
            settings.github_base_url,
            settings.github_token,
        )
    except Exception as exc:
        logger.error("Failed to clone repo %s/%s: %s", owner, repo_name, exc)
        return {"conventions": f"Failed to clone repo: {exc}", "existing_namespaces": []}

    structure_lines: list[str] = []
    sample_files: dict[str, str] = {}

    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules",)]
        rel_root = Path(root).relative_to(repo_dir)
        depth = len(rel_root.parts)
        if depth > 3:
            continue
        indent = "  " * depth
        structure_lines.append(f"{indent}{rel_root}/")
        for fname in files:
            fpath = Path(root) / fname
            structure_lines.append(f"{indent}  {fname}")
            if fname.endswith((".yaml", ".yml")) and len(sample_files) < 6:
                try:
                    content = fpath.read_text(encoding="utf-8")
                    sample_files[str(fpath.relative_to(repo_dir))] = content[:1500]
                except Exception:
                    logger.debug("skipping unreadable sample file %s", fpath)

    structure = "\n".join(structure_lines[:120])
    samples_block = "\n\n".join(
        f"### {path}\n```yaml\n{content}\n```" for path, content in sample_files.items()
    )

    llm = get_llm("research")
    # Repo structure and sample YAML come from the target repo, which we treat
    # as untrusted data (a hostile repo could embed directives).
    messages = build_agent_messages(
        system=_CONVENTIONS_SYSTEM,
        untrusted_blocks=[
            ("repo_structure", structure),
            ("sample_files", samples_block),
        ],
        trusted_tail="Summarize the conventions the repo follows.",
    )

    response = llm.invoke(messages)
    conventions: str = response.content if hasattr(response, "content") else str(response)

    existing_namespaces = _scan_namespaces(repo_dir)
    logger.info(
        "Extracted conventions from %s/%s; found %d namespace(s): %s",
        owner,
        repo_name,
        len(existing_namespaces),
        existing_namespaces,
    )
    return {"conventions": conventions, "existing_namespaces": existing_namespaces}
