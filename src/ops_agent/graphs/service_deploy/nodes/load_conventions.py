"""Node: load_conventions - analyzes target repo and extracts its conventions."""

import logging
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ops_agent.config import get_settings
from ops_agent.graphs.service_deploy.nodes.commit_and_pr import _get_or_clone_repo
from ops_agent.llm.personas import get_llm
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


def load_conventions(state: ServiceDeployState) -> dict[str, Any]:
    """Clone target repo and analyze its conventions."""
    settings = get_settings()

    owner: str = state.get("_owner", "")  # type: ignore[typeddict-item]
    repo_name: str = state.get("_repo", "")  # type: ignore[typeddict-item]

    if not owner or not repo_name:
        logger.warning("No _owner/_repo in state; using generic conventions")
        return {
            "conventions": "Use standard Kubernetes/Flux GitOps conventions: apps directory structure, HelmRelease CRDs, Kustomize overlays, and standard labeling patterns."
        }

    try:
        repo_dir = _get_or_clone_repo(
            owner,
            repo_name,
            Path(settings.repo_location),
            settings.gitea_base_url,
            settings.gitea_token,
        )
    except Exception as exc:
        logger.error("Failed to clone repo %s/%s: %s", owner, repo_name, exc)
        return {"conventions": f"Failed to clone repo: {exc}"}

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
                    pass

    structure = "\n".join(structure_lines[:120])
    samples_block = "\n\n".join(
        f"### {path}\n```yaml\n{content}\n```" for path, content in sample_files.items()
    )

    llm = get_llm("research")
    messages = [
        SystemMessage(content=_CONVENTIONS_SYSTEM),
        HumanMessage(
            content=(
                f"Repository structure:\n{structure}\n\n"
                f"Sample files:\n{samples_block}\n\n"
                "Summarize the conventions."
            )
        ),
    ]

    response = llm.invoke(messages)
    conventions: str = response.content if hasattr(response, "content") else str(response)
    logger.info("Extracted conventions from %s/%s", owner, repo_name)
    return {"conventions": conventions}
