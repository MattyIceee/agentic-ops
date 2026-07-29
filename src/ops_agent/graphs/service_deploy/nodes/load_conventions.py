"""Node: load_conventions - reads example repo and summarizes conventions."""

import os
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ops_agent.config import get_settings
from ops_agent.llm.personas import get_llm
from ops_agent.state import ServiceDeployState

_CONVENTIONS_SYSTEM = """\
You are a Kubernetes/GitOps conventions analyst. Given the contents of a homelab
Flux/Kustomize repository, summarize the conventions used so a code generator can
follow them. Focus on:
- Directory structure (apps/, infrastructure/, etc.)
- Namespace conventions
- Common labels and annotations
- HelmRelease structure if Flux is used
- Kustomization file patterns
- Naming patterns for secrets, configmaps, etc.

Be concise — this summary will be passed to a code generator.
"""


def load_conventions(state: ServiceDeployState) -> dict[str, Any]:
    """Read the example repo and summarize its layout conventions."""
    settings = get_settings()
    repo_path = settings.example_repo_path

    if not repo_path:
        return {"conventions": "No example repo configured. Use generic Kubernetes/Flux conventions."}

    repo_dir = Path(repo_path)
    if not repo_dir.exists():
        return {"conventions": f"Example repo not found at {repo_path}. Using generic conventions."}

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
        f"### {path}\n```yaml\n{content}\n```"
        for path, content in sample_files.items()
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
    return {"conventions": conventions}
