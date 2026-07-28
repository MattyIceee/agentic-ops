"""Tests for both LangGraph graphs — no network, no live model.

Strategy: monkeypatch every node that calls an LLM or external service with a
stub lambda/function BEFORE calling build_*_graph(). Because build functions
read node callables from the module's global namespace at call time, patching
the module attributes before the build call makes the compiled graph capture
the stubs.

assemble_verdict (Graph A) is pure Python, so it runs for real.
_route_after_conventions and _route_after_review (Graph B) are pure Python
routing functions — they also run for real, letting us verify the conditional
and retry edges.
"""

from __future__ import annotations

import pytest

from ops_agent.state import EvidenceItem, Finding, Verdict

# ─────────────────────────── compile tests ─────────────────────────────────


def test_renovate_graph_compiles():
    from ops_agent.graphs.renovate_review import build_renovate_graph

    graph = build_renovate_graph()
    assert graph is not None


def test_scaffold_graph_compiles():
    from ops_agent.graphs.scaffold_deploy import build_scaffold_graph

    graph = build_scaffold_graph()
    assert graph is not None


# ─────────────────────────── Graph A helpers ───────────────────────────────

_RENOVATE_INITIAL: dict = {
    "pr_index": 1,
    "_owner": "myorg",
    "_repo": "myrepo",
    "dependency": "",
    "current_version": "",
    "new_version": "",
    "diff": "",
    "renovate_rating": None,
    "evidence": [],
    "verdict": None,
    "posted": False,
}

# ─────────────────────────── Graph A tests ─────────────────────────────────


def test_renovate_graph_clear_path(monkeypatch):
    """All external-calling nodes are stubbed; assemble_verdict runs for real."""
    import ops_agent.graphs.renovate_review as rr

    monkeypatch.setattr(rr, "ingest_pr", lambda s: {
        "dependency": "requests",
        "current_version": "2.28.0",
        "new_version": "2.32.0",
        "diff": "mock diff",
        "renovate_rating": "high",
    })
    monkeypatch.setattr(rr, "research", lambda s: {
        "evidence": [EvidenceItem(source="changelog", url="https://example.com", text="no breaking changes")],
    })
    # Return empty dict → _findings absent → assemble_verdict defaults to []
    monkeypatch.setattr(rr, "extract_breaking_changes", lambda s: {})
    monkeypatch.setattr(rr, "post_review", lambda s: {"posted": True})

    graph = rr.build_renovate_graph()
    result = graph.invoke(dict(_RENOVATE_INITIAL))

    assert result["posted"] is True
    assert isinstance(result["verdict"], Verdict)
    assert result["verdict"].decision == "clear"


def test_renovate_graph_breaking_path(monkeypatch):
    """Stub assemble_verdict to produce a 'breaking' verdict and verify post_review runs."""
    import ops_agent.graphs.renovate_review as rr

    finding = Finding(claim="API removed", source="changelog", quote="removed foo() in v2")

    monkeypatch.setattr(rr, "ingest_pr", lambda s: {
        "dependency": "mylib",
        "current_version": "1.0.0",
        "new_version": "2.0.0",
        "diff": "mock diff",
        "renovate_rating": None,
    })
    monkeypatch.setattr(rr, "research", lambda s: {
        "evidence": [EvidenceItem(source="changelog", url=None, text="removed foo()")],
    })
    monkeypatch.setattr(rr, "extract_breaking_changes", lambda s: {})
    # _findings is not in the TypedDict so LangGraph drops it between nodes;
    # stub assemble_verdict directly to test the 'breaking' branch.
    monkeypatch.setattr(rr, "assemble_verdict", lambda s: {
        "verdict": Verdict(decision="breaking", findings=[finding], summary="1 breaking change found."),
    })
    monkeypatch.setattr(rr, "post_review", lambda s: {"posted": True})

    graph = rr.build_renovate_graph()
    result = graph.invoke(dict(_RENOVATE_INITIAL))

    assert result["posted"] is True
    assert result["verdict"].decision == "breaking"
    assert len(result["verdict"].findings) == 1


# ─────────────────────────── Graph B helpers ───────────────────────────────

_SCAFFOLD_INITIAL: dict = {
    "request": "deploy myapp",
    "provided_links": [],
    "spec": {},
    "service_evidence": [],
    "helm_chart_found": False,
    "helm_chart_ref": None,
    "conventions": "",
    "manifests": {},
    "review_passed": False,
    "review_issues": [],
    "retry_count": 0,
    "pr_url": None,
}

_BASE_STUBS = {
    "parse_request": lambda s: {
        "spec": {"name": "myapp", "namespace": "default", "ports": [8080]},
        "provided_links": [],
    },
    "research_service": lambda s: {
        "service_evidence": [EvidenceItem(source="docs", url=None, text="image: myapp:latest")],
    },
    "load_conventions": lambda s: {"conventions": "apps/<name>/ layout"},
    "commit_and_pr": lambda s: {"pr_url": "https://gitea.test/myorg/myrepo/pulls/1"},
}

# ─────────────────────────── Graph B tests ─────────────────────────────────


def test_scaffold_graph_helm_true_path(monkeypatch):
    """When a Helm chart is found, graph routes through generate_helmrelease."""
    import ops_agent.graphs.scaffold_deploy as sd

    for attr, stub in _BASE_STUBS.items():
        monkeypatch.setattr(sd, attr, stub)

    monkeypatch.setattr(sd, "assess_helm", lambda s: {
        "helm_chart_found": True,
        "helm_chart_ref": "bitnami/nginx",
        "service_evidence": [],
    })
    monkeypatch.setattr(sd, "generate_helmrelease", lambda s: {
        "manifests": {"helmrelease.yaml": "apiVersion: helm.toolkit.fluxcd.io/v2beta1"},
        "retry_count": s.get("retry_count", 0),
        "review_issues": [],
    })
    monkeypatch.setattr(sd, "self_review", lambda s: {
        "review_passed": True,
        "review_issues": [],
    })

    graph = sd.build_scaffold_graph()
    result = graph.invoke(dict(_SCAFFOLD_INITIAL))

    assert result["helm_chart_found"] is True
    assert "helmrelease.yaml" in result["manifests"]
    assert result["pr_url"] == "https://gitea.test/myorg/myrepo/pulls/1"


def test_scaffold_graph_helm_false_path(monkeypatch):
    """When no Helm chart is found, graph routes through generate_kustomize."""
    import ops_agent.graphs.scaffold_deploy as sd

    for attr, stub in _BASE_STUBS.items():
        monkeypatch.setattr(sd, attr, stub)

    monkeypatch.setattr(sd, "assess_helm", lambda s: {
        "helm_chart_found": False,
        "helm_chart_ref": None,
        "service_evidence": [],
    })
    monkeypatch.setattr(sd, "generate_kustomize", lambda s: {
        "manifests": {
            "deployment.yaml": "apiVersion: apps/v1",
            "service.yaml": "apiVersion: v1",
            "kustomization.yaml": "resources:",
        },
        "retry_count": s.get("retry_count", 0),
        "review_issues": [],
    })
    monkeypatch.setattr(sd, "self_review", lambda s: {
        "review_passed": True,
        "review_issues": [],
    })

    graph = sd.build_scaffold_graph()
    result = graph.invoke(dict(_SCAFFOLD_INITIAL))

    assert result["helm_chart_found"] is False
    assert "deployment.yaml" in result["manifests"]
    assert result["pr_url"] == "https://gitea.test/myorg/myrepo/pulls/1"


def test_scaffold_graph_self_review_retry_loop(monkeypatch):
    """self_review fails once, graph retries generate_helmrelease, then passes."""
    import ops_agent.graphs.scaffold_deploy as sd

    generate_calls: list[int] = []
    review_calls: list[int] = []

    def fake_generate_helmrelease(s):
        review_issues = s.get("review_issues", [])
        generate_calls.append(len(generate_calls))
        return {
            "manifests": {"helmrelease.yaml": "yaml"},
            "retry_count": s.get("retry_count", 0) + (1 if review_issues else 0),
            "review_issues": [],
        }

    def fake_self_review(s):
        call_num = len(review_calls)
        review_calls.append(call_num)
        if call_num == 0:
            return {"review_passed": False, "review_issues": ["image is a placeholder"]}
        return {"review_passed": True, "review_issues": []}

    for attr, stub in _BASE_STUBS.items():
        monkeypatch.setattr(sd, attr, stub)

    monkeypatch.setattr(sd, "assess_helm", lambda s: {
        "helm_chart_found": True,
        "helm_chart_ref": "bitnami/myapp",
        "service_evidence": [],
    })
    monkeypatch.setattr(sd, "generate_helmrelease", fake_generate_helmrelease)
    monkeypatch.setattr(sd, "self_review", fake_self_review)

    graph = sd.build_scaffold_graph()
    result = graph.invoke(dict(_SCAFFOLD_INITIAL))

    assert len(generate_calls) == 2, "generate_helmrelease should be called twice"
    assert len(review_calls) == 2, "self_review should be called twice"
    assert result["retry_count"] == 1
    assert result["pr_url"] == "https://gitea.test/myorg/myrepo/pulls/1"
