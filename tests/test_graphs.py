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

from ops_agent.state import EvidenceItem, Finding, RiskAssessment, Verdict

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
    "_findings": [],
    "_risk": None,
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
    monkeypatch.setattr(rr, "assess_risk", lambda s: {})
    monkeypatch.setattr(rr, "post_review", lambda s: {"posted": True})

    graph = rr.build_renovate_graph()
    result = graph.invoke(dict(_RENOVATE_INITIAL))

    assert result["posted"] is True
    assert isinstance(result["verdict"], Verdict)
    assert result["verdict"].decision == "clear"


def test_renovate_findings_propagate_to_verdict(monkeypatch):
    """Regression guard: _findings from extract must reach assemble_verdict.

    Undeclared state keys are silently dropped by LangGraph, so _findings must be
    a declared channel. assemble_verdict runs for real here to prove it arrives.
    """
    import ops_agent.graphs.renovate_review as rr

    finding = Finding(claim="removed foo()", source="changelog", quote="foo() was removed", category="breaking")

    monkeypatch.setattr(rr, "ingest_pr", lambda s: {
        "dependency": "mylib", "current_version": "1.0.0", "new_version": "2.0.0",
        "diff": "", "renovate_rating": None,
    })
    monkeypatch.setattr(rr, "research", lambda s: {
        "evidence": [EvidenceItem(source="changelog", url=None, text="foo() was removed")],
    })
    monkeypatch.setattr(rr, "extract_breaking_changes", lambda s: {"_findings": [finding]})
    monkeypatch.setattr(rr, "assess_risk", lambda s: {})
    monkeypatch.setattr(rr, "post_review", lambda s: {"posted": True})

    graph = rr.build_renovate_graph()
    result = graph.invoke(dict(_RENOVATE_INITIAL))

    assert result["verdict"].decision == "breaking"
    assert result["verdict"].findings[0].claim == "removed foo()"


def test_renovate_graph_regression_path(monkeypatch):
    """A calendar-vs-semver scheme swap is flagged as a regression, not 'clear'."""
    import ops_agent.graphs.renovate_review as rr

    monkeypatch.setattr(rr, "ingest_pr", lambda s: {
        "dependency": "lscr.io/linuxserver/jellyfin",
        "current_version": "10.11.11", "new_version": "2021.12.16",
        "diff": "", "renovate_rating": None,
    })
    monkeypatch.setattr(rr, "research", lambda s: {
        "evidence": [EvidenceItem(source="dockerhub", url=None, text="pushed over 4 years ago")],
    })
    monkeypatch.setattr(rr, "extract_breaking_changes", lambda s: {})
    monkeypatch.setattr(rr, "assess_risk", lambda s: {})
    monkeypatch.setattr(rr, "post_review", lambda s: {"posted": True})

    graph = rr.build_renovate_graph()
    result = graph.invoke(dict(_RENOVATE_INITIAL))

    assert result["verdict"].decision == "regression"
    assert any(f.category == "regression" for f in result["verdict"].findings)


def test_renovate_graph_reasoned_risk_path(monkeypatch):
    """No verbatim finding, but the reasoned layer flags high risk → needs_human."""
    import ops_agent.graphs.renovate_review as rr

    monkeypatch.setattr(rr, "ingest_pr", lambda s: {
        "dependency": "mylib", "current_version": "1.0.0", "new_version": "2.0.0",
        "diff": "", "renovate_rating": None,
    })
    monkeypatch.setattr(rr, "research", lambda s: {
        "evidence": [EvidenceItem(source="changelog", url=None, text="major rewrite; config format changed")],
    })
    monkeypatch.setattr(rr, "extract_breaking_changes", lambda s: {"_findings": []})
    monkeypatch.setattr(rr, "assess_risk", lambda s: {
        "_risk": RiskAssessment(could_break=True, risk_level="high",
                                rationale="Major-version rewrite with config changes.",
                                signals=["1.x → 2.x major bump", "config format changed"]),
    })
    monkeypatch.setattr(rr, "post_review", lambda s: {"posted": True})

    graph = rr.build_renovate_graph()
    result = graph.invoke(dict(_RENOVATE_INITIAL))

    assert result["verdict"].decision == "needs_human"
    assert result["verdict"].risk is not None
    assert result["verdict"].risk.risk_level == "high"


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
    monkeypatch.setattr(rr, "assess_risk", lambda s: {})
    # Stub assemble_verdict directly to test the 'breaking' branch wiring.
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


# ─────────────────────── version-analysis unit tests ───────────────────────


@pytest.mark.parametrize("current,new,expected", [
    ("10.11.11", "2021.12.16", "scheme_change"),  # semver → calver (the jellyfin trap)
    ("2021.12.16", "10.11.11", "scheme_change"),  # calver → semver
    ("2.32.0", "2.28.0", "downgrade"),            # plain downgrade
    ("2.28.0", "2.32.0", None),                   # normal upgrade
    ("1.0.0", "2.0.0", None),                     # major upgrade, still forward
    ("2020.1.1", "2021.1.1", None),               # calver → calver upgrade
    ("", "2.0.0", None),                          # unknown current → no opinion
    ("latest", "stable", None),                   # non-numeric → no opinion
])
def test_analyze_version_direction(current, new, expected):
    from ops_agent.graphs.renovate_review import _analyze_version_direction

    assert _analyze_version_direction(current, new) == expected


def test_parse_version_components_stops_at_non_numeric():
    from ops_agent.graphs.renovate_review import _parse_version_components

    assert _parse_version_components("10.11.11ubu2604-ls43") == [10, 11, 11]
    assert _parse_version_components("v2.3.4") == [2, 3, 4]
    assert _parse_version_components("nightly") is None


def test_extract_current_version_from_diff():
    from ops_agent.graphs.renovate_review import _extract_current_version_from_diff

    diff = (
        "--- a/jellyfin.tf\n"
        "+++ b/jellyfin.tf\n"
        '-  image = "lscr.io/linuxserver/jellyfin:10.11.11"\n'
        '+  image = "lscr.io/linuxserver/jellyfin:2021.12.16"\n'
    )
    assert _extract_current_version_from_diff(diff, "2021.12.16") == "10.11.11"


def test_scheme_change_regression_finding_shape():
    """assemble_verdict injects a regression finding for a scheme change."""
    from ops_agent.graphs.renovate_review import assemble_verdict

    state = dict(_RENOVATE_INITIAL)
    state.update({
        "dependency": "jellyfin", "current_version": "10.11.11",
        "new_version": "2021.12.16",
        "evidence": [EvidenceItem(source="x", url=None, text="old")],
    })
    out = assemble_verdict(state)
    verdict = out["verdict"]
    assert verdict.decision == "regression"
    assert verdict.findings[0].source == "version-analysis"
