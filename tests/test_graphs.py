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
    from ops_agent.graphs.update_review.update_review import build_graph

    graph = build_graph()
    assert graph is not None


def test_scaffold_graph_compiles():
    from ops_agent.graphs.service_deploy.service_deploy import build_graph

    graph = build_graph()
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
    import ops_agent.graphs.update_review.update_review as rr

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

    graph = rr.build_graph()
    result = graph.invoke(dict(_RENOVATE_INITIAL))

    assert result["posted"] is True
    assert isinstance(result["verdict"], Verdict)
    assert result["verdict"].decision == "clear"


def test_renovate_findings_propagate_to_verdict(monkeypatch):
    """Regression guard: _findings from extract must reach assemble_verdict.

    Undeclared state keys are silently dropped by LangGraph, so _findings must be
    a declared channel. assemble_verdict runs for real here to prove it arrives.
    """
    import ops_agent.graphs.update_review.update_review as rr

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

    graph = rr.build_graph()
    result = graph.invoke(dict(_RENOVATE_INITIAL))

    assert result["verdict"].decision == "breaking"
    assert result["verdict"].findings[0].claim == "removed foo()"


def test_renovate_graph_regression_path(monkeypatch):
    """A calendar-vs-semver scheme swap is flagged as a regression, not 'clear'."""
    import ops_agent.graphs.update_review.update_review as rr

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

    graph = rr.build_graph()
    result = graph.invoke(dict(_RENOVATE_INITIAL))

    assert result["verdict"].decision == "regression"
    assert any(f.category == "regression" for f in result["verdict"].findings)


def test_renovate_graph_reasoned_risk_path(monkeypatch):
    """No verbatim finding, but the reasoned layer flags high risk → needs_human."""
    import ops_agent.graphs.update_review.update_review as rr

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

    graph = rr.build_graph()
    result = graph.invoke(dict(_RENOVATE_INITIAL))

    assert result["verdict"].decision == "needs_human"
    assert result["verdict"].risk is not None
    assert result["verdict"].risk.risk_level == "high"


def test_renovate_graph_breaking_path(monkeypatch):
    """Stub assemble_verdict to produce a 'breaking' verdict and verify post_review runs."""
    import ops_agent.graphs.update_review.update_review as rr

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

    graph = rr.build_graph()
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
    "existing_namespaces": [],
    "manifests": {},
    "review_passed": False,
    "review_issues": [],
    "retry_count": 0,
    "pr_url": None,
}

_BASE_STUBS = {
    "parse_request": lambda s: {
        "spec": {"name": "myapp", "namespace": None, "ports": [8080]},
        "provided_links": [],
    },
    "research_service": lambda s: {
        "service_evidence": [EvidenceItem(source="docs", url=None, text="image: myapp:latest")],
    },
    "load_conventions": lambda s: {
        "conventions": "apps/<name>/ layout",
        "existing_namespaces": ["media", "monitoring"],
    },
    "decide_namespace": lambda s: {"spec": {**s.get("spec", {}), "namespace": "myapp"}},
    "commit_and_pr": lambda s: {"pr_url": "https://gitea.test/myorg/myrepo/pulls/1"},
}

# ─────────────────────────── Graph B tests ─────────────────────────────────


def test_scaffold_graph_helm_true_path(monkeypatch):
    """When a Helm chart is found, graph routes through generate_helmrelease."""
    import ops_agent.graphs.service_deploy.service_deploy as sd

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

    graph = sd.build_graph()
    result = graph.invoke(dict(_SCAFFOLD_INITIAL))

    assert result["helm_chart_found"] is True
    assert "helmrelease.yaml" in result["manifests"]
    assert result["pr_url"] == "https://gitea.test/myorg/myrepo/pulls/1"


def test_scaffold_graph_helm_false_path(monkeypatch):
    """When no Helm chart is found, graph routes through generate_kustomize."""
    import ops_agent.graphs.service_deploy.service_deploy as sd

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

    graph = sd.build_graph()
    result = graph.invoke(dict(_SCAFFOLD_INITIAL))

    assert result["helm_chart_found"] is False
    assert "deployment.yaml" in result["manifests"]
    assert result["pr_url"] == "https://gitea.test/myorg/myrepo/pulls/1"


def test_scaffold_graph_self_review_retry_loop(monkeypatch):
    """self_review fails once, graph retries generate_helmrelease, then passes."""
    import ops_agent.graphs.service_deploy.service_deploy as sd

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

    graph = sd.build_graph()
    result = graph.invoke(dict(_SCAFFOLD_INITIAL))

    assert len(generate_calls) == 2, "generate_helmrelease should be called twice"
    assert len(review_calls) == 2, "self_review should be called twice"
    assert result["retry_count"] == 1
    assert result["pr_url"] == "https://gitea.test/myorg/myrepo/pulls/1"


# ─────────────────────── decide_namespace unit tests ───────────────────────


def test_decide_namespace_honors_user_choice(monkeypatch):
    """An explicit, non-system user namespace is kept without calling the LLM."""
    from ops_agent.graphs.service_deploy.nodes import decide_namespace as dn

    def _boom(*_a, **_k):
        raise AssertionError("LLM should not be called when user specified a namespace")

    monkeypatch.setattr(dn, "get_llm", _boom)

    result = dn.decide_namespace({
        "spec": {"name": "myapp", "namespace": "media"},
        "existing_namespaces": ["media"],
    })
    assert result["spec"]["namespace"] == "media"


def test_decide_namespace_rejects_user_default(monkeypatch):
    """A user 'default' is treated as unset and re-decided (never lands in default)."""
    from ops_agent.graphs.service_deploy.nodes import decide_namespace as dn

    monkeypatch.setattr(dn, "get_llm", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("offline")))

    result = dn.decide_namespace({
        "spec": {"name": "myapp", "namespace": "default"},
        "existing_namespaces": [],
    })
    assert result["spec"]["namespace"] == "myapp"  # guardrail → service name


def test_decide_namespace_falls_back_to_service_name_on_llm_failure(monkeypatch):
    """When the LLM step fails and no namespace is set, fall back to the service name."""
    from ops_agent.graphs.service_deploy.nodes import decide_namespace as dn

    monkeypatch.setattr(dn, "get_llm", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("offline")))

    result = dn.decide_namespace({
        "spec": {"name": "grafana", "namespace": None},
        "existing_namespaces": ["media"],
    })
    assert result["spec"]["namespace"] == "grafana"


def test_decide_namespace_sanitize_rejects_system_namespaces():
    from ops_agent.graphs.service_deploy.nodes.decide_namespace import _sanitize

    assert _sanitize("default") is None
    assert _sanitize("KUBE-SYSTEM") is None
    assert _sanitize("  Media  ") == "media"
    assert _sanitize(None) is None
    assert _sanitize("") is None


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
    from ops_agent.graphs.update_review.nodes.assemble_verdict import _analyze_version_direction

    assert _analyze_version_direction(current, new) == expected


def test_parse_version_components_stops_at_non_numeric():
    from ops_agent.graphs.update_review.nodes.assemble_verdict import _parse_version_components

    assert _parse_version_components("10.11.11ubu2604-ls43") == [10, 11, 11]
    assert _parse_version_components("v2.3.4") == [2, 3, 4]
    assert _parse_version_components("nightly") is None


def test_extract_current_version_from_diff():
    from ops_agent.graphs.update_review.nodes.ingest_pr import _extract_current_version_from_diff

    diff = (
        "--- a/jellyfin.tf\n"
        "+++ b/jellyfin.tf\n"
        '-  image = "lscr.io/linuxserver/jellyfin:10.11.11"\n'
        '+  image = "lscr.io/linuxserver/jellyfin:2021.12.16"\n'
    )
    assert _extract_current_version_from_diff(diff, "2021.12.16") == "10.11.11"


def test_scheme_change_regression_finding_shape():
    """assemble_verdict injects a regression finding for a scheme change."""
    from ops_agent.graphs.update_review.nodes.assemble_verdict import assemble_verdict

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


# ─────────────────────── post_review approval tests ───────────────────────


def test_post_review_approves_clear_verdict(monkeypatch):
    """When verdict is 'clear', post_review should call approve_pr."""
    from ops_agent.graphs.update_review.nodes.post_review import post_review
    from ops_agent.tools.gitea import GiteaClient

    approve_calls = []

    def fake_post_issue_comment(self, owner, repo, index, body):
        return {"id": 1}

    def fake_approve_pr(self, owner, repo, index, body="LGTM"):
        approve_calls.append((owner, repo, index, body))
        return {"id": 2}

    monkeypatch.setattr(GiteaClient, "post_issue_comment", fake_post_issue_comment)
    monkeypatch.setattr(GiteaClient, "approve_pr", fake_approve_pr)
    monkeypatch.setattr(GiteaClient, "close", lambda self: None)

    state = dict(_RENOVATE_INITIAL)
    state.update({
        "dependency": "requests",
        "current_version": "2.28.0",
        "new_version": "2.32.0",
        "verdict": Verdict(
            decision="clear",
            findings=[],
            summary="No breaking changes found.",
            risk=None,
        ),
    })

    result = post_review(state)

    assert result["posted"] is True
    assert len(approve_calls) == 1
    assert approve_calls[0] == ("myorg", "myrepo", 1, "No breaking changes found.")


def test_post_review_does_not_approve_breaking_verdict(monkeypatch):
    """When verdict is 'breaking', post_review should NOT call approve_pr."""
    from ops_agent.graphs.update_review.nodes.post_review import post_review
    from ops_agent.tools.gitea import GiteaClient

    approve_calls = []

    def fake_post_issue_comment(self, owner, repo, index, body):
        return {"id": 1}

    def fake_approve_pr(self, owner, repo, index, body="LGTM"):
        approve_calls.append((owner, repo, index, body))
        return {"id": 2}

    monkeypatch.setattr(GiteaClient, "post_issue_comment", fake_post_issue_comment)
    monkeypatch.setattr(GiteaClient, "approve_pr", fake_approve_pr)
    monkeypatch.setattr(GiteaClient, "close", lambda self: None)

    state = dict(_RENOVATE_INITIAL)
    state.update({
        "dependency": "requests",
        "current_version": "2.28.0",
        "new_version": "2.32.0",
        "verdict": Verdict(
            decision="breaking",
            findings=[Finding(claim="API removed", source="changelog", quote="removed foo()")],
            summary="Breaking changes found.",
            risk=None,
        ),
    })

    result = post_review(state)

    assert result["posted"] is True
    assert len(approve_calls) == 0


def test_post_review_approval_failure_does_not_fail_post(monkeypatch):
    """If approve_pr fails, post_review should still return posted=True."""
    from ops_agent.graphs.update_review.nodes.post_review import post_review
    from ops_agent.tools.gitea import GiteaClient

    def fake_post_issue_comment(self, owner, repo, index, body):
        return {"id": 1}

    def fake_approve_pr_fails(self, owner, repo, index, body="LGTM"):
        raise Exception("Authorization failed: token needs write:repository scope")

    monkeypatch.setattr(GiteaClient, "post_issue_comment", fake_post_issue_comment)
    monkeypatch.setattr(GiteaClient, "approve_pr", fake_approve_pr_fails)
    monkeypatch.setattr(GiteaClient, "close", lambda self: None)

    state = dict(_RENOVATE_INITIAL)
    state.update({
        "dependency": "requests",
        "current_version": "2.28.0",
        "new_version": "2.32.0",
        "verdict": Verdict(
            decision="clear",
            findings=[],
            summary="No breaking changes found.",
            risk=None,
        ),
    })

    result = post_review(state)

    # Even though approve_pr failed, the comment was posted
    assert result["posted"] is True


# ─────────────────── interactive re-drive: shared helpers ───────────────────


class _FakeActivityClient:
    """Fake GiteaClient exposing comments + reviews for new_steering_inputs."""

    def __init__(self, comments=None, reviews=None, review_comments=None):
        self._comments = comments or []
        self._reviews = reviews or []
        self._review_comments = review_comments or {}

    def list_issue_comments(self, owner, repo, index):
        return self._comments

    def list_pr_reviews(self, owner, repo, index):
        return self._reviews

    def list_pr_review_comments(self, owner, repo, index, review_id):
        return self._review_comments.get(review_id, [])


def test_new_steering_inputs_filters_and_boundary():
    """Only non-bot activity newer than our last activity, with real text."""
    from ops_agent.graphs.interactive import new_steering_inputs

    client = _FakeActivityClient(
        comments=[
            {"id": 1, "user": {"login": "renovate"}, "created_at": "2024-01-01T00:00:00Z", "body": "old"},
            {"id": 2, "user": {"login": "ops-agent"}, "created_at": "2024-02-01T00:00:00Z", "body": "our review"},
            {"id": 3, "user": {"login": "human"}, "created_at": "2024-03-01T00:00:00Z", "body": "please recheck"},
        ],
    )
    out = new_steering_inputs(client, "o", "r", 1, "ops-agent")
    # Only comment 3: comment 1 predates our boundary, comment 2 is ours.
    assert [c["ref"] for c in out] == ["3"]
    assert out[0]["text"] == "please recheck"


def test_new_steering_inputs_picks_up_review_request_changes():
    """Regression: a 'Request changes' REVIEW (not a comment) is steering input."""
    from ops_agent.graphs.interactive import new_steering_inputs

    client = _FakeActivityClient(
        comments=[],  # nothing in the comments endpoint (the reported bug)
        reviews=[
            {"id": 2, "user": {"login": "mattyice"}, "state": "REQUEST_CHANGES",
             "submitted_at": "2026-08-02T03:16:43-05:00", "body": "use plex instead of jellyfin"},
        ],
    )
    out = new_steering_inputs(client, "o", "r", 171, "ops-agent")
    assert len(out) == 1
    assert out[0]["kind"] == "review"
    assert out[0]["text"] == "use plex instead of jellyfin"


def test_new_steering_inputs_ignores_bare_approval_and_own_activity():
    """An empty-body approval and our own review never count as steering."""
    from ops_agent.graphs.interactive import new_steering_inputs

    client = _FakeActivityClient(
        comments=[],
        reviews=[
            {"id": 5, "user": {"login": "human"}, "state": "APPROVED",
             "submitted_at": "2026-08-02T04:00:00-05:00", "body": ""},
            {"id": 6, "user": {"login": "ops-agent"}, "state": "APPROVED",
             "submitted_at": "2026-08-02T05:00:00-05:00", "body": "LGTM"},
        ],
    )
    assert new_steering_inputs(client, "o", "r", 1, "ops-agent") == []


def test_new_steering_inputs_boundary_handles_mixed_timezones():
    """Offset timestamps must compare chronologically, not lexicographically."""
    from ops_agent.graphs.interactive import new_steering_inputs

    client = _FakeActivityClient(
        # Our review at 08:00Z; the human comment at 04:00-05:00 == 09:00Z is
        # LATER in real time but sorts BEFORE as a raw string.
        comments=[
            {"id": 1, "user": {"login": "human"}, "created_at": "2026-08-02T04:00:00-05:00", "body": "recheck"},
        ],
        reviews=[
            {"id": 2, "user": {"login": "ops-agent"}, "state": "APPROVED",
             "submitted_at": "2026-08-02T08:00:00Z", "body": "done"},
        ],
    )
    out = new_steering_inputs(client, "o", "r", 1, "ops-agent")
    assert [c["ref"] for c in out] == ["1"]  # correctly seen as newer


# ─────────────────── update_review triage / classify ────────────────────────


class _FakeTriageClient:
    def __init__(self, pr=None, comments=None, reviews=None):
        self._pr = pr or {}
        self._comments = comments or []
        self._reviews = reviews or []

    def get_pr(self, owner, repo, index):
        return self._pr

    def list_issue_comments(self, owner, repo, index):
        return self._comments

    def list_pr_reviews(self, owner, repo, index):
        return self._reviews

    def list_pr_review_comments(self, owner, repo, index, review_id):
        return []

    def close(self):
        pass


def test_ur_triage_first_run_routes_full_without_gitea(monkeypatch):
    """No prior verdict → full pipeline, and NO Gitea calls are made."""
    from ops_agent.graphs.update_review.nodes import triage as tri

    def _boom():
        raise AssertionError("triage must not touch Gitea on the first run")

    monkeypatch.setattr(tri, "GiteaClient", _boom)

    out = tri.triage({"verdict": None})
    assert out["_route"] == "full"
    assert out["new_inputs"] == []


def test_ur_triage_new_commit_routes_full(monkeypatch):
    """A head SHA different from the last reviewed one → full re-review."""
    from ops_agent.graphs.update_review.nodes import triage as tri

    client = _FakeTriageClient(pr={"commits": [{"id": "sha_new"}]}, comments=[])
    monkeypatch.setattr(tri, "GiteaClient", lambda: client)
    monkeypatch.setattr(tri, "get_bot_login", lambda: "ops-agent")

    out = tri.triage({
        "verdict": Verdict(decision="clear", findings=[], summary="ok"),
        "thread_id": "sha_old", "_owner": "o", "_repo": "r", "pr_index": 1,
    })
    assert out["_route"] == "full"


def test_ur_triage_new_comment_routes_comment(monkeypatch):
    """A foreign comment past our boundary → comment route with new_inputs set."""
    from ops_agent.graphs.update_review.nodes import triage as tri

    comments = [
        {"id": 7, "user": {"login": "human"}, "created_at": "2024-03-01T00:00:00Z", "body": "recheck plz"},
    ]
    client = _FakeTriageClient(pr={"commits": [{"id": "sha_old"}]}, comments=comments)
    monkeypatch.setattr(tri, "GiteaClient", lambda: client)
    monkeypatch.setattr(tri, "get_bot_login", lambda: "ops-agent")

    out = tri.triage({
        "verdict": Verdict(decision="clear", findings=[], summary="ok"),
        "thread_id": "sha_old", "_owner": "o", "_repo": "r", "pr_index": 1,
    })
    assert out["_route"] == "comment"
    assert out["new_inputs"][0]["text"] == "recheck plz"
    assert out["new_inputs"][0]["author"] == "human"


def test_ur_triage_self_comment_is_no_change(monkeypatch):
    """A comment authored by us must NOT re-trigger the graph."""
    from ops_agent.graphs.update_review.nodes import triage as tri

    comments = [
        {"id": 8, "user": {"login": "ops-agent"}, "created_at": "2024-03-01T00:00:00Z", "body": "our review"},
    ]
    client = _FakeTriageClient(pr={"commits": [{"id": "sha_old"}]}, comments=comments)
    monkeypatch.setattr(tri, "GiteaClient", lambda: client)
    monkeypatch.setattr(tri, "get_bot_login", lambda: "ops-agent")

    out = tri.triage({
        "verdict": Verdict(decision="clear", findings=[], summary="ok"),
        "thread_id": "sha_old", "_owner": "o", "_repo": "r", "pr_index": 1,
    })
    assert out["_route"] == "none"
    assert out["new_inputs"] == []


def test_classify_comment_defaults_to_light_on_llm_failure(monkeypatch):
    """If the reasoning call fails, degrade to the cheaper 'light' re-drive."""
    from ops_agent.graphs.update_review.nodes import classify_comment as cc

    def _boom(_persona):
        raise RuntimeError("offline")

    monkeypatch.setattr(cc, "get_llm", _boom)

    out = cc.classify_comment({
        "dependency": "requests", "current_version": "1", "new_version": "2",
        "verdict": Verdict(decision="clear", findings=[], summary="ok"),
        "new_inputs": [{"author": "human", "text": "are you sure?"}],
    })
    assert out["_route"] == "light"


# ───────────────── update_review re-drive end-to-end (MemorySaver) ──────────


def test_ur_redrive_light_path_no_double_evidence(monkeypatch):
    """A comment-driven light re-drive re-judges without re-researching and
    without duplicating the accumulating evidence channel."""
    import ops_agent.graphs.update_review.update_review as ur
    from ops_agent.graphs.update_review.nodes import classify_comment as cc
    from ops_agent.graphs.update_review.nodes import triage as tri
    from langgraph.checkpoint.memory import MemorySaver

    # Compile with an in-memory checkpointer so state persists across turns.
    saver = MemorySaver()
    monkeypatch.setattr("ops_agent.checkpointing.get_checkpoint_saver", lambda: saver)

    research_calls: list[int] = []

    monkeypatch.setattr(ur, "ingest_pr", lambda s: {
        "dependency": "requests", "current_version": "2.28.0", "new_version": "2.32.0",
        "diff": "", "renovate_rating": None, "thread_id": "sha1",
    })

    def stub_research(s):
        research_calls.append(1)
        return {"evidence": [EvidenceItem(source="changelog", url=None, text="notes")]}

    monkeypatch.setattr(ur, "research", stub_research)
    monkeypatch.setattr(ur, "extract_breaking_changes", lambda s: {})
    monkeypatch.setattr(ur, "assess_risk", lambda s: {})
    monkeypatch.setattr(ur, "post_review", lambda s: {"posted": True})

    # Second turn: triage sees a new foreign comment (no new commit).
    comments = [
        {"id": 9, "user": {"login": "human"}, "created_at": "2024-03-01T00:00:00Z", "body": "are you sure?"},
    ]
    monkeypatch.setattr(tri, "GiteaClient", lambda: _FakeTriageClient(
        pr={"commits": [{"id": "sha1"}]}, comments=comments))
    monkeypatch.setattr(tri, "get_bot_login", lambda: "ops-agent")
    # classify picks the cheap path.
    monkeypatch.setattr(cc, "get_llm", lambda _p: (_ for _ in ()).throw(RuntimeError("x")))

    graph = ur.build_graph()
    config = {"configurable": {"thread_id": "t1"}}

    initial = dict(_RENOVATE_INITIAL)
    initial.update({"thread_id": "sha1", "new_inputs": [], "turn": 0, "_route": ""})
    first = graph.invoke(initial, config)
    assert first["verdict"] is not None
    assert len(research_calls) == 1
    assert len(first["evidence"]) == 1

    # Re-drive with the increment only (mirrors run()).
    second = graph.invoke({"turn": 1}, config)
    assert second["turn"] == 1
    assert len(research_calls) == 1, "light re-drive must not re-run research"
    assert len(second["evidence"]) == 1, "evidence must not be doubled on re-drive"


def test_ur_redrive_idle_tick_is_noop(monkeypatch):
    """No new commit and no foreign comment → triage ends the run untouched."""
    import ops_agent.graphs.update_review.update_review as ur
    from ops_agent.graphs.update_review.nodes import triage as tri
    from langgraph.checkpoint.memory import MemorySaver

    saver = MemorySaver()
    monkeypatch.setattr("ops_agent.checkpointing.get_checkpoint_saver", lambda: saver)

    monkeypatch.setattr(ur, "ingest_pr", lambda s: {
        "dependency": "requests", "current_version": "2.28.0", "new_version": "2.32.0",
        "diff": "", "renovate_rating": None, "thread_id": "sha1",
    })
    monkeypatch.setattr(ur, "research", lambda s: {
        "evidence": [EvidenceItem(source="c", url=None, text="n")]})
    monkeypatch.setattr(ur, "extract_breaking_changes", lambda s: {})
    monkeypatch.setattr(ur, "assess_risk", lambda s: {})

    post_calls: list[int] = []
    monkeypatch.setattr(ur, "post_review", lambda s: post_calls.append(1) or {"posted": True})

    # Idle: only our own comment exists, head unchanged.
    comments = [{"id": 2, "user": {"login": "ops-agent"}, "created_at": "2024-02-01T00:00:00Z", "body": "review"}]
    monkeypatch.setattr(tri, "GiteaClient", lambda: _FakeTriageClient(
        pr={"commits": [{"id": "sha1"}]}, comments=comments))
    monkeypatch.setattr(tri, "get_bot_login", lambda: "ops-agent")

    graph = ur.build_graph()
    config = {"configurable": {"thread_id": "t2"}}
    initial = dict(_RENOVATE_INITIAL)
    initial.update({"thread_id": "sha1", "new_inputs": [], "turn": 0, "_route": ""})
    graph.invoke(initial, config)
    assert len(post_calls) == 1  # first run posted

    graph.invoke({"turn": 1}, config)
    assert len(post_calls) == 1, "idle re-drive must not post again"


# ───────────────── service_deploy triage / push-to-existing ─────────────────


def test_sd_triage_first_run_routes_full(monkeypatch):
    """No PR yet → full scaffold, no Gitea calls."""
    from ops_agent.graphs.service_deploy.nodes import triage as tri

    monkeypatch.setattr(tri, "GiteaClient", lambda: (_ for _ in ()).throw(AssertionError("no gitea")))
    out = tri.triage({"pr_url": None, "pr_index": None})
    assert out["_route"] == "full"


def test_sd_triage_steer_resets_review_budget(monkeypatch):
    """A foreign PR comment → steer, with a fresh self-review budget."""
    from ops_agent.graphs.service_deploy.nodes import triage as tri

    comments = [{"id": 4, "user": {"login": "human"}, "created_at": "2024-03-01T00:00:00Z", "body": "use 2 replicas"}]
    monkeypatch.setattr(tri, "GiteaClient", lambda: _FakeTriageClient(comments=comments))
    monkeypatch.setattr(tri, "get_bot_login", lambda: "ops-agent")

    out = tri.triage({
        "pr_url": "http://pr/5", "pr_index": 5, "_owner": "o", "_repo": "r",
    })
    assert out["_route"] == "steer"
    assert out["new_inputs"][0]["text"] == "use 2 replicas"
    assert out["retry_count"] == 0
    assert out["review_issues"] == []


def _install_fake_git(monkeypatch, tmp_path, push_result=None):
    """Stub the git plumbing in commit_and_pr so only PR logic is exercised.

    ``push_result`` is the list of PushInfo-like objects origin.push returns;
    default [] means a clean, accepted push.
    """
    import ops_agent.graphs.service_deploy.nodes.commit_and_pr as cp

    class _FakeGit:
        def add(self, *a, **k):
            pass

    class _FakeIndex:
        def commit(self, *a, **k):
            return None

    class _FakeRemote:
        def push(self, *a, **k):
            return push_result if push_result is not None else []

    class _FakeRepo:
        def __init__(self, *a, **k):
            self.git = _FakeGit()
            self.index = _FakeIndex()
            self.remotes = type("R", (), {"origin": _FakeRemote()})()

    monkeypatch.setattr(cp, "_get_or_clone_repo", lambda *a, **k: tmp_path)
    monkeypatch.setattr(cp, "_sync_and_branch", lambda *a, **k: None)
    monkeypatch.setattr(cp.gitlib, "Repo", _FakeRepo)
    return cp


def test_sd_commit_pushes_to_existing_pr(monkeypatch, tmp_path):
    """On a steer turn (pr_index set) commit_and_pr must NOT open a new PR."""
    cp = _install_fake_git(monkeypatch, tmp_path)

    posted: list[tuple] = []

    class _FakeClient:
        def create_pr(self, **k):
            raise AssertionError("must not open a second PR when one exists")

        def post_issue_comment(self, owner, repo, index, body):
            posted.append((index, body))
            return {"id": 1}

        def close(self):
            pass

    monkeypatch.setattr(cp, "GiteaClient", lambda: _FakeClient())

    result = cp.commit_and_pr({
        "_owner": "o", "_repo": "r",
        "spec": {"name": "myapp"},
        "manifests": {"deployment.yaml": "yaml"},
        "pr_url": "http://pr/5", "pr_index": 5,
        "new_inputs": [{"author": "human", "text": "use 2 replicas"}],
    })
    assert result["pr_url"] == "http://pr/5"
    assert result["pr_index"] == 5
    assert result["pr_branch"] == "ops-agent/deploy-myapp"
    assert len(posted) == 1 and posted[0][0] == 5  # ack on the existing PR


def test_sd_commit_opens_pr_on_first_run(monkeypatch, tmp_path):
    """First run (no pr_index) opens a PR and records its identity."""
    cp = _install_fake_git(monkeypatch, tmp_path)

    class _FakeClient:
        def create_pr(self, **k):
            return {"html_url": "http://pr/9", "number": 9}

        def post_issue_comment(self, *a, **k):
            raise AssertionError("first run should not post a steer ack")

        def close(self):
            pass

    monkeypatch.setattr(cp, "GiteaClient", lambda: _FakeClient())

    result = cp.commit_and_pr({
        "_owner": "o", "_repo": "r",
        "spec": {"name": "myapp"},
        "manifests": {"deployment.yaml": "yaml"},
        "pr_url": None, "pr_index": None,
    })
    assert result["pr_url"] == "http://pr/9"
    assert result["pr_index"] == 9
    assert result["pr_branch"] == "ops-agent/deploy-myapp"


def test_sd_commit_raises_on_rejected_push(monkeypatch, tmp_path):
    """A remote-rejected push must raise (never post a false 'pushed' ack)."""
    import git as gitlib

    class _RejectedInfo:
        flags = gitlib.PushInfo.REJECTED
        summary = "non-fast-forward"

    cp = _install_fake_git(monkeypatch, tmp_path, push_result=[_RejectedInfo()])

    class _FakeClient:
        def post_issue_comment(self, *a, **k):
            raise AssertionError("must not post an ack when the push was rejected")

        def create_pr(self, **k):
            raise AssertionError("must not open a PR when the push was rejected")

        def close(self):
            pass

    monkeypatch.setattr(cp, "GiteaClient", lambda: _FakeClient())

    with pytest.raises(RuntimeError, match="Failed to push"):
        cp.commit_and_pr({
            "_owner": "o", "_repo": "r",
            "spec": {"name": "myapp"},
            "manifests": {"deployment.yaml": "yaml"},
            "pr_url": "http://pr/5", "pr_index": 5,
            "new_inputs": [{"author": "human", "text": "use 2 replicas"}],
        })
