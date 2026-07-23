#!/usr/bin/env python3
"""Comprehensive Helm template unit tests for Cogtrix chart.

Renders templates with various values combinations and asserts on the output.
Run:  python3 helm/cogtrix/tests/test_templates.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

CHART_DIR = Path(__file__).resolve().parent.parent
RELEASE_NAME = "test-release"


def _helm_template(*extra_args: str) -> list[dict[str, Any]]:
    """Run `helm template` and return parsed documents."""
    cmd = [
        "helm",
        "template",
        RELEASE_NAME,
        str(CHART_DIR),
        *extra_args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    docs = []
    for doc in yaml.safe_load_all(result.stdout):
        if doc:  # Skip empty documents
            docs.append(doc)
    return docs


def _find(docs: list[dict[str, Any]], kind: str, name_substr: str = "") -> dict[str, Any] | None:
    """Find first document matching kind and optional name substring."""
    for doc in docs:
        if doc.get("kind") == kind:
            meta_name = doc.get("metadata", {}).get("name", "")
            if not name_substr or name_substr in meta_name:
                return doc
    return None


def _find_all(docs: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    """Find all documents of a given kind."""
    return [doc for doc in docs if doc.get("kind") == kind]


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


class TestDefaultValues:
    """Render with zero overrides."""

    def setup_method(self):
        self.docs = _helm_template()

    def test_helm_lint_passes(self):
        subprocess.run(
            ["helm", "lint", str(CHART_DIR)],
            capture_output=True,
            check=True,
        )

    def test_all_manifests_present(self):
        kinds = {d["kind"] for d in self.docs}
        assert "ServiceAccount" in kinds
        assert "PersistentVolumeClaim" in kinds
        assert "Service" in kinds
        assert "Deployment" in kinds
        assert "Pod" in kinds  # helm test hook

    def test_no_hpa_by_default(self):
        assert _find(self.docs, "HorizontalPodAutoscaler") is None

    def test_no_ingress_by_default(self):
        assert _find(self.docs, "Ingress") is None

    def test_no_secret_by_default(self):
        assert _find(self.docs, "Secret") is None

    def test_no_configmap_by_default(self):
        assert _find(self.docs, "ConfigMap") is None

    def test_networkpolicy_enabled_by_default(self):
        np = _find(self.docs, "NetworkPolicy")
        assert np is not None
        assert np["spec"]["policyTypes"] == ["Ingress", "Egress"]
        assert np["spec"]["ingress"][0]["ports"][0]["port"] == 8000
        assert np["spec"]["egress"][0]["ports"][0]["port"] == 53
        assert np["spec"]["egress"][1]["ports"][0]["port"] == 443

    def test_no_pdb_by_default(self):
        assert _find(self.docs, "PodDisruptionBudget") is None

    def test_no_servicemonitor_by_default(self):
        assert _find(self.docs, "ServiceMonitor") is None

    def test_replica_count_is_one(self):
        deploy = _find(self.docs, "Deployment")
        assert deploy is not None
        assert deploy["spec"]["replicas"] == 1

    def test_deployment_has_init_container(self):
        deploy = _find(self.docs, "Deployment")
        init = deploy["spec"]["template"]["spec"].get("initContainers", [])
        assert len(init) == 1
        assert init[0]["name"] == "init-data"

    def test_main_container_security_context(self):
        deploy = _find(self.docs, "Deployment")
        container = deploy["spec"]["template"]["spec"]["containers"][0]
        sc = container["securityContext"]
        assert sc["allowPrivilegeEscalation"] is False
        assert sc["readOnlyRootFilesystem"] is True
        assert sc["capabilities"]["drop"] == ["ALL"]

    def test_pod_security_context(self):
        deploy = _find(self.docs, "Deployment")
        psc = deploy["spec"]["template"]["spec"]["securityContext"]
        assert psc["runAsNonRoot"] is True
        assert psc["runAsUser"] == 1000
        assert psc["runAsGroup"] == 1000

    def test_service_is_clusterip(self):
        svc = _find(self.docs, "Service")
        assert svc["spec"]["type"] == "ClusterIP"
        assert svc["spec"]["ports"][0]["port"] == 8000

    def test_pvc_size(self):
        pvc = _find(self.docs, "PersistentVolumeClaim")
        assert pvc["spec"]["resources"]["requests"]["storage"] == "10Gi"
        assert pvc["spec"]["accessModes"] == ["ReadWriteOnce"]

    def test_service_account_name_matches(self):
        deploy = _find(self.docs, "Deployment")
        sa_name = deploy["spec"]["template"]["spec"]["serviceAccountName"]
        sa = _find(self.docs, "ServiceAccount")
        assert sa is not None
        assert sa["metadata"]["name"] == sa_name

    def test_image_tag_defaults_to_appversion(self):
        deploy = _find(self.docs, "Deployment")
        image = deploy["spec"]["template"]["spec"]["containers"][0]["image"]
        assert image == "northlandpositronics/cogtrix:0.2.6"

    def test_liveness_probe(self):
        deploy = _find(self.docs, "Deployment")
        container = deploy["spec"]["template"]["spec"]["containers"][0]
        probe = container["livenessProbe"]
        assert probe["httpGet"]["path"] == "/api/v1/health"
        assert probe["httpGet"]["port"] == "http"
        assert probe["initialDelaySeconds"] == 30

    def test_readiness_probe(self):
        deploy = _find(self.docs, "Deployment")
        container = deploy["spec"]["template"]["spec"]["containers"][0]
        probe = container["readinessProbe"]
        assert probe["httpGet"]["path"] == "/api/v1/health/ready"
        assert probe["httpGet"]["port"] == "http"

    def test_startup_probe(self):
        deploy = _find(self.docs, "Deployment")
        container = deploy["spec"]["template"]["spec"]["containers"][0]
        probe = container["startupProbe"]
        assert probe["httpGet"]["path"] == "/api/v1/health"
        assert probe["failureThreshold"] == 30

    def test_labels_include_standard_helm_labels(self):
        deploy = _find(self.docs, "Deployment")
        labels = deploy["metadata"]["labels"]
        assert "app.kubernetes.io/name" in labels
        assert "app.kubernetes.io/instance" in labels
        assert "app.kubernetes.io/version" in labels
        assert "helm.sh/chart" in labels
        assert "app.kubernetes.io/managed-by" in labels

    def test_container_env_has_defaults(self):
        deploy = _find(self.docs, "Deployment")
        env = deploy["spec"]["template"]["spec"]["containers"][0]["env"]
        env_dict = {e["name"]: e.get("value") for e in env}
        assert env_dict.get("COGTRIX_DATA_DIR") == "/data"
        assert env_dict.get("TZ") == "UTC"
        assert env_dict.get("PYTHONUNBUFFERED") == "1"

    def test_volume_mounts(self):
        deploy = _find(self.docs, "Deployment")
        mounts = deploy["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]
        names = {m["name"] for m in mounts}
        assert "data" in names

    def test_volumes_include_pvc(self):
        deploy = _find(self.docs, "Deployment")
        vols = deploy["spec"]["template"]["spec"]["volumes"]
        data_vol = next(v for v in vols if v["name"] == "data")
        assert "persistentVolumeClaim" in data_vol

    def test_resources_set(self):
        deploy = _find(self.docs, "Deployment")
        res = deploy["spec"]["template"]["spec"]["containers"][0]["resources"]
        assert res["limits"]["cpu"] == "2000m"
        assert res["limits"]["memory"] == "4Gi"
        assert res["requests"]["cpu"] == "500m"
        assert res["requests"]["memory"] == "1Gi"


class TestSecretsEnabled:
    """Secrets + env var injection."""

    def setup_method(self):
        self.docs = _helm_template(
            "--set",
            "secrets.enabled=true",
            "--set",
            "secrets.jwtSecret=my-jwt-secret",
            "--set",
            "secrets.openaiApiKey=sk-test",
            "--set",
            "secrets.anthropicApiKey=anthro-test",
            "--set",
            "secrets.databaseUrl=postgresql://test",
        )

    def test_secret_manifest_exists(self):
        secret = _find(self.docs, "Secret")
        assert secret is not None
        assert secret["type"] == "Opaque"

    def test_jwt_secret_value(self):
        secret = _find(self.docs, "Secret")
        assert secret["stringData"]["jwt-secret"] == "my-jwt-secret"

    def test_openai_key_value(self):
        secret = _find(self.docs, "Secret")
        assert secret["stringData"]["openai-api-key"] == "sk-test"

    def test_anthropic_key_value(self):
        secret = _find(self.docs, "Secret")
        assert secret["stringData"]["anthropic-api-key"] == "anthro-test"

    def test_database_url_value(self):
        secret = _find(self.docs, "Secret")
        assert secret["stringData"]["database-url"] == "postgresql://test"

    def test_env_vars_use_secret_ref(self):
        deploy = _find(self.docs, "Deployment")
        env = deploy["spec"]["template"]["spec"]["containers"][0]["env"]
        env_dict = {e["name"]: e for e in env}

        assert "COGTRIX_JWT_SECRET" in env_dict
        assert env_dict["COGTRIX_JWT_SECRET"]["valueFrom"]["secretKeyRef"]["key"] == "jwt-secret"

        assert "OPENAI_API_KEY" in env_dict
        assert env_dict["OPENAI_API_KEY"]["valueFrom"]["secretKeyRef"]["key"] == "openai-api-key"

        assert "DATABASE_URL" in env_dict
        assert env_dict["DATABASE_URL"]["valueFrom"]["secretKeyRef"]["key"] == "database-url"

    def test_unset_secrets_not_present(self):
        secret = _find(self.docs, "Secret")
        # gemini, xai, deepseek, ollama were not set
        keys = set(secret["stringData"].keys())
        assert "gemini-api-key" not in keys
        assert "xai-api-key" not in keys
        assert "deepseek-api-key" not in keys


class TestConfigEnabled:
    """ConfigMap mounted as .cogtrix.yaml."""

    def setup_method(self):
        self.docs = _helm_template(
            "--set",
            "config.enabled=true",
        )

    def test_configmap_exists(self):
        cm = _find(self.docs, "ConfigMap")
        assert cm is not None
        assert ".cogtrix.yaml" in cm["data"]

    def test_configmap_mounted(self):
        deploy = _find(self.docs, "Deployment")
        mounts = deploy["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]
        cm_mount = next((m for m in mounts if m.get("name") == "config"), None)
        assert cm_mount is not None
        assert cm_mount["mountPath"] == "/app/.cogtrix.yaml"
        assert cm_mount["subPath"] == ".cogtrix.yaml"
        assert cm_mount.get("readOnly") is True

    def test_config_volume_present(self):
        deploy = _find(self.docs, "Deployment")
        vols = deploy["spec"]["template"]["spec"]["volumes"]
        config_vol = next((v for v in vols if v.get("name") == "config"), None)
        assert config_vol is not None
        assert "configMap" in config_vol


class TestIngress:
    """Ingress template."""

    def test_ingress_disabled_by_default(self):
        docs = _helm_template()
        assert _find(docs, "Ingress") is None

    def test_ingress_enabled(self):
        docs = _helm_template(
            "--set",
            "ingress.enabled=true",
            "--set",
            "ingress.className=nginx",
            "--set",
            "ingress.hosts[0].host=cogtrix.example.com",
            "--set",
            "ingress.hosts[0].paths[0].path=/",
            "--set",
            "ingress.hosts[0].paths[0].pathType=Prefix",
        )
        ing = _find(docs, "Ingress")
        assert ing is not None
        assert ing["spec"]["ingressClassName"] == "nginx"
        rule = ing["spec"]["rules"][0]
        assert rule["host"] == "cogtrix.example.com"
        assert rule["http"]["paths"][0]["path"] == "/"
        assert rule["http"]["paths"][0]["pathType"] == "Prefix"

    def test_ingress_tls(self):
        docs = _helm_template(
            "--set",
            "ingress.enabled=true",
            "--set",
            "ingress.tls[0].secretName=cogtrix-tls",
            "--set",
            "ingress.tls[0].hosts[0]=cogtrix.example.com",
        )
        ing = _find(docs, "Ingress")
        tls = ing["spec"]["tls"][0]
        assert tls["secretName"] == "cogtrix-tls"
        assert tls["hosts"] == ["cogtrix.example.com"]


class TestHPA:
    """HorizontalPodAutoscaler."""

    def test_hpa_disabled_by_default(self):
        docs = _helm_template()
        assert _find(docs, "HorizontalPodAutoscaler") is None

    def test_hpa_enabled(self):
        docs = _helm_template(
            "--set",
            "autoscaling.enabled=true",
            "--set",
            "autoscaling.minReplicas=2",
            "--set",
            "autoscaling.maxReplicas=20",
        )
        hpa = _find(docs, "HorizontalPodAutoscaler")
        assert hpa is not None
        assert hpa["spec"]["minReplicas"] == 2
        assert hpa["spec"]["maxReplicas"] == 20

    def test_replicas_omitted_when_hpa_enabled(self):
        docs = _helm_template("--set", "autoscaling.enabled=true")
        deploy = _find(docs, "Deployment")
        assert "replicas" not in deploy["spec"]

    def test_hpa_metrics(self):
        docs = _helm_template(
            "--set",
            "autoscaling.enabled=true",
            "--set",
            "autoscaling.targetCPUUtilizationPercentage=60",
            "--set",
            "autoscaling.targetMemoryUtilizationPercentage=75",
        )
        hpa = _find(docs, "HorizontalPodAutoscaler")
        metrics = hpa["spec"]["metrics"]
        cpu = next(m for m in metrics if m["resource"]["name"] == "cpu")
        mem = next(m for m in metrics if m["resource"]["name"] == "memory")
        assert cpu["resource"]["target"]["averageUtilization"] == 60
        assert mem["resource"]["target"]["averageUtilization"] == 75


class TestServiceAccount:
    """ServiceAccount toggling."""

    def test_created_by_default(self):
        docs = _helm_template()
        assert _find(docs, "ServiceAccount") is not None

    def test_not_created_when_disabled(self):
        docs = _helm_template("--set", "serviceAccount.create=false")
        assert _find(docs, "ServiceAccount") is None

    def test_uses_default_sa_when_disabled(self):
        docs = _helm_template("--set", "serviceAccount.create=false")
        deploy = _find(docs, "Deployment")
        assert deploy["spec"]["template"]["spec"]["serviceAccountName"] == "default"

    def test_custom_name(self):
        docs = _helm_template("--set", "serviceAccount.name=my-sa")
        deploy = _find(docs, "Deployment")
        sa = _find(docs, "ServiceAccount")
        assert deploy["spec"]["template"]["spec"]["serviceAccountName"] == "my-sa"
        assert sa["metadata"]["name"] == "my-sa"


class TestServiceAccountRBAC:
    """ServiceAccount RBAC toggling."""

    def test_rbac_created_by_default(self):
        docs = _helm_template()
        role = _find(docs, "Role")
        binding = _find(docs, "RoleBinding")
        assert role is not None
        assert binding is not None
        assert role["metadata"]["name"] == "test-release-cogtrix"
        assert binding["metadata"]["name"] == "test-release-cogtrix"

    def test_rbac_rules_are_read_only(self):
        docs = _helm_template()
        role = _find(docs, "Role")
        assert role is not None
        rules = role["rules"]
        assert len(rules) == 2

        configmap_rule = next(rule for rule in rules if "configmaps" in rule["resources"])
        secret_rule = next(rule for rule in rules if "secrets" in rule["resources"])
        assert configmap_rule["verbs"] == ["get"]
        assert secret_rule["verbs"] == ["get"]
        assert configmap_rule["resourceNames"] == ["test-release-cogtrix-config"]
        assert secret_rule["resourceNames"] == ["test-release-cogtrix"]

    def test_rbac_not_created_when_disabled(self):
        docs = _helm_template("--set", "serviceAccount.rbac.enabled=false")
        assert _find(docs, "Role") is None
        assert _find(docs, "RoleBinding") is None


class TestPersistenceModes:
    """Stateful (PVC) vs Stateless (emptyDir)."""

    def test_pvc_created_by_default(self):
        docs = _helm_template()
        assert _find(docs, "PersistentVolumeClaim") is not None

    def test_pvc_omitted_when_disabled(self):
        docs = _helm_template("--set", "persistence.enabled=false")
        assert _find(docs, "PersistentVolumeClaim") is None

    def test_emptydir_used_when_disabled(self):
        docs = _helm_template("--set", "persistence.enabled=false")
        deploy = _find(docs, "Deployment")
        vols = deploy["spec"]["template"]["spec"]["volumes"]
        data_vol = next(v for v in vols if v["name"] == "data")
        assert "emptyDir" in data_vol

    def test_existing_claim_used(self):
        docs = _helm_template("--set", "persistence.existingClaim=my-old-pvc")
        deploy = _find(docs, "Deployment")
        vols = deploy["spec"]["template"]["spec"]["volumes"]
        data_vol = next(v for v in vols if v["name"] == "data")
        assert data_vol["persistentVolumeClaim"]["claimName"] == "my-old-pvc"
        # New PVC should not be created
        pvcs = _find_all(docs, "PersistentVolumeClaim")
        assert len(pvcs) == 0


class TestNetworkPolicy:
    """NetworkPolicy toggling."""

    def test_enabled_by_default(self):
        docs = _helm_template()
        np = _find(docs, "NetworkPolicy")
        assert np is not None
        assert np["spec"]["policyTypes"] == ["Ingress", "Egress"]
        assert np["spec"]["ingress"][0]["ports"][0]["port"] == 8000
        assert np["spec"]["egress"][0]["ports"][0]["port"] == 53
        assert np["spec"]["egress"][1]["ports"][0]["port"] == 443

    def test_enabled(self):
        docs = _helm_template(
            "--set",
            "networkPolicy.enabled=true",
            "--set",
            "networkPolicy.ingress[0].from[0].podSelector.matchLabels.app=cogtrix",
        )
        np = _find(docs, "NetworkPolicy")
        assert np is not None
        assert np["spec"]["policyTypes"] == ["Ingress", "Egress"]

    def test_disabled_override(self):
        docs = _helm_template("--set", "networkPolicy.enabled=false")
        assert _find(docs, "NetworkPolicy") is None


class TestPDB:
    """PodDisruptionBudget."""

    def test_disabled_by_default(self):
        docs = _helm_template()
        assert _find(docs, "PodDisruptionBudget") is None

    def test_enabled_with_min_available(self):
        docs = _helm_template(
            "--set",
            "podDisruptionBudget.enabled=true",
            "--set",
            "podDisruptionBudget.minAvailable=1",
        )
        pdb = _find(docs, "PodDisruptionBudget")
        assert pdb["spec"]["minAvailable"] == 1


class TestServiceMonitor:
    """Prometheus ServiceMonitor."""

    def test_disabled_by_default(self):
        docs = _helm_template()
        assert _find(docs, "ServiceMonitor") is None

    def test_enabled(self):
        docs = _helm_template(
            "--set",
            "serviceMonitor.enabled=true",
            "--set",
            "serviceMonitor.interval=15s",
        )
        sm = _find(docs, "ServiceMonitor")
        assert sm["spec"]["endpoints"][0]["interval"] == "15s"


class TestWAHA:
    """WAHA WhatsApp sidecar."""

    def test_disabled_by_default(self):
        docs = _helm_template()
        deploy = _find(docs, "Deployment")
        names = [c["name"] for c in deploy["spec"]["template"]["spec"]["containers"]]
        assert "waha" not in names

    def test_enabled(self):
        docs = _helm_template("--set", "waha.enabled=true")
        deploy = _find(docs, "Deployment")
        containers = deploy["spec"]["template"]["spec"]["containers"]
        names = [c["name"] for c in containers]
        assert "waha" in names
        waha = next(c for c in containers if c["name"] == "waha")
        assert waha["image"] == "devlikeapro/waha:gows-2026.4.2"

    def test_digest_pinning(self):
        docs = _helm_template(
            "--set",
            "waha.enabled=true",
            "--set",
            "waha.image.digest=sha256:abc123",
        )
        deploy = _find(docs, "Deployment")
        waha = next(
            c for c in deploy["spec"]["template"]["spec"]["containers"] if c["name"] == "waha"
        )
        assert waha["image"] == "devlikeapro/waha@sha256:abc123"

    def test_custom_tag(self):
        docs = _helm_template(
            "--set",
            "waha.enabled=true",
            "--set",
            "waha.image.tag=custom-tag",
        )
        deploy = _find(docs, "Deployment")
        waha = next(
            c for c in deploy["spec"]["template"]["spec"]["containers"] if c["name"] == "waha"
        )
        assert waha["image"] == "devlikeapro/waha:custom-tag"


class TestLinkerd:
    """Linkerd service mesh integration."""

    def test_disabled_by_default(self):
        docs = _helm_template()
        deploy = _find(docs, "Deployment")
        annotations = deploy["spec"]["template"]["metadata"].get("annotations") or {}
        assert "linkerd.io/inject" not in annotations
        assert _find(docs, "ServiceProfile") is None

    def test_inject_annotation_present_when_enabled(self):
        docs = _helm_template(
            "--set",
            "linkerd.enabled=true",
            "--set",
            "linkerd.inject=true",
        )
        deploy = _find(docs, "Deployment")
        annotations = deploy["spec"]["template"]["metadata"]["annotations"]
        assert annotations["linkerd.io/inject"] == "enabled"

    def test_no_inject_when_disabled_but_linkerd_enabled(self):
        docs = _helm_template(
            "--set",
            "linkerd.enabled=true",
            "--set",
            "linkerd.inject=false",
        )
        deploy = _find(docs, "Deployment")
        annotations = deploy["spec"]["template"]["metadata"].get("annotations") or {}
        assert "linkerd.io/inject" not in annotations

    def test_service_profile_created_when_enabled(self):
        docs = _helm_template(
            "--set",
            "linkerd.enabled=true",
            "--set",
            "linkerd.profile.enabled=true",
        )
        sp = _find(docs, "ServiceProfile")
        assert sp is not None
        assert sp["spec"]["routes"][0]["name"] == "api"

    def test_service_profile_has_timeouts(self):
        docs = _helm_template(
            "--set",
            "linkerd.enabled=true",
            "--set",
            "linkerd.profile.enabled=true",
            "--set",
            "linkerd.profile.timeouts.api=60s",
            "--set",
            "linkerd.profile.timeouts.health=3s",
        )
        sp = _find(docs, "ServiceProfile")
        routes = {r["name"]: r for r in sp["spec"]["routes"]}
        assert routes["api"]["timeout"] == "60s"
        assert routes["health"]["timeout"] == "3s"


class TestExtraValues:
    """extraEnv, extraVolumes, nodeSelector, tolerations, affinity."""

    def test_extra_env(self):
        docs = _helm_template(
            "--set",
            "extraEnv[0].name=MY_VAR",
            "--set",
            "extraEnv[0].value=my_value",
        )
        deploy = _find(docs, "Deployment")
        env = deploy["spec"]["template"]["spec"]["containers"][0]["env"]
        my_var = next((e for e in env if e["name"] == "MY_VAR"), None)
        assert my_var is not None
        assert my_var["value"] == "my_value"

    def test_extra_volume_mounts(self):
        docs = _helm_template(
            "--set",
            "extraVolumes[0].name=extra-vol",
            "--set",
            "extraVolumes[0].emptyDir={}",
            "--set",
            "extraVolumeMounts[0].name=extra-vol",
            "--set",
            "extraVolumeMounts[0].mountPath=/extra",
        )
        deploy = _find(docs, "Deployment")
        vols = deploy["spec"]["template"]["spec"]["volumes"]
        mounts = deploy["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]
        assert any(v["name"] == "extra-vol" for v in vols)
        assert any(m["name"] == "extra-vol" and m["mountPath"] == "/extra" for m in mounts)

    def test_node_selector(self):
        docs = _helm_template("--set", "nodeSelector.disk=ssd")
        deploy = _find(docs, "Deployment")
        assert deploy["spec"]["template"]["spec"]["nodeSelector"]["disk"] == "ssd"

    def test_tolerations(self):
        docs = _helm_template(
            "--set",
            "tolerations[0].key=dedicated",
            "--set",
            "tolerations[0].operator=Equal",
            "--set",
            "tolerations[0].value=ai",
            "--set",
            "tolerations[0].effect=NoSchedule",
        )
        deploy = _find(docs, "Deployment")
        tol = deploy["spec"]["template"]["spec"]["tolerations"][0]
        assert tol["key"] == "dedicated"
        assert tol["value"] == "ai"


class TestImage:
    """Image repository and tag handling."""

    def test_default_image(self):
        docs = _helm_template()
        deploy = _find(docs, "Deployment")
        image = deploy["spec"]["template"]["spec"]["containers"][0]["image"]
        assert image == "northlandpositronics/cogtrix:0.2.6"

    def test_custom_tag(self):
        docs = _helm_template("--set", "image.tag=latest")
        deploy = _find(docs, "Deployment")
        image = deploy["spec"]["template"]["spec"]["containers"][0]["image"]
        assert image == "northlandpositronics/cogtrix:latest"

    def test_custom_repository(self):
        docs = _helm_template("--set", "image.repository=myreg/cogtrix")
        deploy = _find(docs, "Deployment")
        image = deploy["spec"]["template"]["spec"]["containers"][0]["image"]
        assert image == "myreg/cogtrix:0.2.6"


class TestLabels:
    """Consistency of labels across resources."""

    def setup_method(self):
        self.docs = _helm_template()

    def test_selector_labels_match(self):
        deploy = _find(self.docs, "Deployment")
        svc = _find(self.docs, "Service")
        deploy_sel = deploy["spec"]["selector"]["matchLabels"]
        svc_sel = svc["spec"]["selector"]
        assert deploy_sel["app.kubernetes.io/name"] == svc_sel["app.kubernetes.io/name"]
        assert deploy_sel["app.kubernetes.io/instance"] == svc_sel["app.kubernetes.io/instance"]

    def test_pod_labels_match_selector(self):
        deploy = _find(self.docs, "Deployment")
        pod_labels = deploy["spec"]["template"]["metadata"]["labels"]
        selector = deploy["spec"]["selector"]["matchLabels"]
        for k, v in selector.items():
            assert pod_labels.get(k) == v


class TestAllFeaturesEnabled:
    """Smoke test with every optional feature turned on."""

    def setup_method(self):
        self.docs = _helm_template(
            "--set",
            "secrets.enabled=true",
            "--set",
            "secrets.jwtSecret=test",
            "--set",
            "config.enabled=true",
            "--set",
            "ingress.enabled=true",
            "--set",
            "autoscaling.enabled=true",
            "--set",
            "podDisruptionBudget.enabled=true",
            "--set",
            "networkPolicy.enabled=true",
            "--set",
            "serviceMonitor.enabled=true",
            "--set",
            "waha.enabled=true",
        )

    def test_no_errors(self):
        # If helm template exited non-zero, setup_method would have raised.
        assert len(self.docs) >= 10

    def test_all_expected_kinds_present(self):
        kinds = {d["kind"] for d in self.docs}
        expected = {
            "Secret",
            "ConfigMap",
            "PersistentVolumeClaim",
            "Service",
            "Deployment",
            "Ingress",
            "HorizontalPodAutoscaler",
            "PodDisruptionBudget",
            "NetworkPolicy",
            "ServiceMonitor",
            "ServiceAccount",
            "Pod",
        }
        for k in expected:
            assert k in kinds, f"Missing kind: {k}"


class TestHelmTestHook:
    """The helm test Pod."""

    def test_test_hook_present(self):
        docs = _helm_template()
        test_pods = [
            d
            for d in docs
            if d.get("kind") == "Pod"
            and "test" in d.get("metadata", {}).get("annotations", {}).get("helm.sh/hook", "")
        ]
        assert len(test_pods) == 1
        pod = test_pods[0]
        assert pod["spec"]["containers"][0]["image"] == "busybox:1.37"
        assert any(
            "test-release-cogtrix:8000" in arg
            for arg in pod["spec"]["containers"][0].get("command", [])
        )


if __name__ == "__main__":
    # Run with pytest if available, otherwise a simple runner.
    try:
        import pytest

        sys.exit(pytest.main([__file__, "-v"]))
    except ImportError:
        print("pytest not installed; running basic assertions directly...")
        for cls_name, cls in list(globals().items()):
            if cls_name.startswith("Test"):
                instance = cls()
                if hasattr(instance, "setup_method"):
                    instance.setup_method()
                for method_name in dir(instance):
                    if method_name.startswith("test_"):
                        print(f"  {cls_name}.{method_name} ...", end=" ")
                        try:
                            getattr(instance, method_name)()
                            print("OK")
                        except Exception as exc:
                            print(f"FAIL: {exc}")
                            sys.exit(1)
        print("\nAll tests passed.")
