# Cogtrix Helm Chart

A production-ready Helm chart for deploying [Cogtrix](https://github.com/NorthlandPositronics/Cogtrix) on Kubernetes.

## Prerequisites

- Kubernetes 1.32+
- Helm 3.14+
- PV provisioner support in the underlying infrastructure (if persistence is enabled)

## Installing the Chart

```bash
helm install cogtrix ./helm/cogtrix \
  --namespace cogtrix --create-namespace \
  --set secrets.enabled=true \
  --set secrets.jwtSecret="$(openssl rand -base64 32)" \
  --set secrets.openaiApiKey="sk-..."
```

## Configuration

See [`values.yaml`](values.yaml) for the full list of configurable parameters.

### Required Values

At minimum, you must provide:

- `secrets.jwtSecret` — 32+ character random string for JWT signing
- At least one LLM provider API key (`secrets.openaiApiKey`, `secrets.anthropicApiKey`, etc.)

### Quick Start

```bash
# Create namespace
kubectl create namespace cogtrix

# Install with minimal required secrets
helm install cogtrix ./helm/cogtrix \
  --namespace cogtrix \
  --set secrets.enabled=true \
  --set secrets.jwtSecret="your-random-secret" \
  --set secrets.openaiApiKey="sk-your-key"
```

### With Ingress

```bash
helm install cogtrix ./helm/cogtrix \
  --namespace cogtrix \
  --set ingress.enabled=true \
  --set ingress.className=nginx \
  --set ingress.hosts[0].host=cogtrix.example.com
```

### With Config File

```bash
helm install cogtrix ./helm/cogtrix \
  --namespace cogtrix \
  --set config.enabled=true \
  --set-file config.content=./my-cogtrix-config.yaml
```

## Health Probes

The chart configures three probes against the Cogtrix API:

| Probe | Endpoint | Purpose |
|-------|----------|---------|
| Liveness | `/api/v1/health` | Restarts pod if process is dead |
| Readiness | `/api/v1/health/ready` | Removes pod from service if DB unreachable |
| Startup | `/api/v1/health` | Gives migrations time to run before other probes |

## Persistence

By default, a 10 GiB PVC is created for `/data` because Cogtrix stores SQLite DB, vector indices, message history, and uploads on local disk.

### Stateful mode (default)

Uses a PVC for durable local storage. Sessions, users, and history survive pod restarts.

### Stateless mode

If you want ephemeral pods (no PVC), you must configure external services:

- **Database**: Set `DATABASE_URL` to an external Postgres (instead of local SQLite)
- **Sessions**: Configure `redis_url` in `.cogtrix.yaml` for cross-pod session presence
- **File storage**: Use S3/GCS if you use file upload tools

```bash
helm install cogtrix ./helm/cogtrix \
  --set persistence.enabled=false \
  --set secrets.databaseUrl="postgresql+asyncpg://user:pass@postgres:5432/cogtrix" \
  --set config.enabled=true \
  --set-file config.content=./stateless-config.yaml
```

> **Warning:** Disabling persistence without external DB means all data is ephemeral — pod restarts destroy sessions, users, and history.

## Autoscaling

> **Warning:** Cogtrix sessions are stateful. Each pod keeps live LLM state, memory, and local SQLite/vector DB. Horizontal scaling without session affinity or external shared state will break existing sessions.
>
> To safely scale you need either:
> 1. **Session affinity** (`ingress.sessionAffinity: ClientIP`) so the same user always hits the same pod
> 2. **External Postgres + Redis** for full shared session state (not just presence tracking)
> 3. **Independent agents** — run multiple identical instances that do not share sessions (e.g. one deployment per tenant, or a pool where each client pins to one instance). You must configure your own load-balancing strategy for this case.
>
> By default the chart runs with `replicaCount: 1`.

Enable HPA (only if you have addressed the state sharing requirement above):

```bash
helm upgrade cogtrix ./helm/cogtrix \
  --set autoscaling.enabled=true \
  --set autoscaling.minReplicas=2 \
  --set autoscaling.maxReplicas=10
```

## Secrets Management

> **WARNING:** Never commit a `values.yaml` (or any file) containing live secrets to version control.
> The inline `secrets` stanza is intended for **development only**.

### Option A: External Secrets Operator (recommended for production)

Configure `externalSecrets` to sync secrets from a secret manager (Vault, AWS Secrets Manager, GCP Secret Manager, Azure Key Vault, etc.):

```yaml
externalSecrets:
  enabled: true
  secretStoreRef:
    name: vault-backend
    kind: ClusterSecretStore
  data:
    - secretKey: jwt-secret
      remoteRef:
        key: cogtrix/jwt
        property: secret
    - secretKey: openai-api-key
      remoteRef:
        key: cogtrix/llm
        property: openai
```

### Option B: helm-secrets + SOPS

Encrypt secrets with Mozilla SOPS and decrypt at deploy time:

```bash
# secrets.yaml is encrypted with SOPS
helm secrets install cogtrix ./helm/cogtrix \
  -f values.yaml \
  -f secrets.yaml
```

### Option C: Sealed Secrets

Encrypt secrets into SealedSecret resources with `kubeseal`:

```bash
kubeseal --controller-namespace=kube-system \
  --format yaml < my-secret.yaml > templates/sealed-secret.yaml
```

### Option D: Inline (development only)

```bash
helm install cogtrix ./helm/cogtrix \
  --set secrets.enabled=true \
  --set secrets.jwtSecret="$(openssl rand -base64 32)" \
  --set secrets.openaiApiKey="sk-..."
```

## Security

- Runs as non-root user (UID/GID 1000)
- Read-only root filesystem
- `runAsNonRoot: true`
- `allowPrivilegeEscalation: false`
- Drops all capabilities
- NetworkPolicy is enabled by default. It allows HTTP ingress on port 8000
  from in-cluster sources and egress for DNS (UDP 53) plus outbound HTTPS
  (TCP 443). Override `networkPolicy.ingress` and `networkPolicy.egress` if
  your deployment needs tighter or broader rules.
- Creates a namespace-scoped `Role` and `RoleBinding` by default so the chart can keep service-account permissions explicit and minimal. Set `serviceAccount.rbac.enabled=false` if you do not want those RBAC objects rendered.

## Uninstalling

```bash
helm uninstall cogtrix --namespace cogtrix
```
