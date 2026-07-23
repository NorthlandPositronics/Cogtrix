# API Rate Limiting — Operator Reference

This document covers the application-tier rate limiter built into the Cogtrix
API server. It is the runtime ops reference for the changes shipped under
issue [#1879](https://github.com/NorthlandPositronics/Cogtrix/issues/1879)
(Slices A, B, and the SlowAPI-global follow-up).

## Two-layer model

The application-tier limiter is one half of a defence-in-depth pair:

| Layer | Purpose | Where it runs |
|---|---|---|
| **Ingress tier** (nginx / envoy / Cloudflare) | Blunt abuse mitigation — drop floods before they reach the app | Reverse proxy or CDN, ops-managed |
| **Application tier** (this document) | Business semantics — `auth_register` cap, per-route specifics, user-fingerprint quotas | Cogtrix API process |

Operators should configure **both**. The ingress layer protects the app from
DDoS / scraper bursts the app should never see; the application layer enforces
the rules that need request context the ingress doesn't have (route name,
authenticated user, body shape). Relying on either alone leaves a gap.

This document is the operator reference for the **application tier only**.
For ingress-tier config see your reverse-proxy documentation.

## Configuration sources, in precedence order

1. **Environment variables** — highest precedence. Standard mechanism for
   Kubernetes / Docker / systemd deployments.
2. **`.cogtrix.yaml`** under the `api:` block. Lower precedence than env
   vars so a one-shot ops tweak can override the committed config without
   editing the file.
3. **Built-in defaults** baked into `src/config.py:APIConfig`. Always
   safe single-node values.

The full precedence chain runs at app startup. A malformed value in any
layer surfaces as a `ConfigError` / `RuntimeError` at startup time, not
as a 500 on the first request.

## Per-route rate limits

The four per-route limits operators care about most:

| Route | Config key | Default |
|---|---|---|
| `POST /api/v1/auth/register` | `api.rate_limits.auth_register` | `3/hour` |
| `POST /api/v1/auth/login` | `api.rate_limits.auth_login` | `5/minute` |
| `POST /api/v1/auth/refresh` | `api.rate_limits.auth_refresh` | `5/minute` |
| `POST /api/v1/auth/saml/acs` | `api.rate_limits.saml_acs` | `5/minute` |
| (everything else) | `api.rate_limits.default` | `120/minute` |

### YAML

```yaml
api:
  rate_limits:
    default: "1000/minute"
    auth_register: "100/hour"
    auth_login: "30/minute"
    auth_refresh: "60/minute"
    saml_acs: "30/minute"
```

### Environment variables

The route name uppercases into the var name:

```bash
export COGTRIX_RATE_LIMIT_DEFAULT="1000/minute"
export COGTRIX_RATE_LIMIT_AUTH_REGISTER="100/hour"
export COGTRIX_RATE_LIMIT_AUTH_LOGIN="30/minute"
```

### Spec format

SlowAPI-style `"<N>/<window>"`. Windows accepted (case-insensitive,
optional trailing `s`):

- `second` / `s`
- `minute` / `m`
- `hour` / `h`
- `day` / `d`

Examples: `"3/hour"`, `"100/minute"`, `"1/second"`, `"500/day"`,
`"  5 / m  "`.

Invalid specs raise `ConfigError` at startup — they never reach a request.

## Trusted reverse-proxy CIDRs

By default the rate limiter buckets requests by the TCP peer's IP. Behind
a load balancer this means **every request** comes from the LB and the
entire user population collapses into one bucket. To recover the real
client IP, list your LB / ingress CIDRs:

### YAML

```yaml
api:
  trusted_proxy_cidrs:
    - "10.0.0.0/8"        # K8s pod network
    - "172.16.0.0/12"     # VPC private range
```

### Environment variable

```bash
export COGTRIX_TRUSTED_PROXY_CIDRS="10.0.0.0/8,172.16.0.0/12"
```

When a trusted CIDR list is configured, `_client_key` walks the
`X-Forwarded-For` chain right-to-left honouring the allowlist. Untrusted
hops can't spoof their way into a fresh bucket — see the comment block
on `src/api/rate_limit.py:_client_key` for the full algorithm rationale.

## Multi-replica deployments — opt-in Redis backend

The default rate limiter keeps its sliding window in **per-process memory**.
That's correct for single-node deployments but jitters under horizontal
scaling — N replicas independently enforce their own slice of the
configured limit, so the effective cap per IP is roughly N× the
configured value.

To share the counter across replicas, point the limiter at Redis:

### Install the optional dependency

```bash
pip install cogtrix[api,redis]
```

### YAML

```yaml
api:
  redis_url: "redis://redis.svc.cluster.local:6379/0"
```

### Environment variable (takes precedence over YAML)

```bash
export COGTRIX_REDIS_URL="redis://user:secret@redis.svc:6379/0"
```

### What you get

Both code paths use the shared backend:

- **Per-route limits** (`auth_register` etc.) — `MovingWindowRateLimiter`
  over `limits.storage.RedisStorage`.
- **SlowAPI global blunt guard** (`120/minute` default) — `Limiter`
  rebuilt at startup with `storage_uri=<your-url>`.

The startup log names the active backend with any inline password
redacted:

```
Rate-limit backend: shared counter at redis://user:***@redis.svc:6379/0
```

### When the env var is set but the package isn't installed

You get a clear `RuntimeError` at startup pointing at the install extra,
rather than a silent 500 on the first request:

```
COGTRIX_REDIS_URL / api.redis_url is set but the 'redis' package is
not installed. Install with: pip install cogtrix[api,redis]
```

### Behaviour during a Redis outage

Rate-limit enforcement **fails open** — a transient backend error logs a
WARNING and lets the request through. The user-visible alternative (5xx
on every request until Redis recovers) was judged worse than the brief
window where rate limits don't enforce. The log entries are searchable
on `Rate-limit backend hit() raised`.

## Common operational tasks

### Lift the registration cap for a load test

Without restarting the API:

```bash
export COGTRIX_RATE_LIMIT_AUTH_REGISTER="10000/hour"
# (re)deploy the API, env var picks up at startup
```

### Disable the application-tier limiter entirely (development only)

Not currently supported as a runtime knob. The ops practice is to set
each route's limit to a very high value (e.g. `"1000000/day"`) — the
limiter still runs but never trips.

### Inspect the active backend

The startup log line names the backend. There is no runtime introspection
endpoint by design — exposing rate-limit state would itself need a rate
limit.

### Reset the counter after a deploy

Application-tier `MemoryStorage` resets on every process restart, which
already happens during a rolling deploy. The Redis backend is
intentionally NOT reset on startup — wiping it would nuke counters owned
by other replicas in their sliding window. If you need a hard Redis
reset, flush the relevant DB out of band:

```bash
redis-cli -h redis.svc -n 0 FLUSHDB
```

## Out of scope for this layer

- **DDoS / volumetric attack mitigation** — handle at the ingress tier.
- **Geographic blocking** — handle at the CDN / WAF tier.
- **Per-user (post-login) fingerprint keying** — separate failure mode
  not yet implemented; tracked as a follow-up on
  [#1879](https://github.com/NorthlandPositronics/Cogtrix/issues/1879).
- **Custom strategies** (token bucket, fixed window) — the limiter uses
  a moving-window strategy throughout; switching strategies requires a
  code change.

## Related source files

- `src/api/rate_limit.py` — `_enforce_per_route`, `configure_rate_limit_backend`,
  `_client_key`
- `src/config.py:APIConfig` — config schema and validation
- `src/api/app.py` — startup wiring (precedence resolution + backend install)
- `tests/test_api_rate_limit_config.py` — Slice A regression tests
- `tests/test_api_rate_limit_redis_backend.py` — Slice B + follow-up tests
