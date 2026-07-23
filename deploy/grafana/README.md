# Cogtrix Grafana Dashboard

Pre-built Grafana dashboard for Cogtrix monitoring.

## Metrics

The dashboard visualizes these Prometheus metrics:

| Panel | Metric | Description |
|-------|--------|-------------|
| API Request Rate | `cogtrix_api_requests_total` | API requests by route, method, status |
| LLM Latency | `cogtrix_llm_latency_seconds` | P95/P99 latency by provider/model |
| LLM Requests | `cogtrix_llm_requests_total` | LLM requests by provider/model |
| Tool Calls | `cogtrix_tool_calls_total` | Tool calls by name |
| Active Sessions | `cogtrix_sessions_active` | Active (non-archived) sessions |
| Error Rate | `cogtrix_api_requests_total` (5xx) | 5xx error rate |
| DB Connections | `cogtrix_db_connections` | Active/idle pool connections |

## Import Instructions

1. In Grafana, go to **Dashboards** → **New** → **Import**
2. Upload `cogtrix-dashboard.json`
3. Configure variables:
   - **datasource**: Prometheus data source
   - **environment**: `production`, `staging`, or `development`
   - **namespace**: Your Kubernetes namespace (default: `default`)

## Variables

- `datasource`: Prometheus data source (Grafana variable)
- `environment`: Environment filter (production/staging/development)
- `namespace`: Kubernetes namespace filter

## Alert Annotations

The dashboard includes alert annotations for error rate spikes. Configure alerts in Grafana:

- **Error Rate Alert**: Trigger when 5xx rate exceeds threshold for 5 minutes
- **Latency Alert**: Trigger when P99 latency exceeds threshold for 5 minutes

## Screenshots

Include screenshots in documentation showing:
- Dashboard overview
- Key panels with sample data
- Alert configuration
