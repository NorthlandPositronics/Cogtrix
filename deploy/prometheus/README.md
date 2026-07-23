# Prometheus Alert Rules for Cogtrix

This directory contains Prometheus alert rules for monitoring Cogtrix operations.

## Overview

The alert rules are defined in `alerts.rules.yml` and monitor:

1. **API Error Rate** - Alerts when API error rate exceeds 5% over 5 minutes
2. **LLM Latency** - Alerts when p99 latency exceeds 10 seconds over 5 minutes
3. **Database Pool** - Alerts when DB pool usage exceeds 80%
4. **Active Sessions** - Alerts when active sessions exceed 90% of capacity
5. **Stripe Webhooks** - Alerts on any Stripe webhook failures

## Deployment

### Prerequisites

- Prometheus 2.40+ with rule evaluation enabled
- Access to modify Prometheus configuration

### Steps

1. **Validate the rules file** (requires Docker):
   ```bash
   docker run --rm -v $(pwd)/deploy/prometheus:/rules prom/prometheus promtool check rules /rules/alerts.rules.yml
   ```

2. **Copy the rules file to your Prometheus server**:
   ```bash
   scp deploy/prometheus/alerts.rules.yml prometheus-server:/etc/prometheus/rules/
   ```

3. **Update Prometheus configuration** (`prometheus.yml`):
   ```yaml
   rule_files:
     - "/etc/prometheus/rules/alerts.rules.yml"
   
   scrape_configs:
     # ... your existing scrape configs
   ```

4. **Reload Prometheus configuration**:
   ```bash
   # Method 1: Send SIGHUP to Prometheus process
   kill -HUP $(pgrep prometheus)
   
   # Method 2: Use Prometheus reload endpoint (if configured)
   curl -X POST http://localhost:9090/-/reload
   ```

5. **Verify rules are loaded**:
   ```bash
   # Check via Prometheus UI
   curl http://localhost:9090/api/v1/rules | jq .
   
   # Or check alerts
   curl http://localhost:9090/api/v1/alerts | jq .
   ```

## Alert Categories

### Critical Severity
- **API Error Rate** - Immediate impact on service availability
- **LLM Latency** - Degraded performance affecting users
- **Stripe Webhooks** - Payment processing failures

### Warning Severity
- **Database Pool** - Potential bottleneck in data access
- **Active Sessions** - Capacity planning indicator

## Customization

Adjust thresholds and durations based on your operational requirements:

```yaml
# Example: Increase tolerance for API errors during maintenance
expr: sum(rate(cogtrix_api_requests_total{status=~"5.."}[5m])) / sum(rate(cogtrix_api_requests_total[5m])) > 0.10
for: 10m
```

## Monitoring

Once deployed, verify alerts are firing by checking the Prometheus UI at `/alerts` or using the API.

## Troubleshooting

### Rules not loading
- Check Prometheus logs for syntax errors
- Verify file permissions on the rules file
- Ensure the rule file path is correct in `prometheus.yml`

### Alerts not firing
- Verify metric names match what's exported by Cogtrix
- Check that the metrics are being scraped by Prometheus
- Adjust the evaluation interval if needed

## Related Documentation

- [Prometheus Alerting Rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting-rules/)
- [PromQL Documentation](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Cogtrix Monitoring Guide](https://internal.cogtrix/docs/monitoring)
