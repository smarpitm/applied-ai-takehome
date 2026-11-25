---
title: SLO Targets and Error Budgets
topic: operations
---

# SLO Targets and Error Budgets

## Latency and Availability Objectives

The NimbusStack Query API enforces a monthly availability SLO of 99.95% (21.9 minutes maximum downtime per 30-day window). Warm cache-hit retrieval p50 latency target is 18 ms and p95 latency target is 45 ms; cold object-store partition reads allow a p95 ceiling of 220 ms. End-to-end generation latency budgets depend on the selected LLM router tier, targeting under 900 ms p95 for fast hosted models.

## Error Budget Burn Policy

When error budget burn exceeds 5% over a rolling 6-hour window, feature deployments freeze automatically until the on-call reliability engineer completes root-cause remediation. Canary rollbacks trigger if 5xx error rate exceeds 0.5% for 3 consecutive minutes or if p95 retrieval latency breaches 350 ms across 500 sampled queries.

## Health Probes and Graceful Degradation

Readiness checks verify LanceDB manifest readability and local embedding model warm state before accepting traffic on port 8000. If the primary hosted LLM API hits HTTP 429 rate limits or times out after 10 seconds, the router fails over automatically to the configured local fallback model and emits fallback_triggered=true inside structured JSON telemetry.
