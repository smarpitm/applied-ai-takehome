---
title: Zero-Trust IAM and mTLS Policy
topic: security
---

# Zero-Trust IAM and mTLS Policy

## Workload Identity and Certificate TTL

All internal RPC services authenticate using SPIFFE workload identities and mutual TLS (mTLS). Short-lived X.509 certificates are issued with a strict 24-hour time-to-live (TTL) and rotated automatically at 12 hours by the node agent. Static long-lived API tokens are prohibited in production clusters. Every workload certificate encodes namespace, cluster region, and service role inside Subject Alternative Name (SAN) URIs so downstream proxies enforce identity-aware RBAC without shared secrets.

## Audit Trail Retention

Every authorization decision is emitted as a structured JSON event to an immutable append-only sink. Hot audit logs remain searchable for 90 days and are archived to S3 Glacier Deep Archive for 7 years to meet SOC 2 Type II compliance controls. Log payloads record request_id, caller SPIFFE ID, policy version, decision latency, and allow/deny reason while redacting bearer credentials and tenant document bodies.

## Break-Glass Emergency Access

During severity-1 control-plane outages where the primary identity issuer is unreachable, two senior SREs must jointly approve a hardware-backed YubiKey quorum to mint a 60-minute break-glass role. Break-glass sessions trigger real-time PagerDuty notifications and mandatory post-incident session replay review within 24 hours.

## Cross-Account Access Boundaries

Production vector buckets deny cross-account assume-role chains unless explicitly whitelisted by AWS Organizations SCPs with external ID and source VPC endpoint conditions attached.
