---
title: API Idempotency Keys and Retry Policy
topic: api_reference
---

# API Idempotency Keys and Retry Policy

## Idempotency Key Semantics

Mutating POST endpoints accept an Idempotency-Key HTTP header up to 128 ASCII characters. Keys remain valid for 24 hours; repeating the same key with an identical payload replays the cached 200 or 201 JSON response, while reusing a key with a mismatched body hash returns HTTP 409 Conflict. Clients should generate UUIDv4 keys per logical action and avoid sharing keys across distinct tenant workspaces.

## Exponential Backoff with Full Jitter

Clients receiving HTTP 429 or 503 must honor the Retry-After response header when present, otherwise apply exponential backoff starting at 250 ms base delay capped at 16 seconds across a maximum of 5 attempts with full random jitter. Unbounded tight retry loops are flagged by edge WAF rules and temporarily throttled for 60 seconds.

## Request Timeouts and Streaming

Synchronous /query calls enforce a 30-second server deadline. Requests supplying invalid k values outside 1 to 50 or unsupported metadata keys fail fast with HTTP 422 validation errors detailing exact field constraints.
