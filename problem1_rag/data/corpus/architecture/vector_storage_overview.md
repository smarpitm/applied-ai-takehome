---
title: Vector Storage Architecture Overview
topic: architecture
---

# Vector Storage Architecture Overview

## Columnar Storage Format

NimbusStack stores embedding vectors in the disk-native Lance columnar format backed by AWS S3 in us-east-1. Each vector uses 384 float32 dimensions (1,536 bytes uncompressed) generated locally by BAAI/bge-small-en-v1.5. Using 384 dimensions instead of 1,536 dimensions reduces raw vector storage footprint and S3 GET bandwidth by 4x at 10 million vectors. Columnar data files separate vector columns from payload text so metadata-only scans avoid reading vector blobs into page cache. Furthermore, zero-copy Arrow memory mapping allows embedded workers to stream slices straight from disk blocks without serializing intermediate protobuf frames across network sockets.

## IVF-PQ Partitioning and Memory Budget

For lightly-queried indexes above 100,000 vectors, NimbusStack enables IVF-PQ indexing with 256 partitions and 48 subvectors (8 bits per code), compressing each 384-dimensional vector from 1,536 bytes down to 48 bytes. Only the matching IVF centroids and requested partitions are fetched on demand, avoiding always-on memory-resident HNSW pods. At 10M vectors, raw PQ codes occupy only 480 MB total, allowing coarse routing tables under 4 MB to remain resident in worker RAM while cold partitions rest on S3. Operators accept two primary trade-offs: cold partition fetches incur higher p95 latency than memory-pinned HNSW graphs, and lossy Product Quantization typically trades 1.5% to 3% Recall@5 for roughly 32x vector compression.

## Deterministic Upsert and Fragment Versioning

Every chunk row receives a deterministic chunk_id computed as the SHA-256 digest of normalized source_id concatenated with chunk_index. Before writing updated chunks, the ingestor compares the stored doc_hash (SHA-256 of raw document text) against the incoming document hash. Unchanged files skip re-embedding entirely. Modified files delete prior rows matching source_id and then execute LanceDB merge_insert keyed on chunk_id, guaranteeing zero duplicate vectors even if a document shrinks in chunk count.

## When to Migrate Back to Managed Vector Databases

While embedded LanceDB dominates lightly-queried workloads below 5 queries per second (QPS), teams should migrate back to a dedicated managed service such as Pinecone or Qdrant Cloud when sustained traffic exceeds 50 QPS with strict sub-20 ms p99 SLAs, when multi-region active-active write replication under 100 ms becomes mandatory, or when streaming updates exceed 1,000 vector mutations per second requiring continuous HNSW background indexing.
