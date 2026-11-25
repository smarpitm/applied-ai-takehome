# Problem 1 — Monthly Cost Comparison Across Scale (100K / 1M / 10M Vectors)

| Vectors | Raw 384d (GB) | LanceDB Stored (GB) | LanceDB S3+Compute ($/mo) | Self-Hosted Qdrant ($/mo) | Pinecone Serverless ($/mo) | Pinecone Pod ($/mo) | Savings vs Pod |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100K | 0.143 | 0.072 | $15.49 | $31.17 | $42.84 | $70.00 | 77.9% |
| 1M | 1.431 | 0.719 | $15.51 | $31.17 | $129.00 | $70.00 | 77.8% |
| 10M | 14.305 | 7.190 | $30.85 | $61.86 | $520.15 | $700.00 | 95.6% |

## Explicit Pricing & Workload Assumptions
- **Embeddings:** `BAAI/bge-small-en-v1.5`, `384` float32 dims (`1536` bytes/vector raw; 4x smaller than 1536-dim `6,144` bytes).
- **LanceDB Compression & Payload:** IVF-PQ at `48` bytes/vector + `720` bytes compressed text/metadata + `4` bytes index overhead.
- **Workload:** Lightly-queried index at `0.2` QPS (~`518,400` queries/month, `1.5` S3 GETs/query at `$0.0004/1K` GETs, S3 Standard at `$0.023/GB-mo`).
- **Compute:** AWS `t4g.small` (`$15.18/mo`) up to 1M vectors, `t4g.medium` (`$30.37/mo`) at 10M vectors (zero idle RAM scaling per vector; scaling down to serverless Lambda/Fargate cuts low-QPS idle compute further).
- **Managed DB Comparison:** Pinecone Pod (`s1.x1` / `p1.x1` at `$70.0/pod/mo`, ~1M 384-dim vectors/pod) and Pinecone Serverless (`$0.33/GB-mo` + `$8.25/1M RUs`).
- **Self-Hosted Qdrant Bonus:** Dedicated EC2 (`t4g.medium`/`t4g.large`) + EBS `gp3` (`$0.08/GB-mo`) with `on_disk` memmap + `int8` scalar quantization.
