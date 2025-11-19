# EmbedKitV2 (Week 1)

Clean, modern embedding API for rapid benchmarking and development without legacy constraints.

## Quick Start

```
import EmbedKitV2

let manager = ModelManager()
let model = try await manager.loadAppleModel() // Week 1 returns a mock

let emb = try await model.embed("Hello world")
print(emb.dimensions) // 384

let batch = try await model.embedBatch(["a","b","c"], options: .init())
print(batch.count)
```

## Metrics

`ModelMetrics` provides:
- Totals: requests, tokens
- Latency: average, p50, p95, p99
- Throughput: tokens/sec
- Memory: per-snapshot (best-effort on Darwin)
- Histograms: latency, token counts (sliding window)

Reset metrics:
```
try await manager.resetMetrics(for: model.id)
```

## Scripts

Use the helper to keep caches local and avoid sandbox issues:

```
./Scripts/spm_v2.sh build
./Scripts/spm_v2.sh build-tests
./Scripts/spm_v2.sh test --filter Week1IntegrationTests
```

If you pass a bare class name for `--filter`, it auto-prefixes `EmbedKitV2Tests.`

## Notes

- Week 1 model is a deterministic mock for fast validation.
- Tokenizer is simple but honors truncation/padding flags.
- CoreML backend and advanced batching land in Week 2.

