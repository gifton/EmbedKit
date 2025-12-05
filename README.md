# EmbedKit

A high-performance text embedding framework for Apple platforms. Generate, store, and search text embeddings on-device using CoreML, with optional GPU acceleration via Metal.

![Swift](https://img.shields.io/badge/Swift-6.0-orange.svg)
![Platforms](https://img.shields.io/badge/Platforms-iOS%2026%20|%20macOS%2026%20|%20tvOS%2026%20|%20visionOS%203-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Tests](https://img.shields.io/badge/Tests-1431%20passing-brightgreen.svg)

## Overview

EmbedKit is part of the **Vector Suite Kit (VSK)** ecosystem, providing production-ready text embedding generation for semantic search, similarity matching, and vector storage. Built with Swift 6 concurrency, automatic GPU acceleration, and comprehensive test coverage.

### Key Features

- **On-Device Inference** - CoreML backend for private, fast embeddings
- **GPU Acceleration** - Automatic Metal acceleration with transparent CPU fallback
- **Vector Storage** - Built-in similarity search via VectorIndex integration
- **Multiple Tokenizers** - WordPiece, BPE, and SentencePiece support
- **ONNX Support** - Optional ONNX Runtime backend for .onnx models
- **Production Ready** - 1431 tests, actor-based concurrency, comprehensive error handling

## Installation

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/gifton/EmbedKit.git", from: "0.1.0")
]
```

**Products:**
- `EmbedKit` - Core library with CoreML + VectorIndex + VectorAccelerate
- `EmbedKitONNX` - Optional ONNX Runtime support (adds ~50-100MB)

### Dependencies

EmbedKit integrates with the VSK ecosystem:
- **[VectorCore](https://github.com/gifton/VectorCore)** - High-performance vector operations
- **[VectorIndex](https://github.com/gifton/VectorIndex)** - Vector storage and similarity search
- **[VectorAccelerate](https://github.com/gifton/VectorAccelerate)** - GPU-accelerated distance computation

## Quick Start

### Basic Embedding Generation

```swift
import EmbedKit

// Create an embedding model
let model = try await AppleEmbeddingModel(
    modelURL: Bundle.main.url(forResource: "MiniLM-L12-v2", withExtension: "mlpackage")!
)

// Generate embedding for text
let embedding = try await model.embed("Hello, world!")
print("Dimensions: \(embedding.vector.count)")  // 384

// Batch processing
let texts = ["First document", "Second document", "Third document"]
let embeddings = try await model.embedBatch(texts, options: BatchOptions())

// Calculate similarity
let similarity = embeddings[0].similarity(to: embeddings[1])
print("Similarity: \(similarity)")  // 0.0 to 1.0
```

### Vector Storage & Search

```swift
import EmbedKit

// Create an embedding store with automatic GPU acceleration
let store = try await EmbeddingStore(
    config: .default(dimension: 384),  // HNSW index, cosine metric
    model: model
)

// Store text (embedding computed automatically)
_ = try await store.store(text: "Swift concurrency is powerful")
_ = try await store.store(text: "Metal provides GPU acceleration")
_ = try await store.store(text: "CoreML enables on-device ML")

// Semantic search
let results = try await store.search(text: "GPU programming", k: 2)
for result in results {
    print("\(result.similarity): \(result.text ?? "")")
}
```

### GPU Acceleration Control

```swift
// Automatic (default) - uses GPU when beneficial
let autoStore = try await EmbeddingStore(
    config: .default(dimension: 384, computePreference: .auto)
)

// Force CPU for deterministic testing
let cpuStore = try await EmbeddingStore(
    config: .exact(dimension: 384, computePreference: .cpuOnly)
)

// Check acceleration status
let isGPU = store.isAccelerationAvailable
let stats = await store.accelerationStatistics()
print("GPU ops: \(stats?.gpuOperations ?? 0)")
```

### Index Configurations

```swift
// Small datasets - exact search
let exactConfig = IndexConfiguration.exact(dimension: 384)

// Medium datasets - fast approximate search (default)
let hnswConfig = IndexConfiguration.default(dimension: 384)

// Large datasets - scalable IVF index
let ivfConfig = IndexConfiguration.scalable(dimension: 384, expectedSize: 100_000)

// Custom configuration
let customConfig = IndexConfiguration(
    indexType: .hnsw,
    dimension: 384,
    metric: .cosine,
    storeText: true,
    hnswConfig: .accurate,
    computePreference: .auto
)
```

### Reranking Strategies

```swift
// Exact cosine reranking for precision
let results = try await store.search(
    text: "query",
    k: 10,
    rerank: ExactCosineRerank()
)

// Diversity reranking (MMR)
let diverse = try await store.search(
    text: "query",
    k: 10,
    rerank: DiversityRerank(lambda: 0.7)
)

// Threshold filtering
let filtered = try await store.search(
    text: "query",
    k: 10,
    rerank: ThresholdRerank(minSimilarity: 0.5)
)
```

### Persistence

```swift
// Save store to disk
try await store.save(to: URL(fileURLWithPath: "/path/to/store"))

// Load from disk
let loaded = try await EmbeddingStore.load(from: URL(fileURLWithPath: "/path/to/store"))
```

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    EmbeddingStore                           │
│  - Stores embeddings with text/metadata                     │
│  - Semantic search with reranking                           │
│  - Automatic GPU acceleration                               │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  EmbeddingModel │  │   VectorIndex   │  │ Acceleration    │
│  - CoreML       │  │  - Flat/HNSW/   │  │  Manager        │
│  - ONNX         │  │    IVF indexes  │  │  - GPU/CPU      │
│  - Mock         │  │  - Persistence  │  │    routing      │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Directory Structure

```
Sources/EmbedKit/
├── Core/              # Embedding types, protocols, errors
├── Models/            # AppleEmbeddingModel, MockEmbeddingModel
├── Tokenization/      # WordPiece, BPE, SentencePiece tokenizers
├── Backends/          # CoreML backend implementation
├── Storage/           # EmbeddingStore, IndexConfiguration, Reranking
├── Acceleration/      # GPU acceleration manager
├── Caches/            # Token caching
└── Management/        # Model management

Sources/EmbedKitONNX/  # Optional ONNX Runtime support
```

## Supported Models

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| MiniLM-L6-v2 | 384 | 45MB | Fast, good for mobile |
| MiniLM-L12-v2 | 384 | 85MB | Better quality |
| all-mpnet-base-v2 | 768 | 180MB | High quality |
| BERT-base | 768 | 220MB | General purpose |

## Performance

### GPU Acceleration Thresholds

| Operation | GPU Threshold | Reasoning |
|-----------|---------------|-----------|
| Batch Distance | ≥1000 vectors | GPU overhead amortized |
| Vector Dimension | ≥64 | SIMD efficiency |
| Normalization | ≥100 vectors | Batch benefit |

### VectorAccelerate Speedups (when thresholds met)

- L2 Distance: ~82x speedup
- Cosine Similarity: ~97x speedup
- Top-K Selection: ~45x speedup

## Testing

```bash
# Run all tests (1431 tests, 358 suites)
swift test

# Run specific test suite
swift test --filter EmbedKitTests

# Run ONNX tests
swift test --filter EmbedKitONNXTests
```

### Test Coverage

| Component | Tests |
|-----------|-------|
| Core Types | 80+ |
| Tokenization | 60+ |
| Storage/Index | 120+ |
| Metal/Acceleration | 200+ |
| Rate Limiting | 45+ |
| Back Pressure | 50+ |
| Reranking | 50+ |
| Streaming | 100+ |
| Concurrency | 150+ |
| Edge Cases | 100+ |
| Integration | 200+ |

## Requirements

- iOS 18.0+ / macOS 15.0+ / tvOS 18.0+ / visionOS 2.0+ / watchOS 11.0+
- Swift 6.0+
- Xcode 16.0+
- Metal-capable device (optional, CPU fallback available)

## VSK Ecosystem

EmbedKit is part of the Vector Suite Kit:

| Package | Purpose |
|---------|---------|
| **[VectorCore](https://github.com/gifton/VectorCore)** | Core vector types and operations |
| **[VectorIndex](https://github.com/gifton/VectorIndex)** | Vector storage and similarity search |
| **[VectorAccelerate](https://github.com/gifton/VectorAccelerate)** | GPU-accelerated operations |
| **EmbedKit** | Text embedding generation (this package) |

## License

EmbedKit is available under the MIT license. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or PR on GitHub.

```bash
# Clone and test
git clone https://github.com/gifton/EmbedKit.git
cd EmbedKit
swift test
```
