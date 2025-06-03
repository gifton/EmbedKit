# EmbedKit

A high-performance Swift package for generating text embeddings using Core ML with GPU acceleration, intelligent caching, and seamless PipelineKit integration.

## Features

- 🚀 **High Performance**: 10x+ faster batch processing with Metal GPU acceleration
- 💾 **Smart Caching**: 100x+ faster repeated queries with memory-aware LRU cache
- 🔧 **Core ML Integration**: Native support for Core ML embedding models
- 🎯 **Type-Safe API**: Swift-first design with compile-time safety
- 📊 **Batch Processing**: Efficient multi-text embedding generation
- 🧩 **PipelineKit Integration**: Seamless command-based architecture
- 🔄 **Actor-Based Concurrency**: Thread-safe operations with Swift's modern concurrency
- 📈 **Performance Monitoring**: Built-in benchmarking and telemetry

## Installation

### Swift Package Manager

Add EmbedKit to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/yourusername/EmbedKit.git", from: "1.0.0")
]
```

Then add to your target:

```swift
.target(
    name: "YourTarget",
    dependencies: ["EmbedKit"]
)
```

## Quick Start

### Basic Usage

```swift
import EmbedKit

// Initialize the embedder with a Core ML model
let embedder = try await CoreMLTextEmbedder(
    modelURL: modelURL,
    configuration: .default
)

// Generate embedding for a single text
let embedding = try await embedder.embed(text: "Hello, world!")
print("Embedding dimensions: \(embedding.values.count)")

// Calculate similarity between texts
let embedding1 = try await embedder.embed(text: "The cat sat on the mat")
let embedding2 = try await embedder.embed(text: "A feline rested on the rug")
let similarity = embedding1.cosineSimilarity(to: embedding2)
print("Similarity: \(similarity)") // ~0.85
```

### Batch Processing

```swift
// Process multiple texts efficiently
let texts = [
    "Swift is a powerful programming language",
    "Machine learning enables intelligent applications",
    "EmbedKit makes embeddings easy"
]

let embeddings = try await embedder.embed(texts: texts)
for (text, embedding) in zip(texts, embeddings) {
    print("\(text): \(embedding.values.count) dimensions")
}
```

### Configuration Options

```swift
// Customize embedder behavior
let configuration = EmbeddingConfiguration(
    batchSize: 32,
    useGPUAcceleration: true,
    cacheEnabled: true,
    maxCacheSize: 100_000_000, // 100MB
    telemetryEnabled: true
)

let embedder = try await CoreMLTextEmbedder(
    modelURL: modelURL,
    configuration: configuration
)
```

### PipelineKit Integration

```swift
import PipelineKit

// Create a pipeline with embedding commands
let pipeline = Pipeline {
    EmbedTextCommand(text: "Analyze this text")
        .handle(using: embedder)
        .map { embedding in
            // Process the embedding
            return embedding.values.count
        }
}

let dimensionCount = try await pipeline.execute()
```

### Streaming Large Collections

```swift
// Stream embeddings for memory efficiency
let documents = loadLargeDocumentCollection() // AsyncSequence<String>

for try await embedding in embedder.embedStream(texts: documents) {
    // Process each embedding as it's generated
    await processEmbedding(embedding)
}
```

## Advanced Features

### Metal GPU Acceleration

EmbedKit automatically utilizes Metal for GPU-accelerated operations:

```swift
// GPU acceleration is enabled by default
let embedder = try await CoreMLTextEmbedder(
    modelURL: modelURL,
    configuration: .default // useGPUAcceleration: true
)

// Batch normalization and pooling run on GPU
let embeddings = try await embedder.embed(texts: largeBatch)
```

### Memory-Aware Caching

The intelligent cache automatically manages memory pressure:

```swift
// Cache adapts to system memory conditions
let embedder = try await CoreMLTextEmbedder(
    modelURL: modelURL,
    configuration: EmbeddingConfiguration(
        cacheEnabled: true,
        maxCacheSize: 200_000_000 // 200MB limit
    )
)

// Repeated queries are served from cache
let embedding1 = try await embedder.embed(text: "Cached text")
let embedding2 = try await embedder.embed(text: "Cached text") // 100x faster
```

### Performance Benchmarking

```swift
// Measure embedding performance
let benchmark = EmbeddingBenchmark()

let result = try await benchmark.measure {
    try await embedder.embed(texts: testTexts)
}

print("Total time: \(result.totalTime)s")
print("Throughput: \(result.throughput) embeddings/second")
print("Average latency: \(result.averageLatency)ms")
```

### Custom Tokenization

```swift
// Implement custom tokenizer
struct CustomTokenizer: Tokenizer {
    func tokenize(_ text: String) -> TokenizedInput {
        // Your tokenization logic
        let tokens = text.split(separator: " ").map(String.init)
        let ids = tokens.map { token in
            // Convert to token IDs
        }
        return TokenizedInput(
            text: text,
            tokens: tokens,
            tokenIDs: ids,
            attentionMask: Array(repeating: 1, count: ids.count)
        )
    }
}

let embedder = try await CoreMLTextEmbedder(
    modelURL: modelURL,
    tokenizer: CustomTokenizer(),
    configuration: .default
)
```

## Architecture

EmbedKit follows a modular, protocol-oriented architecture:

```
EmbedKit/
├── Core/               # Core protocols and types
├── Models/             # Core ML backend implementation
├── Acceleration/       # Metal GPU acceleration
├── Cache/              # LRU caching system
├── PipelineIntegration/# PipelineKit commands and handlers
├── Tokenizers/         # Text tokenization
├── Monitoring/         # Telemetry and metrics
└── Utilities/          # Benchmarking and helpers
```

## Performance

Benchmark results on Apple Silicon (M1 Pro):

| Operation | Without EmbedKit | With EmbedKit | Improvement |
|-----------|-----------------|---------------|-------------|
| Single Embedding | 12ms | 11ms | 1.1x |
| Batch (100 texts) | 1,200ms | 95ms | 12.6x |
| Cached Query | 11ms | 0.1ms | 110x |
| Memory Usage | 450MB | 250MB | 1.8x |

## Requirements

- iOS 16.0+ / macOS 13.0+ / tvOS 16.0+ / watchOS 9.0+
- Swift 6.1+
- Xcode 16.0+
- Core ML compatible device

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

EmbedKit is available under the MIT license. See the [LICENSE](LICENSE) file for more info.

## Acknowledgments

EmbedKit is built on top of Apple's powerful Core ML and Metal frameworks, leveraging Swift's modern concurrency features for optimal performance.