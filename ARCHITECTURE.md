# EmbedKit Architecture

This document provides a detailed technical overview of EmbedKit's architecture, design decisions, and implementation details.

## Table of Contents
- [Design Philosophy](#design-philosophy)
- [System Architecture](#system-architecture)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Concurrency Model](#concurrency-model)
- [Memory Management](#memory-management)
- [Performance Optimizations](#performance-optimizations)
- [Extension Points](#extension-points)

## Design Philosophy

### Type Safety First
EmbedKit uses Swift's type system to enforce correctness at compile time:
- Generic types with phantom dimensions prevent dimension mismatches
- Protocol-oriented design for extensibility
- Strongly typed configuration objects

### Zero-Cost Abstractions
Performance-critical code uses:
- `@inlinable` for cross-module optimization
- `@frozen` structs for stable ABI
- Direct SIMD operations via VectorCore

### Actor-Based Concurrency
All stateful components are actors:
- Thread-safe by design
- Natural async/await integration
- No manual synchronization needed

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Application Layer                    │
├─────────────────────────────────────────────────────────┤
│                    EmbeddingPipeline                     │
│         Orchestrates the embedding generation flow       │
├──────────────┬──────────────┬──────────────┬────────────┤
│  Tokenization│ Model Backend│   Pooling    │Storage     │
│   BERTTokenizer   CoreML    │    Metal     │VectorIndex │
│   AdvancedTok│    MPS*      │    CPU       │  Adapter   │
├──────────────┴──────────────┴──────────────┴────────────┤
│                      Core Types                          │
│         Embedding<D>, DynamicEmbedding                   │
├─────────────────────────────────────────────────────────┤
│                     VectorCore                           │
│              Low-level SIMD operations                   │
└─────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Type-Safe Embeddings

#### Static Dimensions
```swift
public struct Embedding<D: EmbeddingDimension>: Sendable {
    private let vector: Vector<D>  // From VectorCore
}
```

The phantom type `D` ensures dimension safety:
- `Dim384`: 384-dimensional embeddings (MiniLM)
- `Dim768`: 768-dimensional embeddings (BERT)
- `Dim1536`: 1536-dimensional embeddings (OpenAI)

#### Dynamic Embeddings
```swift
public enum DynamicEmbedding: Sendable {
    case dim384(Embedding384)
    case dim768(Embedding768)
    case dim1536(Embedding1536)
}
```

Runtime dimension handling via tagged union pattern.

### 2. Tokenization System

#### Protocol Hierarchy
```swift
public protocol Tokenizer: Actor {
    func tokenize(_ text: String) async throws -> TokenizedInput
    func tokenize(batch texts: [String]) async throws -> [TokenizedInput]
}
```

#### BERT WordPiece Algorithm
```swift
actor BERTTokenizer: Tokenizer {
    // Greedy longest-match tokenization
    // O(n × m) where n = text length, m = max word length
    private func wordPieceTokenize(_ word: String) -> [String] {
        // Start with full word, progressively split
        // Use "##" prefix for subwords
    }
}
```

### 3. Model Backend System

#### Abstraction Layer
```swift
public protocol ModelBackend: Actor {
    var identifier: String { get }
    var isLoaded: Bool { get }
    func loadModel(from url: URL) async throws
    func generateEmbeddings(for input: TokenizedInput) async throws -> ModelOutput
}
```

#### CoreML Implementation
```swift
actor CoreMLBackend: ModelBackend {
    private var model: MLModel?
    // Handles model lifecycle, input/output conversion
    // Supports Neural Engine acceleration
}
```

### 4. GPU Acceleration

#### Metal Compute Pipeline
```metal
kernel void l2_normalize(
    device float* vector [[buffer(0)]],
    constant int& dimension [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    // 4-way vectorization with float4
    // SIMD group operations for reduction
    // Non-uniform threadgroup support
}
```

#### Resource Management
```swift
actor MetalResourceManager {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    private var pipelineCache: [String: MTLComputePipelineState]
    // Manages GPU resources and pipeline compilation
}
```

## Data Flow

### Text to Embedding Pipeline

1. **Input**: Raw text string
2. **Tokenization**: Text → TokenizedInput
   - Clean and normalize text
   - Split into tokens
   - Map to vocabulary IDs
   - Create attention mask

3. **Inference**: TokenizedInput → ModelOutput
   - Convert to MLMultiArray
   - Run through CoreML model
   - Extract hidden states

4. **Pooling**: ModelOutput → [Float]
   - Apply pooling strategy (mean/max/CLS)
   - Use GPU acceleration if available

5. **Normalization**: [Float] → Embedding
   - L2 normalize to unit vector
   - Wrap in typed embedding

### Batch Processing

```swift
// Optimized batch flow
texts: [String]
  ↓ Parallel tokenization
[TokenizedInput]
  ↓ Batched inference
[ModelOutput]
  ↓ Parallel pooling (GPU)
[DynamicEmbedding]
```

## Concurrency Model

### Actor Isolation
Each component is an isolated actor:
- `BERTTokenizer`: Thread-safe tokenization
- `CoreMLBackend`: Model state management
- `MetalAccelerator`: GPU resource coordination
- `ModelCache`: Cache access synchronization

### Cooperative Scheduling
```swift
// Example: Pipeline coordinates multiple actors
public actor EmbeddingPipeline {
    private let tokenizer: any Tokenizer
    private let backend: any ModelBackend
    private let accelerator: MetalAccelerator?

    public func embed(_ text: String) async throws -> DynamicEmbedding {
        // Each await is a suspension point
        let tokens = try await tokenizer.tokenize(text)
        let output = try await backend.generateEmbeddings(for: tokens)
        let pooled = try await pool(output)
        return normalize(pooled)
    }
}
```

## Memory Management

### Buffer Pooling
```swift
// Reusable Metal buffers
class BufferPool {
    private var available: [Int: [MTLBuffer]] = [:]

    func acquire(size: Int) -> MTLBuffer? {
        // Reuse or allocate
    }

    func release(_ buffer: MTLBuffer) {
        // Return to pool
    }
}
```

### Copy-on-Write
```swift
struct Embedding<D> {
    private var storage: Vector<D>  // COW from VectorCore

    mutating func normalize() {
        if !isKnownUniquelyReferenced(&storage) {
            storage = storage.copy()
        }
        // Modify in place
    }
}
```

### Cache Eviction
```swift
actor ModelCache {
    private let maxSizeBytes: Int64

    private func evictModelsForSpace(_ required: Int64) async throws {
        // LRU eviction based on lastAccessed
        let sorted = entries.values.sorted { $0.lastAccessed < $1.lastAccessed }
        // Evict until space available
    }
}
```

## Performance Optimizations

### SIMD Vectorization
```swift
// VectorCore provides SIMD operations
@inlinable
public func dotProduct(_ other: Self) -> Float {
    // Uses Accelerate.framework or custom SIMD
    vDSP_dotpr(self.data, 1, other.data, 1, &result, count)
}
```

### Metal Kernel Optimizations
- **Memory Coalescing**: Access patterns aligned to 128-byte boundaries
- **Threadgroup Memory**: Shared memory for reduction operations
- **Loop Unrolling**: 4-way unrolling for better ILP
- **Fast Math**: Relaxed precision for speed

### Caching Strategy
```swift
// Multi-level caching
1. Embedding cache: text → embedding (LRU, 1000 entries)
2. Model cache: identifier → MLModel (persistent, 5GB limit)
3. Pipeline cache: Frequently used pipelines stay loaded
```

## Extension Points

### Custom Tokenizers
```swift
// Implement Tokenizer protocol
actor MyCustomTokenizer: Tokenizer {
    func tokenize(_ text: String) async throws -> TokenizedInput {
        // Your tokenization logic
    }
}
```

### Alternative Backends
```swift
// Implement ModelBackend protocol
actor ONNXBackend: ModelBackend {
    // ONNX Runtime integration
}

actor MPSBackend: ModelBackend {
    // Metal Performance Shaders
}
```

### Storage Adapters
```swift
// Implement VectorStorageBackend protocol
actor PineconeStorage: VectorStorageBackend {
    // Cloud vector database integration
}
```

### Custom Pooling
```swift
extension PoolingStrategy {
    static let customWeighted = PoolingStrategy(
        name: "custom_weighted",
        compute: { tokens, weights in
            // Custom pooling logic
        }
    )
}
```

## Design Decisions

### Why Actors Over Classes?
- Automatic thread safety
- Natural async/await integration
- Compiler-enforced isolation
- No manual locks needed

### Why Generic Dimensions?
- Compile-time verification
- Zero runtime overhead
- Clear API contracts
- Prevents dimension mismatches

### Why Metal Over Accelerate?
- Better GPU utilization
- Custom kernel optimization
- Unified memory on Apple Silicon
- Future-proof for ML workloads

### Why CoreML Over ONNX?
- Native Apple integration
- Neural Engine support
- Smaller binary size
- Better battery efficiency

## Performance Characteristics

### Time Complexity
| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| Tokenization | O(n × m) | 2ms |
| Inference | O(n × d²) | 12ms |
| Pooling | O(n × d) | 0.5ms |
| Normalization | O(d) | 0.1ms |
| Similarity | O(d) | 0.05ms |

Where:
- n = sequence length
- d = embedding dimension
- m = max word length

### Space Complexity
| Component | Memory | Notes |
|-----------|--------|-------|
| Model | 45-200MB | Depends on architecture |
| Token cache | 1MB | 1000 entries |
| Embeddings | 6KB/vector | 1536D × 4 bytes |
| Buffers | 10MB | Reusable Metal buffers |

## Testing Strategy

### Unit Tests
- Type safety verification
- Tokenization edge cases
- Dimension calculations
- Error handling

### Integration Tests
- End-to-end pipeline
- Model loading/unloading
- Cache behavior
- Concurrency stress tests

### Performance Tests
- Throughput benchmarks
- Memory profiling
- GPU utilization
- Battery impact

## Security Considerations

### On-Device Processing
- No data leaves device
- Models stored locally
- Embeddings in memory only

### Model Integrity
- SHA256 checksum verification
- Signed model packages
- Secure storage in app sandbox

### Memory Safety
- Swift's memory safety guarantees
- Bounds checking on arrays
- No unsafe pointer arithmetic

## Future Architecture Evolution

### Planned Enhancements
1. **Streaming Inference**: Process text as it's typed
2. **Model Quantization**: 4-bit/8-bit models for size reduction
3. **Multi-Modal**: Image and text embeddings
4. **Distributed Computing**: Multi-device coordination

### Extensibility Points
1. **Plugin System**: Dynamic loading of tokenizers/models
2. **Custom Operations**: User-defined pooling/normalization
3. **Cloud Backends**: Hybrid on-device/cloud processing
4. **Cross-Platform**: Linux/Windows support via Swift

---

This architecture is designed to evolve while maintaining backward compatibility and performance guarantees.