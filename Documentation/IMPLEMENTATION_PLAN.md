# EmbedKit Implementation Plan

## Architecture Overview

### Core Design Principles
1. **Actor-Based Concurrency**: Thread-safe embedding operations using Swift actors
2. **Hardware Acceleration**: Metal/GPU acceleration for batch processing
3. **Modular Design**: Pluggable model backends (Core ML, MLX, custom)
4. **Memory Efficiency**: Streaming support, smart caching, and memory-mapped models
5. **PipelineKit Native**: First-class integration with command-pipeline architecture

### Key Components

```
┌─────────────────────────────────────────────────────────────┐
│                         EmbedKit                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │  TextEmbedder   │  │ ModelLoader  │  │   Tokenizer   │ │
│  │   (Protocol)    │  │              │  │               │ │
│  └────────┬────────┘  └──────┬───────┘  └───────┬───────┘ │
│           │                   │                   │         │
│  ┌────────▼────────┐  ┌──────▼───────┐  ┌───────▼───────┐ │
│  │  CoreMLBackend  │  │ Model Cache  │  │  SentencePiece│ │
│  │                 │  │              │  │   Tokenizer   │ │
│  └────────┬────────┘  └──────────────┘  └───────────────┘ │
│           │                                                 │
│  ┌────────▼────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │ MetalAccelerator│  │   LRU Cache  │  │Streaming      │ │
│  │                 │  │              │  │  Embedder     │ │
│  └─────────────────┘  └──────────────┘  └───────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              PipelineKit Integration                 │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ EmbedTextCommand │ EmbeddingMiddleware │ Metrics    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Dependencies Required
1. **Core ML** - For model inference
2. **Metal/MetalPerformanceShaders** - For GPU acceleration
3. **PipelineKit** - For command architecture integration
4. **SentencePiece/Tokenizers** - For text tokenization
5. **OSLog** - For structured logging
6. **XCTest/Testing** - For comprehensive testing

## Phase 1: Foundation (Week 1-2)

### Goals
- Basic text embedding functionality
- Core ML model loading
- Simple API design
- Initial PipelineKit integration

### Steps

#### 1.1 Project Setup
- [ ] Update Package.swift with dependencies
- [ ] Create folder structure
- [ ] Setup development environment
- [ ] Add PipelineKit as local dependency

#### 1.2 Core Protocols
```swift
// TextEmbedder.swift
public protocol TextEmbedder: Actor {
    associatedtype EmbeddingVector: Collection where EmbeddingVector.Element == Float
    
    func embed(_ text: String) async throws -> EmbeddingVector
    func embed(batch texts: [String]) async throws -> [EmbeddingVector]
    var dimensions: Int { get }
    var modelIdentifier: String { get }
}

// ModelBackend.swift
public protocol ModelBackend {
    func loadModel(from url: URL) async throws
    func generateEmbedding(for tokens: [Int]) async throws -> [Float]
    var isLoaded: Bool { get }
}
```

#### 1.3 Basic Core ML Implementation
- [ ] CoreMLBackend implementation
- [ ] Model loading from .mlmodelc files
- [ ] Simple tokenization (whitespace initially)
- [ ] Error handling for model failures

#### 1.4 Minimal PipelineKit Integration
- [ ] Create EmbedTextCommand
- [ ] Basic context integration
- [ ] Simple error propagation

### Deliverables
- Working embed() function
- Load small model (MiniLM-L6-v2)
- Basic unit tests
- Simple CLI demo

## Phase 2: Performance & Robustness (Week 3-4)

### Goals
- Metal acceleration for batching
- Proper tokenization
- Caching layer
- Memory management

### Steps

#### 2.1 Metal Acceleration
```swift
// MetalAccelerator.swift
actor MetalAccelerator {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var kernels: [String: MTLComputePipelineState]
    
    func acceleratedNormalize(_ vectors: [[Float]]) async -> [[Float]]
    func acceleratedPooling(_ tokens: [[Float]], strategy: PoolingStrategy) async -> [Float]
}
```

#### 2.2 Advanced Tokenization
- [ ] Integrate SentencePiece or similar
- [ ] Support for subword tokenization
- [ ] Handle special tokens (CLS, SEP)
- [ ] Truncation and padding strategies

#### 2.3 Caching System
- [ ] LRU cache implementation
- [ ] Configurable cache size
- [ ] Cache key generation (hash-based)
- [ ] Memory pressure handling

#### 2.4 Batch Processing
- [ ] Efficient batch tokenization
- [ ] Optimized memory allocation
- [ ] Parallel processing where applicable

### Deliverables
- 10x performance improvement for batches
- Memory-efficient for 10k+ embeddings
- Comprehensive benchmarks
- Tokenization unit tests

## Phase 3: Advanced Features (Week 5-6)

### Goals
- Streaming support
- Model versioning
- Advanced PipelineKit features
- Production hardening

### Steps

#### 3.1 Streaming Implementation
```swift
// StreamingEmbedder.swift
public actor StreamingEmbedder<Embedder: TextEmbedder> {
    func embedStream<S: AsyncSequence>(_ texts: S) -> AsyncStream<Result<Embedder.EmbeddingVector, Error>> 
    where S.Element == String
    
    func embedStreamWithBackpressure<S: AsyncSequence>(
        _ texts: S,
        maxConcurrent: Int = 10
    ) -> AsyncStream<Result<Embedder.EmbeddingVector, Error>>
}
```

#### 3.2 Model Management
- [ ] Model versioning system
- [ ] Hot-swapping capabilities
- [ ] Model metadata storage
- [ ] Automatic model updates

#### 3.3 Advanced PipelineKit Integration
- [ ] EmbeddingMiddleware with caching
- [ ] Metrics collection middleware
- [ ] Custom context extensions
- [ ] Pipeline presets

#### 3.4 Production Features
- [ ] Comprehensive error handling
- [ ] Graceful degradation
- [ ] Resource limits
- [ ] Telemetry and monitoring

### Deliverables
- Stream processing of large documents
- Model A/B testing capability
- Production-ready error handling
- Performance monitoring

## Phase 4: Optimization & Polish (Week 7-8)

### Goals
- Model quantization support
- Memory-mapped models
- API refinements
- Documentation

### Steps

#### 4.1 Quantization Support
- [ ] Int8/Int4 model support
- [ ] Dynamic quantization
- [ ] Quality vs size tradeoffs
- [ ] Automatic fallbacks

#### 4.2 Memory Optimization
- [ ] Memory-mapped model loading
- [ ] Lazy loading strategies
- [ ] Aggressive memory reclamation
- [ ] iOS/macOS memory limits

#### 4.3 API Polish
- [ ] Convenience initializers
- [ ] Builder pattern for configuration
- [ ] Async sequence support
- [ ] SwiftUI property wrappers

#### 4.4 Documentation & Examples
- [ ] API documentation
- [ ] Usage examples
- [ ] Performance guide
- [ ] Migration guide

### Deliverables
- 4x memory reduction with quantization
- Comprehensive documentation
- Example apps
- Performance tuning guide

## Testing Strategy

### Unit Tests
- Protocol conformance
- Tokenization accuracy
- Cache behavior
- Error handling

### Integration Tests
- PipelineKit integration
- Model loading
- Batch processing
- Memory limits

### Performance Tests
- Embedding speed (tokens/sec)
- Memory usage
- Cache hit rates
- Concurrent operations

### Device Tests
- iPhone (various models)
- iPad
- Mac (Apple Silicon)
- Memory pressure scenarios

## Success Metrics

### Performance
- **Single embedding**: < 10ms on M1
- **Batch (100 texts)**: < 100ms on M1
- **Memory overhead**: < 500MB for model + cache
- **Cache hit rate**: > 80% for common queries

### Quality
- **Test coverage**: > 90%
- **API stability**: No breaking changes after v1.0
- **Documentation**: 100% public API documented
- **Error handling**: All failure modes handled

### Integration
- **PipelineKit**: Seamless command integration
- **VectorStoreKit**: Direct compatibility
- **Platform support**: iOS 16+, macOS 13+

## Risk Mitigation

### Technical Risks
1. **Model size constraints**: Start with small models, add streaming for large ones
2. **Memory pressure**: Implement aggressive caching policies, memory warnings
3. **Performance bottlenecks**: Profile early, optimize critical paths
4. **API changes**: Design for extensibility, use protocols

### Timeline Risks
1. **Core ML limitations**: Have fallback implementations ready
2. **Metal complexity**: Start simple, iterate on optimizations
3. **Testing overhead**: Automate early, continuous integration

## Future Considerations

### Post-v1.0 Features
- MLX backend support
- Custom model training
- Multilingual models
- Fine-tuning capabilities
- Distributed embedding generation
- WebAssembly compilation

### Integration Opportunities
- Siri Shortcuts
- App Intents
- CloudKit sync
- SharePlay support

This plan provides a structured approach to building EmbedKit while maintaining flexibility for discoveries and adjustments during development.