# Changelog

All notable changes to EmbedKit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha] - 2024-11-24

### Added

#### Core Embedding
- `Embedding` type with vector storage and similarity computation
- `EmbeddingMetadata` for tracking model info, processing time, and pooling strategy
- `ModelID` for identifying embedding models (provider/name/version)
- Pooling strategies: mean, max, CLS, weighted mean, attention-weighted

#### Embedding Models
- `EmbeddingModel` protocol for embedding generation
- `AppleEmbeddingModel` - CoreML-based embedding with Apple NLContextualEmbedding
- `LocalCoreMLModel` - Direct CoreML .mlpackage model support
- `MockEmbeddingModel` - Testing utility with configurable dimensions

#### Tokenization
- `WordPieceTokenizer` - BERT-compatible tokenization
- `BPETokenizer` - Byte-Pair Encoding support
- `SentencePieceTokenizer` - SentencePiece model support
- `SimpleTokenizer` - Basic whitespace tokenization
- `Vocabulary` - Token-to-ID mapping with special token handling
- Token caching with `TokenCache`

#### CoreML Backend
- `CoreMLBackend` - Full CoreML model inference
- Automatic input key detection (input_ids, attention_mask, token_type_ids)
- Batch processing support
- Device selection (CPU, GPU, Neural Engine)

#### Vector Storage (VectorIndex Integration)
- `EmbeddingStore` - Main storage actor for embeddings
- Index types: Flat (exact), HNSW (approximate), IVF (scalable)
- `IndexConfiguration` with factory methods (.default, .exact, .fast, .scalable)
- Text storage with embeddings
- Metadata support
- Persistence (save/load)

#### Search & Reranking
- Similarity search with configurable k
- Filter support for metadata-based filtering
- `RerankingStrategy` protocol
- `ExactCosineRerank` - Precise similarity reranking
- `DiversityRerank` - MMR-based diversity (lambda parameter)
- `ThresholdRerank` - Minimum similarity filtering
- `CompositeRerank` - Chain multiple strategies
- `NoRerank` - Pass-through for benchmarking

#### GPU Acceleration (VectorAccelerate Integration)
- `AccelerationManager` - Automatic GPU/CPU routing
- `ComputePreference` enum (auto, cpuOnly, gpuOnly)
- `AccelerationThresholds` - Configurable GPU usage thresholds
- `AccelerationStatistics` - Usage tracking
- Transparent CPU fallback when GPU unavailable
- Batch distance computation with Metal

#### Configuration
- `EmbeddingConfiguration` - Model configuration
- `BatchOptions` - Batch processing options
- `HNSWConfiguration` - HNSW index tuning (M, efConstruction, efSearch)
- `IVFConfiguration` - IVF index tuning (nlist, nprobe)

#### ONNX Support (EmbedKitONNX)
- `ONNXEmbeddingModel` - ONNX Runtime-based embedding
- Dynamic shape support
- CPU and CoreML execution providers

### Testing
- 578 tests across 159 test suites
- Comprehensive coverage for all components
- Concurrency stress tests
- Edge case handling
- Memory pressure tests

### Dependencies
- VectorCore v0.1.4+
- VectorIndex v0.1.1+
- VectorAccelerate (local/v0.1.0-alpha+)
- swift-log v1.5.0+
- onnxruntime-swift-package-manager v1.19.0+ (EmbedKitONNX only)

### Platforms
- macOS 15.0+
- iOS 18.0+
- tvOS 18.0+
- watchOS 11.0+
- visionOS 2.0+

---

## [Unreleased]

### Planned
- Cloud model support
- Model quantization
- Streaming inference
- Performance profiling tools
- Example applications
