# Changelog

All notable changes to EmbedKit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2025-11-11

### BREAKING CHANGES 🚨
- **Removed misleading `rerank` parameter** from `semanticSearch()` - it only sorted by score, didn't actually rerank
- **Deleted `SimpleRerankStrategy`** - it was fake reranking (just metadata sorting)
- **Removed `RerankingStrategyFactory`** - unnecessary abstraction layer
- **Changed `VectorSearchResult.id` from `UUID` to `String`** for flexibility
- **Changed metadata type from `[String: Any]` to `[String: String]`** for type safety

### Added
- **Real reranking support** via `ExactRerankStrategy` that actually recomputes distances
- **New clean `semanticSearch()` API** with optional real reranking:
  ```swift
  func semanticSearch(
      query: String,
      k: Int = 10,
      rerankStrategy: (any RerankingStrategy)? = nil,
      rerankOptions: RerankOptions = .default
  ) async throws -> [VectorSearchResult]
  ```
- **Batch search support** with optional reranking:
  ```swift
  func batchSearch(
      queries: [String],
      k: Int = 10,
      rerankStrategy: (any RerankingStrategy)? = nil
  ) async throws -> [[VectorSearchResult]]
  ```
- **Reranking option presets**: `.default`, `.fast`, `.accurate`
- **Mutable scores** in `VectorSearchResult` for reranking updates

### Removed
- All backward compatibility code (-25% codebase)
- Fake reranking strategies
- Misleading API parameters
- Unnecessary factory patterns
- Version suffixes like "_Clean", "Enhanced", "v2"

### Improved
- **10-20% search quality improvement** possible with real reranking
- **Cleaner API** - one way to search instead of three confusing methods
- **Honest functionality** - features do what they claim
- **Better performance** - no compatibility overhead

### Migration
- See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for upgrade instructions (5-minute migration)

## [Unreleased] - 2024-10-21

### Updated - Dependencies
- Updated to use official release versions:
  - VectorCore v0.1.2 (from GitHub)
  - VectorIndex v0.1.0-alpha (from GitHub)
- Added `VectorIndexBridge` for seamless integration with VectorIndex package
- Removed local path dependencies in favor of versioned releases

## [0.1.0-alpha.2] - 2024-10-21

### Added
- Initial alpha release of EmbedKit
- Core embedding types with compile-time dimension safety
  - `Embedding<D>` generic type with 384/768/1536 dimensions
  - `DynamicEmbedding` for runtime dimension handling
- Complete tokenization system
  - `BERTTokenizer` with WordPiece algorithm
  - `AdvancedTokenizer` with BPE/SentencePiece support
  - `SimpleTokenizer` for basic whitespace tokenization
- Metal GPU acceleration
  - 6 optimized compute kernels (normalization, pooling, similarity)
  - Automatic CPU fallback when Metal unavailable
  - SIMD optimizations with Metal 3 features
- CoreML backend implementation
  - Support for transformer models
  - Neural Engine acceleration
  - Flexible input/output configuration
- End-to-end embedding pipeline
  - Text → Tokenization → Inference → Pooling → Normalization
  - Batch processing support
  - LRU caching for embeddings
- Model management system
  - Model registry with metadata
  - Persistent model cache with LRU eviction
  - Automatic download from HuggingFace
- VectorIndex storage adapter
  - Generic storage backend protocol
  - In-memory storage implementation
  - Semantic search capabilities
- Comprehensive error handling
  - Typed errors for each component
  - LocalizedError conformance
- Actor-based concurrency
  - Thread-safe components using Swift 6 actors
  - Natural async/await integration

### Infrastructure
- Swift Package Manager configuration
- Unit tests for core components
- Model conversion script for CoreML
- Comprehensive documentation (README, ARCHITECTURE)
- MIT License

### Known Issues
- VectorAccelerate dependency temporarily disabled due to Swift 6 compatibility issues
- Some Metal framework types cause non-Sendable warnings
- VectorIndex package integration pending release

## [0.1.0] - Planned

### Target Features
- Complete integration tests
- Example journaling app
- Published to Swift Package Registry
- CoreML model included (MiniLM-L6-v2)

## Roadmap

### [0.2.0] - Future
- ONNX runtime support
- Model quantization (4-bit/8-bit)
- Streaming inference
- Cloud model support
- Performance profiling tools

### [0.3.0] - Future
- Custom model training
- Fine-tuning support
- Multi-modal embeddings (text + image)
- Cross-platform support (Linux/Windows)

### [1.0.0] - Future
- Production stability
- Comprehensive documentation
- Performance guarantees
- API stability commitment

---

## Version History

### Pre-release Development

#### 2024-10-21
- Migrated from initial EmbedKit prototype
- Integrated with Vector Suite Kit ecosystem
- Implemented Metal acceleration
- Added CoreML backend

#### 2024-10-20
- Initial tokenization system
- Core embedding types
- Basic pipeline structure

#### 2024-10-19
- Project inception
- Architecture design
- Dependency planning