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
- Performance profiling tools
- Example applications

## [0.2.3] - 2026-01

### Added

#### UMAP Dimensionality Reduction (VectorAccelerate 0.3.3)
- `UMAPConfiguration` - Configuration for GPU-accelerated UMAP projection
  - Factory presets: `.visualization2D()`, `.visualization3D()`, `.quickPreview()`, `.highQuality()`, `.clusterEmphasis()`, `.globalStructure()`
  - Configurable parameters: targetDimension, neighbors, minDistance, iterations, learningRate, negativeSampleRate, spread
  - Full validation with descriptive errors
- `AccelerationManager.umapProject(embeddings:config:)` - GPU-accelerated UMAP projection
- `AccelerationManager.umapGradient(embeddings:lowDimEmbeddings:neighbors:)` - Compute gradients for custom optimization
- `ModelManager.projectToVisualize(_:dimensions:)` - Embed documents and project to 2D/3D
- `ModelManager.projectEmbeddings(_:dimensions:)` - Project pre-computed embeddings
- `EmbeddingConfiguration.forVisualization(maxLength:)` - Configuration preset for visualization
- `UseCase.visualization` - New use case for dimensionality reduction

### Removed

#### Clustering APIs (Migrated to SwiftTopics)
- **BREAKING**: `ModelManager.clusterDocuments(_:numberOfClusters:maxIterations:)` - Use SwiftTopics for clustering
- **BREAKING**: `ClusteringViewModel` - Use SwiftTopics for clustering UI
- **BREAKING**: `EmbeddingConfiguration.forClustering()` - Use `.forVisualization()` or custom config
- **BREAKING**: `UseCase.clustering` - Use `.visualization` for dimensionality reduction

### Dependencies
- VectorAccelerate 0.3.3+ (UMAP gradient kernel)

---

## [0.2.2] - 2024-12

### Added

#### Metal 4 API Upgrade
- `Metal4CommandAllocator` - Command allocator pool for Metal 4 residency sets
- `Metal4CommandEncoding` - Tensor-aware command encoding
- `Metal4ResourceManagement` - Residency set and memory tier management
- `Metal4Extensions` - Device capability detection for Metal 4 features
- `TensorDescriptorFactory` - MTLTensorDescriptor creation for ML operations
- `TensorResultExtractor` - Type-safe extraction from MTLTensor results
- `TensorLifecycleManager` - Automatic tensor memory management
- `TensorOperationDispatcher` - Batched tensor operation dispatch
- `TensorStorageManager` - Tensor buffer allocation and pooling
- New Metal shaders: `FusedOperationsV2`, `TensorNormalizationV2`, `TensorPoolingV2`, `TensorSimilarityV2`

#### Configuration Factories
- `EmbeddingConfiguration.forSemanticSearch()` - Optimized for search applications
- `EmbeddingConfiguration.forRAG()` - Optimized for RAG pipelines
- `EmbeddingConfiguration.forClustering()` - Optimized for clustering/classification
- `EmbeddingConfiguration.forSimilarity()` - Optimized for text pair similarity
- `EmbeddingConfiguration.forDocuments()` - Optimized for document embeddings
- `EmbeddingConfiguration.forShortText()` - Optimized for queries/titles
- `EmbeddingConfiguration.forMiniLM()` - MiniLM model presets
- `EmbeddingConfiguration.forBERT()` - BERT model presets

#### EmbeddingGenerator (VectorProducer)
- `EmbeddingGenerator` actor - High-level embedding generation with `VectorProducer` conformance
- `produce(_:)` methods for batch and single-text embedding
- `generateWithProgress(_:)` - Streaming embeddings with `OperationProgress`
- `generateEmbeddingsWithProgress(_:)` - Streaming with `BatchProgress`
- `generateBatch(_:onProgress:)` - Batch processing with progress callbacks
- `produceWithIDs(_:)` - ID-preserving batch embedding
- `produceWithIDsAndProgress(_:)` - ID-preserving streaming
- `generateBatchWithIDs(_:onProgress:)` - ID-preserving batch with callbacks
- `GeneratorConfiguration` with presets: `.default`, `.highThroughput`, `.lowLatency`, `.forSemanticSearch()`, `.forRAG()`
- `warmup()` and `release()` lifecycle methods

#### BatchProgress
- `BatchProgress` struct - Embedding-specific progress with throughput metrics
- Integration with VectorCore's `OperationProgress`
- Batch indexing and token counting
- Factory methods: `.started()`, `.completed()`, `.batchCompleted()`
- Builder methods: `.withCurrent()`, `.withThroughput()`, `.nextBatch()`

#### Streaming Enhancements
- `StreamingEmbeddingGenerator` - Async streaming embedding generation
- Cancellable embedding tasks with `Task.checkCancellation()` support
- Back-pressure management with multiple strategies (suspend, dropOldest, dropNewest, error)
- Rate limiting with adaptive strategies (tokenBucket, slidingWindow, fixedWindow, leakyBucket)

#### VSK Integration
- `SharedMetalContextManager` - Cross-package Metal resource sharing between EmbedKit and VectorAccelerate
- `SharedMetalContextConfiguration` with presets: `.default`, `.forEmbedding`, `.forSearch`, `.debug`
- `SharedContextStatistics` - Unified GPU resource monitoring
- `AccelerationManager.createWithSharedContext()` - Shared context factory
- `EmbeddingStore.createWithSharedContext()` - Shared context factory
- `EmbedKitError: VSKError` conformance with error codes 2000-2999
- `isRecoverable`, `domain`, `context` properties on all errors
- Recovery suggestions for all error cases

#### SwiftUI Support Enhancements
- `ModelProvider` enum - Configure which model to use in ViewModels (`.system()` or `.mock`)
- `EmbeddingViewModel(modelProvider:)` - Configurable model provider (defaults to system)
- `SimilarityViewModel(modelProvider:)` - Configurable model provider (defaults to system)
- `EmbeddingViewModel.preview` - SwiftUI preview helper with mock model
- `SimilarityViewModel.preview` - SwiftUI preview helper with mock model

#### Convenience API Enhancements
- `ModelManager.getOrCreateSystemModel()` - Cached system model for convenience APIs
- `ModelManager.clearCachedSystemModel()` - Clear cached model to free memory

### Changed
- **BREAKING**: `ModelManager.loadAppleModel()` now returns Apple's NLContextualEmbedding model instead of mock
- Convenience APIs (`quickEmbed`, `semanticSearch`, `clusterDocuments`, `similarityMatrix`, `embedSequence`) now use real NLContextualEmbedding embeddings by default
- SwiftUI ViewModels (`EmbeddingViewModel`, `SimilarityViewModel`) now default to system model
- `ClusteringViewModel` now uses real semantic embeddings via `ModelManager.clusterDocuments()`
- Upgraded minimum Metal version support to Metal 4
- Improved GPU memory management with residency sets
- Enhanced batch processing with adaptive sizing
- Optimized tensor operations with fused kernels

### Deprecated
- `ModelManager.loadAppleModel(real:)` - Use `loadAppleModel()` for real models or `loadMockModel()` for testing

### Fixed
- Removed unnecessary nil-coalescing on non-optional `tokenizationConcurrency`

### Testing
- **1431 tests** across 358 test suites (up from 578 tests)
- Comprehensive Metal 4 API tests:
  - `CommandAllocatorTests` - Pool management and statistics
  - `TensorDescriptorFactoryTests` - Descriptor creation and validation
  - `TensorResultExtractorTests` - Type-safe extraction
  - `MetalDeviceExtensionsTests` - Device capability detection
  - `TensorLifecycleManagerTests` - Memory lifecycle management
  - `TensorOperationDispatcherTests` - Batch dispatch operations
  - `TensorStorageManagerTests` - Buffer allocation tests
  - `Metal4CommandEncodingTests` - Command encoding tests
  - `Metal4ResourceManagementTests` - Residency management tests
  - `Metal4EdgeCaseTests` - Edge case coverage
- EmbeddingGenerator tests:
  - `EmbeddingGeneratorTests` - Core functionality
  - `EmbeddingGeneratorProgressTests` - Progress streaming
  - `EmbeddingGeneratorIDTests` - ID-preserving methods
- BatchProgress tests:
  - `BatchProgressTests` - Progress tracking
- VSK Integration tests:
  - `VSKErrorConformanceTests` - Error protocol conformance
  - `SharedMetalContextManagerTests` - Context sharing
  - `MetalContextIntegrationTests` - Cross-package integration
  - `CrossPackageIntegrationTests` - End-to-end integration
  - `V020IntegrationTests` - Version integration tests
- Rate limiting comprehensive tests (45+ tests)
- Back-pressure controller comprehensive tests (50+ tests)
- Reranking strategy comprehensive tests (50+ tests)
- CI batched into test-metal, test-streaming, test-embedding jobs
