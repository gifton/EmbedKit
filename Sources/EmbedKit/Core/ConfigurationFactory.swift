// EmbedKit - Configuration Factory
// Unified configuration presets for common embedding pipeline use cases
//
// ## API Relationship
//
// This file provides `ConfigurationFactory` which creates complete `PipelineConfiguration`
// instances that bundle together:
// - `EmbeddingConfiguration` (from Types.swift) - tokenization, pooling, normalization
// - `BatchOptions` (from Types.swift) - batching behavior
// - `ComputeConfiguration` (defined here) - GPU/CPU selection
// - `CacheConfiguration` (from CacheTypes.swift) - persistent caching
//
// Use `EmbeddingConfiguration` factories (`.forSemanticSearch()`, `.forRAG()`, etc.)
// when you only need embedding settings. Use `ConfigurationFactory` when you need
// complete pipeline configuration including batching and compute settings.
//
// Example:
//   // Just embedding config:
//   let embeddingConfig = EmbeddingConfiguration.forSemanticSearch()
//
//   // Full pipeline config:
//   let pipelineConfig = ConfigurationFactory.forSemanticSearch()  // includes batch + compute

import Foundation

// MARK: - Pipeline Configuration

/// Unified configuration for the complete embedding pipeline.
///
/// `PipelineConfiguration` bundles together all configuration options needed to set up
/// an embedding pipeline, including embedding behavior, batching strategy, compute
/// preferences, and caching. Use `ConfigurationFactory` to create pre-configured
/// instances for common use cases.
///
/// ## Components
/// - `embedding`: Controls tokenization, pooling, normalization, and max tokens
/// - `batch`: Controls batch sizes, dynamic batching, and sorting
/// - `compute`: Controls GPU/CPU selection and acceleration thresholds
/// - `cache`: Controls persistent caching behavior (optional)
///
/// ## Example
/// ```swift
/// // Use a preset for high-throughput batch processing
/// let config = ConfigurationFactory.highThroughput()
///
/// // Or customize individual components
/// let custom = PipelineConfiguration(
///     embedding: .forSemanticSearch(),
///     batch: .highThroughput,
///     compute: .gpuOptimized()
/// )
/// ```
public struct PipelineConfiguration: Sendable {

    /// Configuration for embedding operations (tokenization, pooling, normalization).
    public let embedding: EmbeddingConfiguration

    /// Configuration for batch processing behavior.
    public let batch: BatchOptions

    /// Configuration for GPU/CPU compute path selection.
    public let compute: ComputeConfiguration

    /// Configuration for persistent embedding cache (optional).
    public let cache: CacheConfiguration?

    /// Memory budget hint in bytes for the pipeline.
    /// Used by adaptive components to adjust their behavior.
    /// Set to nil for automatic detection.
    public let memoryBudget: Int?

    /// Creates a pipeline configuration with the specified components.
    ///
    /// - Parameters:
    ///   - embedding: Embedding configuration (default: balanced defaults)
    ///   - batch: Batch options (default: balanced defaults)
    ///   - compute: Compute configuration (default: auto-selection)
    ///   - cache: Cache configuration (default: nil, no caching)
    ///   - memoryBudget: Memory budget hint in bytes (default: nil, auto-detect)
    public init(
        embedding: EmbeddingConfiguration = .default,
        batch: BatchOptions = .default,
        compute: ComputeConfiguration = .default,
        cache: CacheConfiguration? = nil,
        memoryBudget: Int? = nil
    ) {
        self.embedding = embedding
        self.batch = batch
        self.compute = compute
        self.cache = cache
        self.memoryBudget = memoryBudget
    }
}

// MARK: - Compute Configuration

/// Configuration for GPU-accelerated compute operations.
///
/// EmbedKit uses GPU acceleration (Metal4) for all compute operations.
/// This configuration controls kernel optimization settings.
public struct ComputeConfiguration: Sendable {

    /// Whether to use fused Metal kernels when available.
    /// Fused kernels combine multiple operations (e.g., pooling + normalization)
    /// into a single GPU dispatch for better performance.
    public let useFusedKernels: Bool

    /// Whether to enable adaptive kernel selection based on workload.
    /// When enabled, the system learns from past executions to optimize
    /// kernel selection for specific workload sizes.
    public let adaptiveKernelSelection: Bool

    /// Maximum resident GPU memory in megabytes.
    /// Buffers marked as frequently accessed will be kept resident
    /// up to this limit for faster access.
    public let maxResidentMemoryMB: Int

    /// Creates a compute configuration with the specified options.
    ///
    /// - Parameters:
    ///   - useFusedKernels: Use fused Metal kernels (default: true)
    ///   - adaptiveKernelSelection: Enable adaptive learning (default: true)
    ///   - maxResidentMemoryMB: Max GPU resident memory (default: 512)
    public init(
        useFusedKernels: Bool = true,
        adaptiveKernelSelection: Bool = true,
        maxResidentMemoryMB: Int = 512
    ) {
        self.useFusedKernels = useFusedKernels
        self.adaptiveKernelSelection = adaptiveKernelSelection
        self.maxResidentMemoryMB = maxResidentMemoryMB
    }

    /// Default GPU-optimized configuration.
    public static let `default` = ComputeConfiguration()

    /// Configuration optimized for maximum GPU utilization.
    public static func gpuOptimized() -> ComputeConfiguration {
        ComputeConfiguration(
            useFusedKernels: true,
            adaptiveKernelSelection: true,
            maxResidentMemoryMB: 1024
        )
    }

    /// Configuration for memory-constrained environments.
    public static func memoryEfficient() -> ComputeConfiguration {
        ComputeConfiguration(
            useFusedKernels: true,
            adaptiveKernelSelection: false,
            maxResidentMemoryMB: 128
        )
    }
}

// MARK: - Configuration Factory

/// Factory for creating pre-configured pipeline configurations for common use cases.
///
/// `ConfigurationFactory` provides static methods that return fully configured
/// `PipelineConfiguration` instances optimized for specific scenarios. Each preset
/// carefully balances all configuration options for its intended use case.
///
/// ## Relationship to EmbeddingConfiguration Factories
///
/// `EmbeddingConfiguration` (in Types.swift) has its own factory methods like
/// `.forSemanticSearch()` and `.forRAG()`. Those create **embedding-only** configuration.
///
/// `ConfigurationFactory` creates **complete pipeline** configurations that include:
/// - Embedding configuration (delegates to `EmbeddingConfiguration` factories)
/// - Batch options (sizes, timeouts, concurrency)
/// - Compute configuration (GPU/CPU preferences, thresholds)
/// - Cache configuration (persistence, eviction)
///
/// **When to use which:**
/// - Use `EmbeddingConfiguration.forSemanticSearch()` when you only need embedding settings
/// - Use `ConfigurationFactory.forSemanticSearch()` when you need full pipeline config
///
/// ## Available Presets
///
/// ### Performance-Oriented
/// - `default()`: Balanced defaults suitable for most applications
/// - `highThroughput()`: Maximum batch processing speed
/// - `lowLatency()`: Minimum response time for single items
/// - `gpuOptimized()`: Maximize GPU utilization on Metal-capable devices
///
/// ### Resource-Oriented
/// - `memoryEfficient()`: Minimize memory footprint for constrained environments
/// - `batteryEfficient()`: Optimize for power consumption on mobile devices
///
/// ### Use-Case Oriented
/// - `forSemanticSearch()`: Document search and retrieval applications
/// - `forRAG(chunkSize:)`: Retrieval-Augmented Generation pipelines
/// - `forRealTimeSearch()`: Interactive search with immediate feedback
/// - `forBatchIndexing()`: Offline indexing of large document collections
///
/// ## Example
/// ```swift
/// // High-throughput batch processing
/// let config = ConfigurationFactory.highThroughput()
/// let batcher = AdaptiveBatcher(model: model, config: config.toAdaptiveBatcherConfig())
///
/// // Real-time search with caching
/// let searchConfig = ConfigurationFactory.forRealTimeSearch(enableCache: true)
/// ```
public enum ConfigurationFactory {

    // MARK: - Performance-Oriented Presets

    /// Default configuration with balanced settings for most use cases.
    ///
    /// Provides a good balance between throughput, latency, and resource usage.
    /// Suitable as a starting point when you don't have specific requirements.
    ///
    /// - Batch size: 32 items
    /// - Dynamic batching: enabled with length sorting
    /// - Compute: auto (GPU when beneficial)
    /// - No persistent cache
    ///
    /// - Returns: A balanced pipeline configuration
    public static func `default`() -> PipelineConfiguration {
        PipelineConfiguration(
            embedding: .default,
            batch: .default,
            compute: .default,
            cache: nil,
            memoryBudget: nil
        )
    }

    /// Configuration optimized for maximum batch processing throughput.
    ///
    /// Prioritizes processing speed over latency for individual items.
    /// Best for offline processing, bulk indexing, or background embedding tasks.
    ///
    /// - Batch size: 64-128 items (memory-adaptive)
    /// - Dynamic batching: enabled with aggressive length sorting
    /// - GPU acceleration: aggressive thresholds
    /// - Fused kernels: enabled for combined operations
    /// - Lower GPU threshold for earlier GPU utilization
    ///
    /// - Returns: A throughput-optimized pipeline configuration
    ///
    /// ## Example
    /// ```swift
    /// let config = ConfigurationFactory.highThroughput()
    /// let embeddings = try await batcher.embedBatch(documents, options: config.batch)
    /// ```
    public static func highThroughput() -> PipelineConfiguration {
        PipelineConfiguration(
            embedding: EmbeddingConfiguration(
                maxTokens: 512,
                truncationStrategy: .end,
                paddingStrategy: .batch,
                includeSpecialTokens: true,
                poolingStrategy: .mean,
                normalizeOutput: true,
                inferenceDevice: .auto,
                minElementsForGPU: 2048  // Lower threshold for earlier GPU use
            ),
            batch: BatchOptions(
                maxBatchSize: 128,
                dynamicBatching: true,
                sortByLength: true,
                timeout: 60.0,
                bucketSize: 32,
                maxBatchTokens: 16384,
                tokenizationConcurrency: ProcessInfo.processInfo.activeProcessorCount,
                minBatchSize: 8,
                maxPaddingRatio: 0.5
            ),
            compute: ComputeConfiguration.gpuOptimized(),
            cache: nil,
            memoryBudget: nil
        )
    }

    /// Configuration optimized for minimum latency on single items.
    ///
    /// Prioritizes response time for individual embedding requests.
    /// Best for real-time applications where users are waiting for results.
    ///
    /// - Batch size: 1-8 items (process immediately)
    /// - Dynamic batching: disabled for immediate processing
    /// - No length sorting (avoids delay)
    /// - Short timeout
    ///
    /// - Returns: A latency-optimized pipeline configuration
    ///
    /// ## Example
    /// ```swift
    /// let config = ConfigurationFactory.lowLatency()
    /// let embedding = try await model.embed(query, config: config.embedding)
    /// ```
    public static func lowLatency() -> PipelineConfiguration {
        PipelineConfiguration(
            embedding: EmbeddingConfiguration(
                maxTokens: 256,  // Shorter sequences process faster
                truncationStrategy: .end,
                paddingStrategy: .none,  // No padding overhead
                includeSpecialTokens: true,
                poolingStrategy: .mean,
                normalizeOutput: true,
                inferenceDevice: .auto,
                minElementsForGPU: 16384
            ),
            batch: BatchOptions(
                maxBatchSize: 8,
                dynamicBatching: false,  // Process immediately
                sortByLength: false,     // No delay for sorting
                timeout: 5.0,
                bucketSize: 8
            ),
            compute: ComputeConfiguration(
                useFusedKernels: true,
                adaptiveKernelSelection: false,  // Consistent latency
                maxResidentMemoryMB: 256
            ),
            cache: nil,
            memoryBudget: nil
        )
    }

    /// Configuration optimized for maximum GPU utilization.
    ///
    /// Aggressively uses Metal GPU acceleration for all operations.
    /// Best for M-series Macs or devices with powerful GPUs where you want
    /// to maximize GPU throughput.
    ///
    /// - Aggressive GPU thresholds
    /// - Large batches to amortize GPU dispatch overhead
    /// - Fused kernels for combined operations
    /// - High GPU memory residency for hot data
    ///
    /// - Returns: A GPU-optimized pipeline configuration
    ///
    /// ## Example
    /// ```swift
    /// let config = ConfigurationFactory.gpuOptimized()
    /// // Ensure model also uses GPU inference
    /// let model = try await LocalCoreMLModel.load(path: path, computeUnits: .all)
    /// ```
    public static func gpuOptimized() -> PipelineConfiguration {
        PipelineConfiguration(
            embedding: EmbeddingConfiguration(
                maxTokens: 512,
                truncationStrategy: .end,
                paddingStrategy: .batch,
                includeSpecialTokens: true,
                poolingStrategy: .mean,
                normalizeOutput: true,
                inferenceDevice: .gpu,  // Prefer GPU for inference
                minElementsForGPU: 1024  // Very low threshold
            ),
            batch: BatchOptions(
                maxBatchSize: 64,
                dynamicBatching: true,
                sortByLength: true,
                timeout: 30.0,
                bucketSize: 32,
                maxBatchTokens: 32768,  // Large token batches for GPU
                tokenizationConcurrency: ProcessInfo.processInfo.activeProcessorCount
            ),
            compute: ComputeConfiguration.gpuOptimized(),
            cache: nil,
            memoryBudget: nil
        )
    }

    // MARK: - Resource-Oriented Presets

    /// Configuration optimized for minimum memory footprint.
    ///
    /// Reduces memory usage at the cost of some throughput.
    /// Best for memory-constrained environments like iOS extensions,
    /// background processes, or when running alongside memory-intensive apps.
    ///
    /// - Small batch sizes
    /// - No padding (variable-length processing)
    /// - Conservative GPU thresholds (less GPU memory)
    /// - Smaller token limits
    /// - No in-memory caching
    ///
    /// - Parameter memoryBudgetMB: Maximum memory budget in megabytes (default: 128)
    /// - Returns: A memory-efficient pipeline configuration
    ///
    /// ## Example
    /// ```swift
    /// // For an iOS Share Extension with limited memory
    /// let config = ConfigurationFactory.memoryEfficient(memoryBudgetMB: 64)
    /// ```
    public static func memoryEfficient(memoryBudgetMB: Int = 128) -> PipelineConfiguration {
        PipelineConfiguration(
            embedding: EmbeddingConfiguration(
                maxTokens: 256,  // Shorter sequences use less memory
                truncationStrategy: .end,
                paddingStrategy: .none,  // No padding memory overhead
                includeSpecialTokens: true,
                poolingStrategy: .mean,
                normalizeOutput: true,
                inferenceDevice: .auto,
                minElementsForGPU: 16384  // Prefer CPU to save GPU memory
            ),
            batch: BatchOptions(
                maxBatchSize: 8,  // Small batches
                dynamicBatching: true,
                sortByLength: true,
                timeout: nil,
                bucketSize: 8,
                maxBatchTokens: 1024,  // Limit tokens in memory
                tokenizationConcurrency: 2  // Fewer concurrent tasks
            ),
            compute: ComputeConfiguration.memoryEfficient(),
            cache: nil,  // No cache to save memory
            memoryBudget: memoryBudgetMB * 1024 * 1024
        )
    }

    /// Configuration optimized for battery efficiency on mobile devices.
    ///
    /// Uses moderate batch sizes and smaller GPU memory footprint.
    /// Best for iOS apps where battery life is a priority.
    ///
    /// - Moderate batch sizes
    /// - ANE (Apple Neural Engine) for inference when available
    /// - Lower GPU memory usage
    ///
    /// - Returns: A battery-efficient pipeline configuration
    ///
    /// ## Example
    /// ```swift
    /// let config = ConfigurationFactory.batteryEfficient()
    /// // Use for background indexing on iOS
    /// ```
    public static func batteryEfficient() -> PipelineConfiguration {
        PipelineConfiguration(
            embedding: EmbeddingConfiguration(
                maxTokens: 256,
                truncationStrategy: .end,
                paddingStrategy: .batch,
                includeSpecialTokens: true,
                poolingStrategy: .mean,
                normalizeOutput: true,
                inferenceDevice: .ane,  // Neural Engine is power-efficient
                minElementsForGPU: 32768
            ),
            batch: BatchOptions(
                maxBatchSize: 16,  // Moderate batches
                dynamicBatching: true,
                sortByLength: true,
                timeout: nil,
                bucketSize: 16,
                tokenizationConcurrency: 2  // Fewer threads = less power
            ),
            compute: ComputeConfiguration(
                useFusedKernels: true,
                adaptiveKernelSelection: false,  // Consistent, no learning overhead
                maxResidentMemoryMB: 128
            ),
            cache: nil,
            memoryBudget: nil
        )
    }

    // MARK: - Use-Case Oriented Presets

    /// Configuration optimized for semantic search applications.
    ///
    /// Balanced settings for searching through document collections.
    /// Optimizes for query embedding speed while maintaining quality.
    ///
    /// - Parameters:
    ///   - maxLength: Maximum token length for queries (default: 512)
    ///   - enableCache: Whether to enable persistent caching (default: false)
    /// - Returns: A semantic search-optimized pipeline configuration
    ///
    /// ## Example
    /// ```swift
    /// let config = ConfigurationFactory.forSemanticSearch(enableCache: true)
    /// let queryEmbedding = try await model.embed(query, config: config.embedding)
    /// let results = try await store.search(queryEmbedding, limit: 10)
    /// ```
    public static func forSemanticSearch(
        maxLength: Int = 512,
        enableCache: Bool = false
    ) -> PipelineConfiguration {
        PipelineConfiguration(
            embedding: .forSemanticSearch(maxLength: maxLength),
            batch: BatchOptions(
                maxBatchSize: 32,
                dynamicBatching: true,
                sortByLength: true,
                timeout: 10.0,
                bucketSize: 16
            ),
            compute: .default,
            cache: enableCache ? .default : nil,
            memoryBudget: nil
        )
    }

    /// Configuration optimized for Retrieval-Augmented Generation (RAG) pipelines.
    ///
    /// Optimized for chunking and embedding documents for RAG applications.
    /// Uses smaller chunk sizes suitable for context windows and efficient retrieval.
    ///
    /// - Parameters:
    ///   - chunkSize: Maximum tokens per document chunk (default: 256)
    ///   - enableCache: Whether to enable persistent caching (default: true)
    /// - Returns: A RAG-optimized pipeline configuration
    ///
    /// ## Example
    /// ```swift
    /// let config = ConfigurationFactory.forRAG(chunkSize: 384)
    /// let chunks = splitDocument(document, maxTokens: 384)
    /// let embeddings = try await batcher.embedBatch(chunks, options: config.batch)
    /// ```
    public static func forRAG(
        chunkSize: Int = 256,
        enableCache: Bool = true
    ) -> PipelineConfiguration {
        PipelineConfiguration(
            embedding: .forRAG(chunkSize: chunkSize),
            batch: BatchOptions(
                maxBatchSize: 64,
                dynamicBatching: true,
                sortByLength: true,
                timeout: 30.0,
                bucketSize: 32,
                maxBatchTokens: chunkSize * 64  // Optimize for chunk size
            ),
            compute: .default,
            cache: enableCache ? CacheConfiguration(
                maxEntries: 50_000,
                maxSizeBytes: Int64(250) * 1024 * 1024,  // 250 MB
                enableSemanticDedup: false,
                autoEvict: true,
                ttlSeconds: 0,
                enableWAL: true
            ) : nil,
            memoryBudget: nil
        )
    }

    /// Configuration optimized for real-time interactive search.
    ///
    /// Balances low latency for queries with efficient document indexing.
    /// Includes caching by default for frequently repeated queries.
    ///
    /// - Parameter enableCache: Whether to enable persistent caching (default: true)
    /// - Returns: A real-time search-optimized pipeline configuration
    ///
    /// ## Example
    /// ```swift
    /// let config = ConfigurationFactory.forRealTimeSearch()
    /// // Use for search-as-you-type functionality
    /// let results = try await searchEngine.search(query, config: config)
    /// ```
    public static func forRealTimeSearch(
        enableCache: Bool = true
    ) -> PipelineConfiguration {
        PipelineConfiguration(
            embedding: EmbeddingConfiguration(
                maxTokens: 128,  // Short queries
                truncationStrategy: .end,
                paddingStrategy: .none,
                includeSpecialTokens: true,
                poolingStrategy: .mean,
                normalizeOutput: true,
                inferenceDevice: .auto,
                minElementsForGPU: 8192
            ),
            batch: BatchOptions(
                maxBatchSize: 4,  // Very small for low latency
                dynamicBatching: false,  // Immediate processing
                sortByLength: false,
                timeout: 2.0
            ),
            compute: ComputeConfiguration(
                useFusedKernels: true,
                adaptiveKernelSelection: false,
                maxResidentMemoryMB: 256
            ),
            cache: enableCache ? CacheConfiguration(
                maxEntries: 10_000,
                maxSizeBytes: Int64(50) * 1024 * 1024,  // 50 MB - query cache
                enableSemanticDedup: true,  // Dedupe similar queries
                deduplicationThreshold: 0.95,
                autoEvict: true,
                ttlSeconds: 3600,  // 1 hour TTL for queries
                enableWAL: false  // Speed over durability for cache
            ) : nil,
            memoryBudget: nil
        )
    }

    /// Configuration optimized for offline batch indexing of large document collections.
    ///
    /// Maximizes throughput for indexing millions of documents.
    /// Uses large batches, aggressive GPU utilization, and persistent caching.
    ///
    /// - Parameters:
    ///   - maxTokens: Maximum tokens per document (default: 512)
    ///   - cachePath: Path for the embedding cache (default: nil, auto-generated)
    /// - Returns: A batch indexing-optimized pipeline configuration
    ///
    /// ## Example
    /// ```swift
    /// let config = ConfigurationFactory.forBatchIndexing()
    /// for batch in documents.chunked(into: 1000) {
    ///     let embeddings = try await batcher.embedBatch(batch, options: config.batch)
    ///     try await store.addBatch(embeddings)
    /// }
    /// ```
    public static func forBatchIndexing(
        maxTokens: Int = 512
    ) -> PipelineConfiguration {
        PipelineConfiguration(
            embedding: EmbeddingConfiguration(
                maxTokens: maxTokens,
                truncationStrategy: .end,
                paddingStrategy: .batch,
                includeSpecialTokens: true,
                poolingStrategy: .mean,
                normalizeOutput: true,
                inferenceDevice: .auto,
                minElementsForGPU: 2048
            ),
            batch: BatchOptions(
                maxBatchSize: 128,
                dynamicBatching: true,
                sortByLength: true,
                timeout: 120.0,
                bucketSize: 64,
                maxBatchTokens: 65536,  // Large token batches
                tokenizationConcurrency: ProcessInfo.processInfo.activeProcessorCount,
                minBatchSize: 32,  // Wait for larger batches
                maxPaddingRatio: 0.4
            ),
            compute: ComputeConfiguration.gpuOptimized(),
            cache: CacheConfiguration(
                maxEntries: 1_000_000,
                maxSizeBytes: Int64(2) * 1024 * 1024 * 1024,  // 2 GB
                enableSemanticDedup: false,
                autoEvict: true,
                ttlSeconds: 0,  // No expiration
                enableWAL: true
            ),
            memoryBudget: nil
        )
    }

    // MARK: - Model-Specific Presets

    /// Configuration optimized for MiniLM models (384 dimensions, ~22M parameters).
    ///
    /// MiniLM models are lightweight and efficient, suitable for most devices.
    /// This preset optimizes batch sizes and GPU thresholds for the smaller
    /// embedding dimensions.
    ///
    /// - Parameters:
    ///   - useCase: The intended application
    ///   - enableCache: Whether to enable persistent caching
    /// - Returns: A MiniLM-optimized pipeline configuration
    ///
    /// ## Example
    /// ```swift
    /// let config = ConfigurationFactory.forMiniLM(useCase: .semanticSearch)
    /// let model = try await LocalONNXModel.load(path: "all-MiniLM-L6-v2")
    /// ```
    public static func forMiniLM(
        useCase: EmbeddingConfiguration.UseCase = .semanticSearch,
        enableCache: Bool = false
    ) -> PipelineConfiguration {
        let embeddingConfig = EmbeddingConfiguration.forMiniLM(useCase: useCase)

        return PipelineConfiguration(
            embedding: embeddingConfig,
            batch: BatchOptions(
                maxBatchSize: 64,  // MiniLM handles larger batches well
                dynamicBatching: true,
                sortByLength: true,
                timeout: 30.0,
                bucketSize: 16
            ),
            compute: ComputeConfiguration(
                useFusedKernels: true,
                adaptiveKernelSelection: true,
                maxResidentMemoryMB: 256
            ),
            cache: enableCache ? .default : nil,
            memoryBudget: nil
        )
    }

    /// Configuration optimized for BERT-base models (768 dimensions, ~110M parameters).
    ///
    /// BERT-base models are larger and more compute-intensive. This preset
    /// adjusts batch sizes and GPU thresholds for the larger embedding dimensions
    /// and model memory requirements.
    ///
    /// - Parameters:
    ///   - useCase: The intended application
    ///   - enableCache: Whether to enable persistent caching
    /// - Returns: A BERT-optimized pipeline configuration
    ///
    /// ## Example
    /// ```swift
    /// let config = ConfigurationFactory.forBERT(useCase: .rag)
    /// let model = try await LocalCoreMLModel.load(path: "bert-base-uncased.mlpackage")
    /// ```
    public static func forBERT(
        useCase: EmbeddingConfiguration.UseCase = .semanticSearch,
        enableCache: Bool = false
    ) -> PipelineConfiguration {
        let embeddingConfig = EmbeddingConfiguration.forBERT(useCase: useCase)

        return PipelineConfiguration(
            embedding: embeddingConfig,
            batch: BatchOptions(
                maxBatchSize: 32,  // Smaller batches for larger model
                dynamicBatching: true,
                sortByLength: true,
                timeout: 60.0,
                bucketSize: 32,
                maxBatchTokens: 8192  // Limit total tokens
            ),
            compute: ComputeConfiguration(
                useFusedKernels: true,
                adaptiveKernelSelection: true,
                maxResidentMemoryMB: 512  // More memory for larger embeddings
            ),
            cache: enableCache ? .default : nil,
            memoryBudget: nil
        )
    }

    /// Configuration optimized for large embedding models (1024+ dimensions).
    ///
    /// For models like instructor-xl, e5-large, or custom models with
    /// high-dimensional embeddings. Adjusts memory and batch settings
    /// for the larger vector sizes.
    ///
    /// - Parameters:
    ///   - dimensions: Expected embedding dimensions (default: 1024)
    ///   - maxTokens: Maximum input tokens (default: 512)
    ///   - enableCache: Whether to enable persistent caching
    /// - Returns: A large model-optimized pipeline configuration
    public static func forLargeModel(
        dimensions: Int = 1024,
        maxTokens: Int = 512,
        enableCache: Bool = false
    ) -> PipelineConfiguration {
        // Adjust batch size based on dimensions
        let maxBatch = max(8, 32 - (dimensions / 128))

        return PipelineConfiguration(
            embedding: EmbeddingConfiguration(
                maxTokens: maxTokens,
                truncationStrategy: .end,
                paddingStrategy: .batch,
                includeSpecialTokens: true,
                poolingStrategy: .mean,
                normalizeOutput: true,
                inferenceDevice: .auto,
                minElementsForGPU: 4096
            ),
            batch: BatchOptions(
                maxBatchSize: maxBatch,
                dynamicBatching: true,
                sortByLength: true,
                timeout: 120.0,
                bucketSize: 16,
                maxBatchTokens: maxTokens * maxBatch
            ),
            compute: ComputeConfiguration(
                useFusedKernels: true,
                adaptiveKernelSelection: true,
                maxResidentMemoryMB: 1024  // More memory for large vectors
            ),
            cache: enableCache ? CacheConfiguration(
                maxEntries: 25_000,  // Fewer entries for larger vectors
                maxSizeBytes: Int64(500) * 1024 * 1024,
                enableSemanticDedup: false,
                autoEvict: true,
                ttlSeconds: 0,
                enableWAL: true
            ) : nil,
            memoryBudget: nil
        )
    }
}

// MARK: - PipelineConfiguration Convenience Methods

extension PipelineConfiguration {

    /// Creates an `AdaptiveBatcherConfig` from this pipeline configuration.
    ///
    /// Use this when initializing an `AdaptiveBatcher` with settings from
    /// a pipeline configuration preset.
    ///
    /// - Returns: An AdaptiveBatcherConfig with corresponding settings
    ///
    /// ## Example
    /// ```swift
    /// let config = ConfigurationFactory.highThroughput()
    /// let batcher = AdaptiveBatcher(model: model, config: config.toAdaptiveBatcherConfig())
    /// ```
    public func toAdaptiveBatcherConfig() -> AdaptiveBatcherConfig {
        var config = AdaptiveBatcherConfig()

        // Map batch options
        config.maxBatchSize = batch.maxBatchSize
        config.minBatchSize = batch.minBatchSize ?? 1
        config.autoFlush = batch.dynamicBatching
        config.batchOptions = batch

        // Map memory settings
        if let budget = memoryBudget {
            // Adjust batch sizes based on memory budget
            let budgetMB = budget / (1024 * 1024)
            config.batchSizeByPressure = [
                0.0...0.3: min(batch.maxBatchSize, budgetMB / 4),
                0.3...0.6: min(batch.maxBatchSize / 2, budgetMB / 8),
                0.6...0.8: min(batch.maxBatchSize / 4, budgetMB / 16),
                0.8...1.0: min(16, budgetMB / 32)
            ]
        }

        // Set timeout from batch options
        if let timeout = batch.timeout {
            config.maxLatency = min(timeout / 10, 0.5)  // Fraction of total timeout
        }

        return config
    }

    /// Returns a copy of this configuration with caching enabled.
    ///
    /// - Parameter cacheConfig: The cache configuration to use
    /// - Returns: A new pipeline configuration with caching enabled
    public func withCache(_ cacheConfig: CacheConfiguration = .default) -> PipelineConfiguration {
        PipelineConfiguration(
            embedding: embedding,
            batch: batch,
            compute: compute,
            cache: cacheConfig,
            memoryBudget: memoryBudget
        )
    }

    /// Returns a copy of this configuration with the specified memory budget.
    ///
    /// - Parameter budgetMB: Memory budget in megabytes
    /// - Returns: A new pipeline configuration with the specified budget
    public func withMemoryBudget(mb budgetMB: Int) -> PipelineConfiguration {
        PipelineConfiguration(
            embedding: embedding,
            batch: batch,
            compute: compute,
            cache: cache,
            memoryBudget: budgetMB * 1024 * 1024
        )
    }

    /// Returns a copy of this configuration with memory-efficient settings.
    ///
    /// - Returns: A new pipeline configuration using memory-efficient compute
    public func withMemoryEfficiency() -> PipelineConfiguration {
        PipelineConfiguration(
            embedding: embedding,
            batch: batch,
            compute: .memoryEfficient(),
            cache: cache,
            memoryBudget: memoryBudget
        )
    }
}

// MARK: - CustomStringConvertible

extension PipelineConfiguration: CustomStringConvertible {
    public var description: String {
        var parts: [String] = []

        parts.append("embedding: maxTokens=\(embedding.maxTokens), pool=\(embedding.poolingStrategy)")
        parts.append("batch: max=\(batch.maxBatchSize), dynamic=\(batch.dynamicBatching)")
        parts.append("compute: fused=\(compute.useFusedKernels), memory=\(compute.maxResidentMemoryMB)MB")

        if let cache = cache {
            parts.append("cache: entries=\(cache.maxEntries)")
        }

        if let budget = memoryBudget {
            parts.append("memoryBudget: \(budget / (1024 * 1024))MB")
        }

        return "PipelineConfiguration(\(parts.joined(separator: ", ")))"
    }
}

extension ComputeConfiguration: CustomStringConvertible {
    public var description: String {
        "ComputeConfiguration(fused: \(useFusedKernels), adaptive: \(adaptiveKernelSelection), memory: \(maxResidentMemoryMB)MB)"
    }
}
