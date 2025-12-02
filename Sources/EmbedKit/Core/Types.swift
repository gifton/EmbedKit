// EmbedKit - Core Types

import Foundation

// MARK: - Model Identification

/// Unique identifier for an embedding model.
///
/// Used as a key in `ModelManager` for model lookup and lifecycle management.
/// Provider, name, and version are required and must be non-empty.
///
/// ## Example
/// ```swift
/// let modelID = ModelID(
///     provider: "apple",
///     name: "text-embedding",
///     version: "1.0.0",
///     variant: "base"
/// )
/// print(modelID)  // "apple/text-embedding-base@1.0.0"
/// ```
public struct ModelID: Hashable, Codable, CustomStringConvertible, Sendable {
    /// Model provider (e.g., "apple", "openai", "huggingface"). Must be non-empty.
    public let provider: String

    /// Model name (e.g., "text-embedding", "all-MiniLM-L6-v2"). Must be non-empty.
    public let name: String

    /// Model version (e.g., "1.0.0", "v2"). Must be non-empty.
    public let version: String

    /// Optional model variant (e.g., "base", "small", "large").
    public let variant: String?

    public var description: String {
        let v = variant.map { "-\($0)" } ?? ""
        return "\(provider)/\(name)\(v)@\(version)"
    }

    /// Creates a model identifier with validation.
    ///
    /// - Parameters:
    ///   - provider: Model provider (must be non-empty)
    ///   - name: Model name (must be non-empty)
    ///   - version: Model version (must be non-empty)
    ///   - variant: Optional model variant
    ///
    /// - Precondition: `provider`, `name`, and `version` must be non-empty strings.
    public init(provider: String, name: String, version: String, variant: String? = nil) {
        precondition(!provider.isEmpty, "ModelID provider must not be empty")
        precondition(!name.isEmpty, "ModelID name must not be empty")
        precondition(!version.isEmpty, "ModelID version must not be empty")

        self.provider = provider
        self.name = name
        self.version = version
        self.variant = variant
    }

    /// Creates a ModelID from a description string.
    ///
    /// Expected format: "provider/name@version" or "provider/name-variant@version"
    ///
    /// - Parameter description: String in format "provider/name[-variant]@version"
    /// - Returns: ModelID if parsing succeeds, nil otherwise
    public init?(fromDescription description: String) {
        // Parse: "provider/name[-variant]@version"
        guard let atIndex = description.lastIndex(of: "@"),
              let slashIndex = description.firstIndex(of: "/") else {
            return nil
        }

        let provider = String(description[..<slashIndex])
        let version = String(description[description.index(after: atIndex)...])
        let middlePart = String(description[description.index(after: slashIndex)..<atIndex])

        // Check for variant (last hyphen in middle part)
        let name: String
        let variant: String?
        if let hyphenIndex = middlePart.lastIndex(of: "-") {
            name = String(middlePart[..<hyphenIndex])
            variant = String(middlePart[middlePart.index(after: hyphenIndex)...])
        } else {
            name = middlePart
            variant = nil
        }

        guard !provider.isEmpty, !name.isEmpty, !version.isEmpty else {
            return nil
        }

        self.provider = provider
        self.name = name
        self.version = version
        self.variant = variant
    }
}

// MARK: - Embedding Types
public struct Embedding: Sendable {
    public let vector: [Float]
    public let metadata: EmbeddingMetadata

    public var dimensions: Int { vector.count }

    public init(vector: [Float], metadata: EmbeddingMetadata) {
        self.vector = vector
        self.metadata = metadata
    }

    /// L2 magnitude (Euclidean norm) of the embedding vector.
    /// Uses Accelerate/vDSP for SIMD-optimized computation.
    public var magnitude: Float {
        AccelerateBLAS.magnitude(vector)
    }

    /// Returns an L2-normalized copy of this embedding.
    /// Uses Accelerate/vDSP for SIMD-optimized computation.
    public func normalized() -> Embedding {
        let v = AccelerateBLAS.normalize(vector)
        return Embedding(vector: v, metadata: metadata)
    }

    /// Computes cosine similarity to another embedding.
    /// Uses Accelerate/vDSP for SIMD-optimized computation.
    /// - Parameter other: The other embedding to compare against
    /// - Returns: Cosine similarity in range [-1, 1], or 0 if dimensions don't match
    public func similarity(to other: Embedding) -> Float {
        guard dimensions == other.dimensions else { return 0 }
        return AccelerateBLAS.cosineSimilarity(vector, other.vector)
    }

    /// Computes cosine distance to another embedding (1 - similarity).
    /// Uses Accelerate/vDSP for SIMD-optimized computation.
    public func distance(to other: Embedding) -> Float {
        guard dimensions == other.dimensions else { return 1 }
        return AccelerateBLAS.cosineDistance(vector, other.vector)
    }

    /// Computes Euclidean distance to another embedding.
    /// Uses Accelerate/vDSP for SIMD-optimized computation.
    public func euclideanDistance(to other: Embedding) -> Float {
        guard dimensions == other.dimensions else { return .greatestFiniteMagnitude }
        return AccelerateBLAS.euclideanDistance(vector, other.vector)
    }
}

public struct EmbeddingMetadata: Codable, Sendable {
    public let modelID: ModelID
    public let tokenCount: Int
    public let processingTime: TimeInterval
    public let normalized: Bool
    public let poolingStrategy: PoolingStrategy
    public let truncated: Bool
    public let custom: [String: String]

    public init(
        modelID: ModelID,
        tokenCount: Int,
        processingTime: TimeInterval,
        normalized: Bool = true,
        poolingStrategy: PoolingStrategy = .mean,
        truncated: Bool = false,
        custom: [String: String] = [:]
    ) {
        self.modelID = modelID
        self.tokenCount = tokenCount
        self.processingTime = processingTime
        self.normalized = normalized
        self.poolingStrategy = poolingStrategy
        self.truncated = truncated
        self.custom = custom
    }
}

// MARK: - Configuration

/// Configuration for embedding operations.
///
/// All properties are immutable after creation. Use the initializer to customize values.
///
/// ## Example
/// ```swift
/// let config = EmbeddingConfiguration(
///     maxTokens: 256,
///     poolingStrategy: .cls,
///     inferenceDevice: .ane
/// )
/// ```
public struct EmbeddingConfiguration: Sendable {
    /// Maximum number of tokens per input. Must be > 0.
    public let maxTokens: Int

    /// How to truncate inputs that exceed `maxTokens`.
    public let truncationStrategy: TruncationStrategy

    /// How to pad inputs shorter than the batch maximum.
    public let paddingStrategy: PaddingStrategy

    /// Whether to include special tokens (CLS, SEP) in tokenization.
    public let includeSpecialTokens: Bool

    /// Strategy for pooling token embeddings into a single vector.
    public let poolingStrategy: PoolingStrategy

    /// Whether to L2-normalize output embeddings.
    public let normalizeOutput: Bool

    /// Target device for model inference (CoreML compute units).
    ///
    /// - Note: For GPU acceleration of search/distance operations, configure
    ///   `ComputePreference` on `IndexConfiguration` or `AccelerationManager` instead.
    public let inferenceDevice: ComputeDevice

    /// Minimum number of elements (sequenceLength × dimensions) before GPU acceleration
    /// is used for pooling and normalization operations.
    ///
    /// - GPU pooling: Used when `sequenceLength * dimensions >= minElementsForGPU`
    /// - GPU normalization: Used when `batchSize * dimensions >= minElementsForGPU`
    ///
    /// Default is 8192 (e.g., 22 tokens × 384 dims, or 8 vectors × 1024 dims).
    /// Set to 0 to always prefer GPU, or `Int.max` to always use CPU.
    public let minElementsForGPU: Int

    /// Creates an embedding configuration with the specified options.
    ///
    /// - Parameters:
    ///   - maxTokens: Maximum tokens per input (default: 512, must be > 0)
    ///   - truncationStrategy: How to truncate long inputs (default: .end)
    ///   - paddingStrategy: How to pad short inputs (default: .none)
    ///   - includeSpecialTokens: Include CLS/SEP tokens (default: true)
    ///   - poolingStrategy: Token pooling strategy (default: .mean)
    ///   - normalizeOutput: L2-normalize embeddings (default: true)
    ///   - inferenceDevice: CoreML compute device (default: .auto)
    ///   - minElementsForGPU: GPU threshold for pooling/normalization (default: 8192)
    public init(
        maxTokens: Int = 512,
        truncationStrategy: TruncationStrategy = .end,
        paddingStrategy: PaddingStrategy = .none,
        includeSpecialTokens: Bool = true,
        poolingStrategy: PoolingStrategy = .mean,
        normalizeOutput: Bool = true,
        inferenceDevice: ComputeDevice = .auto,
        minElementsForGPU: Int = 8192
    ) {
        precondition(maxTokens > 0, "maxTokens must be > 0")
        precondition(minElementsForGPU >= 0, "minElementsForGPU must be >= 0")

        self.maxTokens = maxTokens
        self.truncationStrategy = truncationStrategy
        self.paddingStrategy = paddingStrategy
        self.includeSpecialTokens = includeSpecialTokens
        self.poolingStrategy = poolingStrategy
        self.normalizeOutput = normalizeOutput
        self.inferenceDevice = inferenceDevice
        self.minElementsForGPU = minElementsForGPU
    }

    /// Default configuration suitable for most use cases.
    public static let `default` = EmbeddingConfiguration()

    /// Configuration optimized for throughput (larger batches, GPU preference).
    public static let performant = EmbeddingConfiguration(
        maxTokens: 512,
        paddingStrategy: .batch,
        inferenceDevice: .auto,
        minElementsForGPU: 4096
    )

    /// Deprecated: Use `inferenceDevice` instead.
    @available(*, deprecated, renamed: "inferenceDevice")
    public var preferredDevice: ComputeDevice { inferenceDevice }

    /// Returns a copy of this configuration with the specified inference device.
    ///
    /// - Parameter device: The new inference device
    /// - Returns: A new configuration with the updated device
    public func with(inferenceDevice device: ComputeDevice) -> EmbeddingConfiguration {
        EmbeddingConfiguration(
            maxTokens: maxTokens,
            truncationStrategy: truncationStrategy,
            paddingStrategy: paddingStrategy,
            includeSpecialTokens: includeSpecialTokens,
            poolingStrategy: poolingStrategy,
            normalizeOutput: normalizeOutput,
            inferenceDevice: device,
            minElementsForGPU: minElementsForGPU
        )
    }

    /// Returns a copy of this configuration with the specified GPU threshold.
    ///
    /// - Parameter threshold: The new minimum elements for GPU
    /// - Returns: A new configuration with the updated threshold
    public func with(minElementsForGPU threshold: Int) -> EmbeddingConfiguration {
        EmbeddingConfiguration(
            maxTokens: maxTokens,
            truncationStrategy: truncationStrategy,
            paddingStrategy: paddingStrategy,
            includeSpecialTokens: includeSpecialTokens,
            poolingStrategy: poolingStrategy,
            normalizeOutput: normalizeOutput,
            inferenceDevice: inferenceDevice,
            minElementsForGPU: threshold
        )
    }
}

/// Target device for model inference operations.
///
/// Controls which compute units CoreML uses for model inference.
/// This is different from `ComputePreference` (used for search/distance acceleration).
///
/// - `cpu`: Force CPU-only inference. Most compatible, but slowest.
/// - `gpu`: Prefer GPU for inference. Good for large models.
/// - `ane`: Prefer Apple Neural Engine. Best efficiency on supported models.
/// - `auto`: Let CoreML choose the best device (recommended).
///
/// ## Usage
/// ```swift
/// var config = EmbeddingConfiguration()
/// config.inferenceDevice = .ane  // Use Neural Engine for inference
/// ```
///
/// - Note: For GPU acceleration of search/distance operations, use `ComputePreference` instead.
public enum ComputeDevice: String, CaseIterable, Codable, Sendable {
    case cpu
    case gpu
    case ane
    case auto
}
public enum PoolingStrategy: String, CaseIterable, Codable, Sendable { case mean, max, cls, attention }
public enum TruncationStrategy: String, CaseIterable, Codable, Sendable { case none, end, start, middle }
public enum PaddingStrategy: String, CaseIterable, Codable, Sendable { case none, max, batch }

// MARK: - Tokenization Types

/// Configuration for tokenization operations.
///
/// All properties are immutable after creation.
public struct TokenizerConfig: Sendable {
    /// Maximum sequence length. Must be > 0.
    public let maxLength: Int

    /// How to truncate sequences exceeding `maxLength`.
    public let truncation: TruncationStrategy

    /// How to pad sequences shorter than the target length.
    public let padding: PaddingStrategy

    /// Whether to add special tokens (CLS, SEP, etc.).
    public let addSpecialTokens: Bool

    /// Whether to return character offset mappings.
    public let returnOffsets: Bool

    /// Creates a tokenizer configuration with the specified options.
    ///
    /// - Parameters:
    ///   - maxLength: Maximum sequence length (default: 512, must be > 0)
    ///   - truncation: Truncation strategy (default: .end)
    ///   - padding: Padding strategy (default: .none)
    ///   - addSpecialTokens: Add special tokens (default: true)
    ///   - returnOffsets: Return offset mappings (default: false)
    public init(
        maxLength: Int = 512,
        truncation: TruncationStrategy = .end,
        padding: PaddingStrategy = .none,
        addSpecialTokens: Bool = true,
        returnOffsets: Bool = false
    ) {
        precondition(maxLength > 0, "maxLength must be > 0")

        self.maxLength = maxLength
        self.truncation = truncation
        self.padding = padding
        self.addSpecialTokens = addSpecialTokens
        self.returnOffsets = returnOffsets
    }

    /// Default configuration for most tokenizers.
    public static let `default` = TokenizerConfig()
}

public struct TokenizedText: Sendable {
    public let ids: [Int]
    public let tokens: [String]
    public let attentionMask: [Int]
    public var length: Int { ids.count }
}

public struct SpecialTokens: Sendable {
    public let cls: Token?
    public let sep: Token?
    public let pad: Token?
    public let unk: Token?
    public let mask: Token?
    public let bos: Token?
    public let eos: Token?

    public struct Token: Sendable { public let text: String; public let id: Int }
    public init(cls: Token? = nil, sep: Token? = nil, pad: Token? = nil, unk: Token? = nil, mask: Token? = nil, bos: Token? = nil, eos: Token? = nil) {
        self.cls = cls; self.sep = sep; self.pad = pad; self.unk = unk; self.mask = mask; self.bos = bos; self.eos = eos
    }
}

// MARK: - Batch Options

/// Configuration for batch embedding operations.
///
/// All properties are immutable after creation.
///
/// ## Batch Size Limits
/// - `maxBatchSize`: Hard limit on number of items per batch
/// - `maxBatchTokens`: Alternative limit based on total tokens (optional)
/// - If both are set, the more restrictive limit applies
///
/// ## Dynamic Batching
/// When `dynamicBatching` is true:
/// - `sortByLength`: Sort inputs by length to minimize padding
/// - `bucketSize`: Group similar-length inputs together
/// - `minBatchSize`: Minimum items before processing (optional)
/// - `maxPaddingRatio`: Maximum padding ratio before splitting batch (optional)
public struct BatchOptions: Sendable {
    /// Maximum number of items per batch. Must be > 0.
    public let maxBatchSize: Int

    /// Whether to dynamically adjust batch composition.
    public let dynamicBatching: Bool

    /// Whether to sort inputs by length to minimize padding.
    public let sortByLength: Bool

    /// Timeout for batch processing (nil = no timeout).
    public let timeout: TimeInterval?

    /// Size of length buckets when sorting by length. Must be > 0.
    public let bucketSize: Int

    /// Maximum total tokens per batch (optional, alternative to maxBatchSize).
    public let maxBatchTokens: Int?

    /// Number of concurrent tokenization tasks.
    /// Defaults to the number of active processors for parallel tokenization.
    /// Set to 1 for sequential tokenization.
    public let tokenizationConcurrency: Int

    /// Minimum batch size before processing (optional, for dynamic batching).
    public let minBatchSize: Int?

    /// Maximum ratio of padding tokens (optional, for dynamic batching).
    public let maxPaddingRatio: Double?

    /// Creates batch options with the specified configuration.
    ///
    /// - Parameters:
    ///   - maxBatchSize: Maximum items per batch (default: 32, must be > 0)
    ///   - dynamicBatching: Enable dynamic batching (default: true)
    ///   - sortByLength: Sort by length to reduce padding (default: true)
    ///   - timeout: Processing timeout (default: nil)
    ///   - bucketSize: Length bucket size (default: 16, must be > 0)
    ///   - maxBatchTokens: Max tokens per batch (default: nil)
    ///   - tokenizationConcurrency: Parallel tokenization tasks (default: CPU count)
    ///   - minBatchSize: Minimum batch size for dynamic batching (default: nil)
    ///   - maxPaddingRatio: Maximum padding ratio (default: nil)
    public init(
        maxBatchSize: Int = 32,
        dynamicBatching: Bool = true,
        sortByLength: Bool = true,
        timeout: TimeInterval? = nil,
        bucketSize: Int = 16,
        maxBatchTokens: Int? = nil,
        tokenizationConcurrency: Int = ProcessInfo.processInfo.activeProcessorCount,
        minBatchSize: Int? = nil,
        maxPaddingRatio: Double? = nil
    ) {
        precondition(maxBatchSize > 0, "maxBatchSize must be > 0")
        precondition(bucketSize > 0, "bucketSize must be > 0")
        precondition(tokenizationConcurrency > 0, "tokenizationConcurrency must be > 0")
        if let minBatch = minBatchSize {
            precondition(minBatch > 0 && minBatch <= maxBatchSize, "minBatchSize must be > 0 and <= maxBatchSize")
        }
        if let maxTokens = maxBatchTokens {
            precondition(maxTokens > 0, "maxBatchTokens must be > 0")
        }
        if let ratio = maxPaddingRatio {
            precondition(ratio >= 0 && ratio <= 1, "maxPaddingRatio must be in [0, 1]")
        }

        self.maxBatchSize = maxBatchSize
        self.dynamicBatching = dynamicBatching
        self.sortByLength = sortByLength
        self.timeout = timeout
        self.bucketSize = bucketSize
        self.maxBatchTokens = maxBatchTokens
        self.tokenizationConcurrency = tokenizationConcurrency
        self.minBatchSize = minBatchSize
        self.maxPaddingRatio = maxPaddingRatio
    }

    /// Default batch options for most use cases.
    public static let `default` = BatchOptions()

    /// Options optimized for throughput (larger batches).
    public static let highThroughput = BatchOptions(
        maxBatchSize: 64,
        dynamicBatching: true,
        sortByLength: true,
        bucketSize: 32
    )

    /// Options for low-latency (smaller batches, no sorting).
    public static let lowLatency = BatchOptions(
        maxBatchSize: 8,
        dynamicBatching: false,
        sortByLength: false
    )
}

// MARK: - Configuration Factories

extension EmbeddingConfiguration {

    // MARK: - Use Case Factories

    /// Configuration optimized for semantic search applications.
    ///
    /// Designed for searching through document collections using embedding similarity.
    /// Uses batch padding for efficient processing of multiple queries and L2 normalization
    /// for consistent similarity metrics.
    ///
    /// - Parameters:
    ///   - maxLength: Maximum token length for inputs (default: 512)
    ///   - normalize: Whether to L2-normalize outputs (default: true)
    /// - Returns: Configuration optimized for semantic search
    ///
    /// ## Example
    /// ```swift
    /// let config = EmbeddingConfiguration.forSemanticSearch(maxLength: 256)
    /// let embeddings = try await model.embed(queries, config: config)
    /// ```
    public static func forSemanticSearch(
        maxLength: Int = 512,
        normalize: Bool = true
    ) -> EmbeddingConfiguration {
        EmbeddingConfiguration(
            maxTokens: maxLength,
            truncationStrategy: .end,
            paddingStrategy: .batch,
            includeSpecialTokens: true,
            poolingStrategy: .mean,
            normalizeOutput: normalize
        )
    }

    /// Configuration optimized for RAG (Retrieval-Augmented Generation).
    ///
    /// Designed for chunking and embedding documents in RAG pipelines.
    /// Uses smaller chunk sizes and batch padding for efficient processing.
    /// Always normalizes outputs for consistent similarity comparisons.
    ///
    /// - Parameter chunkSize: Maximum tokens per document chunk (default: 256)
    /// - Returns: Configuration optimized for RAG applications
    ///
    /// ## Example
    /// ```swift
    /// let config = EmbeddingConfiguration.forRAG(chunkSize: 384)
    /// let chunks = splitIntoChunks(document, maxTokens: 384)
    /// let embeddings = try await model.embed(chunks, config: config)
    /// ```
    public static func forRAG(
        chunkSize: Int = 256
    ) -> EmbeddingConfiguration {
        EmbeddingConfiguration(
            maxTokens: chunkSize,
            truncationStrategy: .end,
            paddingStrategy: .batch,
            includeSpecialTokens: true,
            poolingStrategy: .mean,
            normalizeOutput: true
        )
    }

    /// Configuration optimized for clustering and classification tasks.
    ///
    /// Uses shorter sequences and batch padding for efficient processing of labeled data.
    /// Suitable for k-means clustering, hierarchical clustering, and classification.
    ///
    /// - Parameter maxLength: Maximum token length for inputs (default: 128)
    /// - Returns: Configuration optimized for clustering/classification
    ///
    /// ## Example
    /// ```swift
    /// let config = EmbeddingConfiguration.forClustering()
    /// let embeddings = try await model.embed(labels, config: config)
    /// let clusters = kMeans(embeddings, k: 10)
    /// ```
    public static func forClustering(
        maxLength: Int = 128
    ) -> EmbeddingConfiguration {
        EmbeddingConfiguration(
            maxTokens: maxLength,
            truncationStrategy: .end,
            paddingStrategy: .batch,
            includeSpecialTokens: true,
            poolingStrategy: .mean,
            normalizeOutput: true
        )
    }

    /// Configuration optimized for similarity comparison between text pairs.
    ///
    /// Designed for computing semantic similarity between question-answer pairs,
    /// duplicate detection, or paraphrase identification.
    ///
    /// - Parameter maxLength: Maximum token length for inputs (default: 256)
    /// - Returns: Configuration optimized for similarity tasks
    ///
    /// ## Example
    /// ```swift
    /// let config = EmbeddingConfiguration.forSimilarity()
    /// let [query, doc] = try await model.embed([queryText, docText], config: config)
    /// let similarity = query.similarity(to: doc)
    /// ```
    public static func forSimilarity(
        maxLength: Int = 256
    ) -> EmbeddingConfiguration {
        EmbeddingConfiguration(
            maxTokens: maxLength,
            truncationStrategy: .end,
            paddingStrategy: .batch,
            includeSpecialTokens: true,
            poolingStrategy: .mean,
            normalizeOutput: true
        )
    }

    /// Configuration for embedding full documents.
    ///
    /// Supports longer sequences (up to 2048 tokens) and uses no padding for
    /// variable-length documents. Best for encoding complete articles or papers.
    ///
    /// - Parameter maxLength: Maximum token length for documents (default: 2048)
    /// - Returns: Configuration optimized for document embeddings
    ///
    /// ## Example
    /// ```swift
    /// let config = EmbeddingConfiguration.forDocuments(maxLength: 1024)
    /// let embedding = try await model.embed(fullArticle, config: config)
    /// ```
    ///
    /// - Note: Long sequences may require significant memory and compute resources.
    public static func forDocuments(
        maxLength: Int = 2048
    ) -> EmbeddingConfiguration {
        EmbeddingConfiguration(
            maxTokens: maxLength,
            truncationStrategy: .end,
            paddingStrategy: .none,
            includeSpecialTokens: true,
            poolingStrategy: .mean,
            normalizeOutput: true
        )
    }

    /// Configuration for short text like queries, titles, or labels.
    ///
    /// Optimized for very short inputs with fixed-length padding for consistent
    /// batch processing. Ideal for search queries, product titles, or category labels.
    ///
    /// - Parameter maxLength: Maximum token length for short text (default: 64)
    /// - Returns: Configuration optimized for short text
    ///
    /// ## Example
    /// ```swift
    /// let config = EmbeddingConfiguration.forShortText()
    /// let embeddings = try await model.embed(queries, config: config)
    /// ```
    public static func forShortText(
        maxLength: Int = 64
    ) -> EmbeddingConfiguration {
        EmbeddingConfiguration(
            maxTokens: maxLength,
            truncationStrategy: .end,
            paddingStrategy: .max,
            includeSpecialTokens: true,
            poolingStrategy: .mean,
            normalizeOutput: true
        )
    }

    // MARK: - Model-Specific Factories

    /// Configuration optimized for MiniLM models (384 dimensions).
    ///
    /// MiniLM models are lightweight and efficient, typically supporting up to 256-512 tokens.
    /// This factory selects appropriate defaults based on the intended use case.
    ///
    /// - Parameter useCase: The intended application (default: .semanticSearch)
    /// - Returns: Configuration tuned for MiniLM architecture
    ///
    /// ## Example
    /// ```swift
    /// let config = EmbeddingConfiguration.forMiniLM(useCase: .rag)
    /// let model = try await LocalONNXModel.load(
    ///     modelPath: "all-MiniLM-L6-v2",
    ///     config: config
    /// )
    /// ```
    public static func forMiniLM(
        useCase: UseCase = .semanticSearch
    ) -> EmbeddingConfiguration {
        switch useCase {
        case .semanticSearch:
            return .forSemanticSearch(maxLength: 256)
        case .rag:
            return .forRAG(chunkSize: 256)
        case .clustering:
            return .forClustering(maxLength: 128)
        case .similarity:
            return .forSimilarity(maxLength: 256)
        }
    }

    /// Configuration optimized for BERT-base models (768 dimensions).
    ///
    /// BERT-base models support up to 512 tokens and provide robust embeddings
    /// across various NLP tasks. This factory provides task-specific configurations.
    ///
    /// - Parameter useCase: The intended application (default: .semanticSearch)
    /// - Returns: Configuration tuned for BERT-base architecture
    ///
    /// ## Example
    /// ```swift
    /// let config = EmbeddingConfiguration.forBERT(useCase: .semanticSearch)
    /// let model = try await CoreMLModel.load(
    ///     modelPath: "bert-base-uncased",
    ///     config: config
    /// )
    /// ```
    public static func forBERT(
        useCase: UseCase = .semanticSearch
    ) -> EmbeddingConfiguration {
        switch useCase {
        case .semanticSearch:
            return .forSemanticSearch(maxLength: 512)
        case .rag:
            return .forRAG(chunkSize: 384)
        case .clustering:
            return .forClustering(maxLength: 256)
        case .similarity:
            return .forSimilarity(maxLength: 512)
        }
    }

    /// Common use cases for factory method selection.
    ///
    /// Each case represents a distinct NLP application with different
    /// optimization priorities for sequence length, padding, and processing.
    public enum UseCase: String, Sendable, CaseIterable {
        /// Semantic search through document collections
        case semanticSearch

        /// Retrieval-Augmented Generation (RAG) pipelines
        case rag

        /// Clustering and classification tasks
        case clustering

        /// Pairwise similarity comparisons
        case similarity
    }
}

