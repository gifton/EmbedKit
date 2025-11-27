// EmbedKit - Core Types

import Foundation

// MARK: - Model Identification
public struct ModelID: Hashable, Codable, CustomStringConvertible, Sendable {
    public let provider: String
    public let name: String
    public let version: String
    public let variant: String?

    public var description: String {
        let v = variant.map { "-\($0)" } ?? ""
        return "\(provider)/\(name)\(v)@\(version)"
    }

    public init(provider: String, name: String, version: String, variant: String? = nil) {
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
public struct EmbeddingConfiguration: Sendable {
    public var maxTokens: Int = 512
    public var truncationStrategy: TruncationStrategy = .end
    public var paddingStrategy: PaddingStrategy = .none
    public var includeSpecialTokens: Bool = true

    public var poolingStrategy: PoolingStrategy = .mean
    public var normalizeOutput: Bool = true

    public var preferredDevice: ComputeDevice = .auto

    /// Minimum number of elements (sequenceLength × dimensions) before GPU acceleration
    /// is used for pooling and normalization operations.
    ///
    /// - GPU pooling: Used when `sequenceLength * dimensions >= minElementsForGPU`
    /// - GPU normalization: Used when `batchSize * dimensions >= minElementsForGPU`
    ///
    /// Default is 8192 (e.g., 22 tokens × 384 dims, or 8 vectors × 1024 dims).
    /// Set to 0 to always prefer GPU, or `Int.max` to always use CPU.
    public var minElementsForGPU: Int = 8192

    public init() {}
}

public enum ComputeDevice: String, CaseIterable, Sendable { case cpu, gpu, ane, auto }
public enum PoolingStrategy: String, CaseIterable, Codable, Sendable { case mean, max, cls, attention }
public enum TruncationStrategy: String, CaseIterable, Codable, Sendable { case none, end, start, middle }
public enum PaddingStrategy: String, CaseIterable, Codable, Sendable { case none, max, batch }

// MARK: - Tokenization Types
public struct TokenizerConfig: Sendable {
    public var maxLength: Int = 512
    public var truncation: TruncationStrategy = .end
    public var padding: PaddingStrategy = .none
    public var addSpecialTokens: Bool = true
    public var returnOffsets: Bool = false
    public init() {}
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
public struct BatchOptions: Sendable {
    public var maxBatchSize: Int = 32
    public var dynamicBatching: Bool = true
    public var sortByLength: Bool = true
    public var timeout: TimeInterval? = nil
    public var bucketSize: Int = 16
    public var maxBatchTokens: Int? = nil

    /// Number of concurrent tokenization tasks.
    /// Defaults to the number of active processors for parallel tokenization.
    /// Set to 1 for sequential tokenization.
    public var tokenizationConcurrency: Int? = ProcessInfo.processInfo.activeProcessorCount

    public var minBatchSize: Int? = nil
    public var maxPaddingRatio: Double? = nil
    public init() {}
}

