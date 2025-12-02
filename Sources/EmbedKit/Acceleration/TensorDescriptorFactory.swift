// EmbedKit - Tensor Descriptor Factory
//
// Factory methods for creating tensor descriptors and parameter structs
// for common embedding operations. Provides convenient configuration for
// Metal 4 tensor kernels.

import Foundation

#if canImport(Metal)
@preconcurrency import Metal

// MARK: - Tensor Descriptor Factory

/// Factory for creating tensor descriptors and parameters for common operations.
///
/// `TensorDescriptorFactory` provides convenience methods for creating
/// properly configured tensors and parameter structs for embedding operations.
///
/// ## Usage
/// ```swift
/// // Create tensors for a pooling operation
/// let (input, output, params) = try TensorDescriptorFactory.forPooling(
///     device: device,
///     batchSize: 32,
///     sequenceLength: 128,
///     dimensions: 384,
///     strategy: .mean
/// )
/// ```
public struct TensorDescriptorFactory {

    // MARK: - Common Embedding Dimensions

    /// Standard embedding dimensions for common models.
    public enum ModelDimensions: Int, Sendable {
        /// MiniLM-L6 (384 dimensions)
        case miniLM = 384

        /// BERT-base, all-MiniLM-L12 (768 dimensions)
        case bertBase = 768

        /// BERT-large (1024 dimensions)
        case bertLarge = 1024

        /// OpenAI ada-002 / text-embedding-3-small (1536 dimensions)
        case ada002 = 1536

        /// OpenAI text-embedding-3-large (3072 dimensions)
        case embedding3Large = 3072
    }

    /// Standard sequence lengths for common models.
    public enum SequenceLength: Int, Sendable {
        /// Short sequences (128 tokens)
        case short = 128

        /// Standard BERT (512 tokens)
        case standard = 512

        /// Long context (2048 tokens)
        case long = 2048

        /// Maximum MiniLM (256 tokens)
        case miniLMMax = 256
    }

    // MARK: - Tensor Creation

    /// Create an empty embedding tensor with standard dimensions.
    ///
    /// - Parameters:
    ///   - device: Metal device
    ///   - batchSize: Number of embeddings
    ///   - model: Model dimensions preset
    /// - Returns: Empty EmbeddingTensor ready for data
    public static func embeddingTensor(
        device: MTLDevice,
        batchSize: Int,
        model: ModelDimensions = .miniLM
    ) throws -> EmbeddingTensor {
        try EmbeddingTensor(
            batchSize: batchSize,
            dimensions: model.rawValue,
            device: device
        )
    }

    /// Create an empty token embedding tensor with standard dimensions.
    ///
    /// - Parameters:
    ///   - device: Metal device
    ///   - batchSize: Number of sequences
    ///   - sequenceLength: Tokens per sequence
    ///   - model: Model dimensions preset
    /// - Returns: Empty TokenEmbeddingTensor ready for data
    public static func tokenTensor(
        device: MTLDevice,
        batchSize: Int,
        sequenceLength: SequenceLength = .short,
        model: ModelDimensions = .miniLM
    ) throws -> TokenEmbeddingTensor {
        try TokenEmbeddingTensor(
            batchSize: batchSize,
            sequenceLength: sequenceLength.rawValue,
            dimensions: model.rawValue,
            device: device
        )
    }

    // MARK: - Pooling Configuration

    /// Configuration for a pooling operation.
    public struct PoolingConfiguration: Sendable {
        /// Input token embeddings tensor
        public let input: TokenEmbeddingTensor
        /// Output pooled embeddings tensor
        public let output: EmbeddingTensor
        /// Parameters for the pooling kernel
        public let params: TensorPoolingParams
    }

    /// Create tensors and parameters for a pooling operation.
    ///
    /// - Parameters:
    ///   - device: Metal device
    ///   - batchSize: Number of sequences
    ///   - sequenceLength: Tokens per sequence
    ///   - dimensions: Embedding dimensions
    ///   - strategy: Pooling strategy
    /// - Returns: Configuration with input, output tensors and parameters
    public static func forPooling(
        device: MTLDevice,
        batchSize: Int,
        sequenceLength: Int,
        dimensions: Int,
        strategy: PoolingStrategy = .mean
    ) throws -> PoolingConfiguration {
        let input = try TokenEmbeddingTensor(
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            dimensions: dimensions,
            device: device
        )

        let output = try EmbeddingTensor(
            batchSize: batchSize,
            dimensions: dimensions,
            device: device
        )

        let params = TensorPoolingParams(
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            dimensions: dimensions,
            strategy: strategy
        )

        return PoolingConfiguration(
            input: input,
            output: output,
            params: params
        )
    }

    // MARK: - Normalization Configuration

    /// Configuration for a normalization operation.
    public struct NormalizationConfiguration: Sendable {
        /// Input/output embeddings tensor (in-place operation)
        public let tensor: EmbeddingTensor
        /// Parameters for the normalization kernel
        public let params: TensorNormalizationParams
    }

    /// Create tensors and parameters for a normalization operation.
    ///
    /// - Parameters:
    ///   - device: Metal device
    ///   - batchSize: Number of embeddings
    ///   - dimensions: Embedding dimensions
    /// - Returns: Configuration with tensor and parameters
    public static func forNormalization(
        device: MTLDevice,
        batchSize: Int,
        dimensions: Int
    ) throws -> NormalizationConfiguration {
        let tensor = try EmbeddingTensor(
            batchSize: batchSize,
            dimensions: dimensions,
            device: device
        )

        let params = TensorNormalizationParams(
            batchSize: batchSize,
            dimensions: dimensions
        )

        return NormalizationConfiguration(
            tensor: tensor,
            params: params
        )
    }

    // MARK: - Similarity Configuration

    /// Configuration for a similarity computation.
    public struct SimilarityConfiguration: Sendable {
        /// Query embeddings tensor
        public let queries: EmbeddingTensor
        /// Key embeddings tensor
        public let keys: EmbeddingTensor
        /// Output similarity matrix buffer
        public let output: MTLBuffer
        /// Parameters for the similarity kernel
        public let params: TensorSimilarityParams
    }

    /// Create tensors and parameters for similarity computation.
    ///
    /// - Parameters:
    ///   - device: Metal device
    ///   - queryBatchSize: Number of query embeddings
    ///   - keyBatchSize: Number of key embeddings
    ///   - dimensions: Embedding dimensions
    /// - Returns: Configuration with query, key tensors, output buffer, and parameters
    public static func forSimilarity(
        device: MTLDevice,
        queryBatchSize: Int,
        keyBatchSize: Int,
        dimensions: Int
    ) throws -> SimilarityConfiguration {
        let queries = try EmbeddingTensor(
            batchSize: queryBatchSize,
            dimensions: dimensions,
            device: device
        )

        let keys = try EmbeddingTensor(
            batchSize: keyBatchSize,
            dimensions: dimensions,
            device: device
        )

        let outputSize = queryBatchSize * keyBatchSize * MemoryLayout<Float>.stride
        guard let output = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            throw EmbedKitError.metalBufferFailed
        }

        let params = TensorSimilarityParams(
            queryBatchSize: queryBatchSize,
            keyBatchSize: keyBatchSize,
            dimensions: dimensions
        )

        return SimilarityConfiguration(
            queries: queries,
            keys: keys,
            output: output,
            params: params
        )
    }

    // MARK: - Fused Pool+Normalize Configuration

    /// Configuration for fused pooling + normalization.
    public struct FusedPoolNormConfiguration: Sendable {
        /// Input token embeddings tensor
        public let input: TokenEmbeddingTensor
        /// Output normalized pooled embeddings tensor
        public let output: EmbeddingTensor
        /// Parameters for the fused kernel
        public let params: FusedPoolNormParams
    }

    /// Create tensors and parameters for fused pool+normalize operation.
    ///
    /// - Parameters:
    ///   - device: Metal device
    ///   - batchSize: Number of sequences
    ///   - sequenceLength: Tokens per sequence
    ///   - dimensions: Embedding dimensions
    ///   - strategy: Pooling strategy
    ///   - normalize: Whether to L2 normalize
    /// - Returns: Configuration with input, output tensors and parameters
    public static func forFusedPoolNorm(
        device: MTLDevice,
        batchSize: Int,
        sequenceLength: Int,
        dimensions: Int,
        strategy: PoolingStrategy = .mean,
        normalize: Bool = true
    ) throws -> FusedPoolNormConfiguration {
        let input = try TokenEmbeddingTensor(
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            dimensions: dimensions,
            device: device
        )

        let output = try EmbeddingTensor(
            batchSize: batchSize,
            dimensions: dimensions,
            device: device
        )

        let params = FusedPoolNormParams(
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            dimensions: dimensions,
            strategy: strategy,
            normalize: normalize
        )

        return FusedPoolNormConfiguration(
            input: input,
            output: output,
            params: params
        )
    }

    // MARK: - Attention Mask

    /// Create an attention mask buffer.
    ///
    /// - Parameters:
    ///   - device: Metal device
    ///   - batchSize: Number of sequences
    ///   - sequenceLength: Tokens per sequence
    /// - Returns: Attention mask buffer (Int32 values, 1=attend, 0=ignore)
    public static func attentionMask(
        device: MTLDevice,
        batchSize: Int,
        sequenceLength: Int
    ) throws -> MTLBuffer {
        let size = batchSize * sequenceLength * MemoryLayout<Int32>.stride
        guard let buffer = device.makeBuffer(length: size, options: .storageModeShared) else {
            throw EmbedKitError.metalBufferFailed
        }
        // Initialize to all 1s (attend to all tokens by default)
        let ptr = buffer.contents().bindMemory(to: Int32.self, capacity: batchSize * sequenceLength)
        for i in 0..<(batchSize * sequenceLength) {
            ptr[i] = 1
        }
        return buffer
    }
}

#endif
