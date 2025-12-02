// EmbedKit - Metal 4 Tensor Types
//
// Typed tensor wrappers for Metal 4 GPU operations.
// These types provide type-safe, dimension-aware buffers for embedding operations.
//
// Metal 4 introduces native tensor types in MSL. These Swift types mirror that
// structure for optimal interoperability and enable fused kernel optimizations.

import Foundation

#if canImport(Metal)
@preconcurrency import Metal
#endif

// MARK: - Embedding Tensor (2D)

/// A 2D tensor optimized for batch embeddings with shape [batchSize, dimensions].
///
/// `EmbeddingTensor` provides a type-safe wrapper around Metal buffers for
/// storing multiple embedding vectors. It supports efficient GPU transfer and
/// is designed for use with Metal 4's tensor operations.
///
/// **Memory Layout**: Row-major contiguous storage
/// ```
/// [v0_d0, v0_d1, ..., v0_dN, v1_d0, v1_d1, ..., v1_dN, ...]
/// ```
///
/// **Usage**:
/// ```swift
/// // Create from embeddings
/// let tensor = try EmbeddingTensor(embeddings: vectors, device: device)
///
/// // Use in GPU operations
/// encoder.setBuffer(tensor.buffer, offset: 0, index: 0)
/// ```
///
/// - Note: Uses `@unchecked Sendable` because MTLBuffer is thread-safe but doesn't conform to Sendable.
public struct EmbeddingTensor: @unchecked Sendable {
    #if canImport(Metal)
    /// The underlying Metal buffer containing embedding data.
    public let buffer: MTLBuffer
    #endif

    /// Number of embeddings in the batch.
    public let batchSize: Int

    /// Dimensionality of each embedding vector.
    public let dimensions: Int

    /// Total number of float elements (batchSize × dimensions).
    public var totalElements: Int { batchSize * dimensions }

    /// Size in bytes of the tensor data.
    public var sizeInBytes: Int { totalElements * MemoryLayout<Float>.stride }

    /// Shape tuple for dimension introspection.
    public var shape: (batchSize: Int, dimensions: Int) {
        (batchSize, dimensions)
    }

    #if canImport(Metal)
    /// Create an empty tensor with specified dimensions.
    ///
    /// - Parameters:
    ///   - batchSize: Number of embedding vectors
    ///   - dimensions: Dimensionality of each vector
    ///   - device: Metal device for buffer allocation
    /// - Throws: `EmbedKitError.metalError` if buffer creation fails
    public init(batchSize: Int, dimensions: Int, device: MTLDevice) throws {
        self.batchSize = batchSize
        self.dimensions = dimensions

        let size = batchSize * dimensions * MemoryLayout<Float>.stride
        guard let buffer = device.makeBuffer(length: max(size, 1), options: .storageModeShared) else {
            throw EmbedKitError.metalBufferFailed
        }
        self.buffer = buffer
    }

    /// Create a tensor from existing embedding vectors.
    ///
    /// - Parameters:
    ///   - embeddings: Array of embedding vectors (must have consistent dimensions)
    ///   - device: Metal device for buffer allocation
    /// - Throws: `EmbedKitError.dimensionMismatch` if vectors have inconsistent dimensions
    public init(embeddings: [[Float]], device: MTLDevice) throws {
        guard let first = embeddings.first else {
            // Empty tensor
            self.batchSize = 0
            self.dimensions = 0
            guard let buffer = device.makeBuffer(length: 4, options: .storageModeShared) else {
                throw EmbedKitError.metalBufferFailed
            }
            self.buffer = buffer
            return
        }

        let dims = first.count
        for v in embeddings {
            guard v.count == dims else {
                throw EmbedKitError.dimensionMismatch(expected: dims, got: v.count)
            }
        }

        self.batchSize = embeddings.count
        self.dimensions = dims

        // Flatten to row-major contiguous storage
        let flat = embeddings.flatMap { $0 }
        let size = flat.count * MemoryLayout<Float>.stride

        guard let buffer = device.makeBuffer(
            bytes: flat,
            length: size,
            options: .storageModeShared
        ) else {
            throw EmbedKitError.metalBufferFailed
        }
        self.buffer = buffer
    }

    /// Create a tensor from a flat array with known dimensions.
    ///
    /// - Parameters:
    ///   - data: Flat row-major float array
    ///   - batchSize: Number of vectors
    ///   - dimensions: Dimensions per vector
    ///   - device: Metal device for buffer allocation
    /// - Throws: `EmbedKitError.dimensionMismatch` if data.count != batchSize × dimensions
    public init(data: [Float], batchSize: Int, dimensions: Int, device: MTLDevice) throws {
        guard data.count == batchSize * dimensions else {
            throw EmbedKitError.dimensionMismatch(
                expected: batchSize * dimensions,
                got: data.count
            )
        }

        self.batchSize = batchSize
        self.dimensions = dimensions

        let size = data.count * MemoryLayout<Float>.stride
        guard let buffer = device.makeBuffer(
            bytes: data,
            length: max(size, 1),
            options: .storageModeShared
        ) else {
            throw EmbedKitError.metalBufferFailed
        }
        self.buffer = buffer
    }

    /// Read embeddings back from the GPU buffer.
    ///
    /// - Returns: Array of embedding vectors
    public func toEmbeddings() -> [[Float]] {
        guard batchSize > 0 && dimensions > 0 else { return [] }

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        let flat = Array(UnsafeBufferPointer(start: ptr, count: totalElements))

        var result: [[Float]] = []
        result.reserveCapacity(batchSize)

        var offset = 0
        for _ in 0..<batchSize {
            let end = offset + dimensions
            result.append(Array(flat[offset..<end]))
            offset = end
        }

        return result
    }

    /// Access a single embedding by index (zero-copy read).
    ///
    /// - Parameter index: Index of the embedding to access (0..<batchSize)
    /// - Returns: The embedding vector at the specified index
    /// - Precondition: index must be in valid range
    public func embedding(at index: Int) -> [Float] {
        precondition(index >= 0 && index < batchSize, "Index out of range")

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        let offset = index * dimensions
        return Array(UnsafeBufferPointer(start: ptr + offset, count: dimensions))
    }

    // MARK: - Write Operations

    /// Write embeddings to the tensor buffer.
    ///
    /// - Parameter embeddings: Array of embedding vectors (must match tensor dimensions)
    /// - Throws: `EmbedKitError.dimensionMismatch` if dimensions don't match
    public func write(embeddings: [[Float]]) throws {
        guard embeddings.count == batchSize else {
            throw EmbedKitError.dimensionMismatch(expected: batchSize, got: embeddings.count)
        }

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        var offset = 0
        for embedding in embeddings {
            guard embedding.count == dimensions else {
                throw EmbedKitError.dimensionMismatch(expected: dimensions, got: embedding.count)
            }
            embedding.withUnsafeBufferPointer { src in
                (ptr + offset).update(from: src.baseAddress!, count: dimensions)
            }
            offset += dimensions
        }
    }

    /// Write flat data to the tensor buffer.
    ///
    /// - Parameter data: Flat row-major float array
    /// - Throws: `EmbedKitError.dimensionMismatch` if size doesn't match
    public func write(data: [Float]) throws {
        guard data.count == totalElements else {
            throw EmbedKitError.dimensionMismatch(expected: totalElements, got: data.count)
        }

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        data.withUnsafeBufferPointer { src in
            ptr.update(from: src.baseAddress!, count: totalElements)
        }
    }

    /// Write a single embedding at the specified index.
    ///
    /// - Parameters:
    ///   - embedding: The embedding vector to write
    ///   - index: Index in the batch (0..<batchSize)
    /// - Throws: `EmbedKitError.dimensionMismatch` if dimensions don't match
    public func write(embedding: [Float], at index: Int) throws {
        guard index >= 0 && index < batchSize else {
            throw EmbedKitError.invalidConfiguration("Index \(index) out of range [0, \(batchSize))")
        }
        guard embedding.count == dimensions else {
            throw EmbedKitError.dimensionMismatch(expected: dimensions, got: embedding.count)
        }

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        let offset = index * dimensions
        embedding.withUnsafeBufferPointer { src in
            (ptr + offset).update(from: src.baseAddress!, count: dimensions)
        }
    }

    /// Clear the tensor buffer to zeros.
    public func clear() {
        memset(buffer.contents(), 0, sizeInBytes)
    }
    #endif
}

// MARK: - Token Embedding Tensor (3D)

/// A 3D tensor for token-level embeddings with shape [batchSize, sequenceLength, dimensions].
///
/// `TokenEmbeddingTensor` stores sequences of token embeddings, typically the output
/// of transformer models before pooling. It supports efficient GPU operations for
/// pooling, attention, and other sequence-level computations.
///
/// **Memory Layout**: Row-major contiguous storage
/// ```
/// [b0_t0_d0, b0_t0_d1, ..., b0_t0_dD, b0_t1_d0, ..., b0_tS_dD, b1_t0_d0, ...]
/// ```
///
/// **Usage**:
/// ```swift
/// // Create from transformer output
/// let tensor = try TokenEmbeddingTensor(
///     batchSize: 4,
///     sequenceLength: 128,
///     dimensions: 384,
///     device: device
/// )
///
/// // Use in pooling kernel
/// encoder.setBuffer(tensor.buffer, offset: 0, index: 0)
/// ```
///
/// - Note: Uses `@unchecked Sendable` because MTLBuffer is thread-safe but doesn't conform to Sendable.
public struct TokenEmbeddingTensor: @unchecked Sendable {
    #if canImport(Metal)
    /// The underlying Metal buffer containing token embedding data.
    public let buffer: MTLBuffer
    #endif

    /// Number of sequences in the batch.
    public let batchSize: Int

    /// Number of tokens per sequence.
    public let sequenceLength: Int

    /// Dimensionality of each token embedding.
    public let dimensions: Int

    /// Total number of float elements (batchSize × sequenceLength × dimensions).
    public var totalElements: Int { batchSize * sequenceLength * dimensions }

    /// Size in bytes of the tensor data.
    public var sizeInBytes: Int { totalElements * MemoryLayout<Float>.stride }

    /// Shape tuple for dimension introspection.
    public var shape: (batchSize: Int, sequenceLength: Int, dimensions: Int) {
        (batchSize, sequenceLength, dimensions)
    }

    /// Number of elements per sequence (sequenceLength × dimensions).
    public var elementsPerSequence: Int { sequenceLength * dimensions }

    #if canImport(Metal)
    /// Create an empty tensor with specified dimensions.
    ///
    /// - Parameters:
    ///   - batchSize: Number of sequences
    ///   - sequenceLength: Tokens per sequence
    ///   - dimensions: Embedding dimensionality
    ///   - device: Metal device for buffer allocation
    /// - Throws: `EmbedKitError.metalError` if buffer creation fails
    public init(
        batchSize: Int,
        sequenceLength: Int,
        dimensions: Int,
        device: MTLDevice
    ) throws {
        self.batchSize = batchSize
        self.sequenceLength = sequenceLength
        self.dimensions = dimensions

        let size = batchSize * sequenceLength * dimensions * MemoryLayout<Float>.stride
        guard let buffer = device.makeBuffer(length: max(size, 1), options: .storageModeShared) else {
            throw EmbedKitError.metalBufferFailed
        }
        self.buffer = buffer
    }

    /// Create a tensor from token embeddings (batch of sequences).
    ///
    /// - Parameters:
    ///   - tokens: 3D array [batchSize][sequenceLength][dimensions]
    ///   - device: Metal device for buffer allocation
    /// - Throws: `EmbedKitError.dimensionMismatch` if dimensions are inconsistent
    public init(tokens: [[[Float]]], device: MTLDevice) throws {
        guard let firstBatch = tokens.first, let firstToken = firstBatch.first else {
            // Empty tensor
            self.batchSize = 0
            self.sequenceLength = 0
            self.dimensions = 0
            guard let buffer = device.makeBuffer(length: 4, options: .storageModeShared) else {
                throw EmbedKitError.metalBufferFailed
            }
            self.buffer = buffer
            return
        }

        let seqLen = firstBatch.count
        let dims = firstToken.count

        // Validate all dimensions
        for batch in tokens {
            guard batch.count == seqLen else {
                throw EmbedKitError.dimensionMismatch(expected: seqLen, got: batch.count)
            }
            for token in batch {
                guard token.count == dims else {
                    throw EmbedKitError.dimensionMismatch(expected: dims, got: token.count)
                }
            }
        }

        self.batchSize = tokens.count
        self.sequenceLength = seqLen
        self.dimensions = dims

        // Flatten to row-major contiguous storage
        var flat: [Float] = []
        flat.reserveCapacity(batchSize * sequenceLength * dimensions)
        for batch in tokens {
            for token in batch {
                flat.append(contentsOf: token)
            }
        }

        let size = flat.count * MemoryLayout<Float>.stride
        guard let buffer = device.makeBuffer(
            bytes: flat,
            length: size,
            options: .storageModeShared
        ) else {
            throw EmbedKitError.metalBufferFailed
        }
        self.buffer = buffer
    }

    /// Create a tensor from flat data with known dimensions.
    ///
    /// - Parameters:
    ///   - data: Flat row-major float array
    ///   - batchSize: Number of sequences
    ///   - sequenceLength: Tokens per sequence
    ///   - dimensions: Embedding dimensionality
    ///   - device: Metal device for buffer allocation
    /// - Throws: `EmbedKitError.dimensionMismatch` if data count doesn't match dimensions
    public init(
        data: [Float],
        batchSize: Int,
        sequenceLength: Int,
        dimensions: Int,
        device: MTLDevice
    ) throws {
        let expectedCount = batchSize * sequenceLength * dimensions
        guard data.count == expectedCount else {
            throw EmbedKitError.dimensionMismatch(
                expected: expectedCount,
                got: data.count
            )
        }

        self.batchSize = batchSize
        self.sequenceLength = sequenceLength
        self.dimensions = dimensions

        let size = data.count * MemoryLayout<Float>.stride
        guard let buffer = device.makeBuffer(
            bytes: data,
            length: max(size, 1),
            options: .storageModeShared
        ) else {
            throw EmbedKitError.metalBufferFailed
        }
        self.buffer = buffer
    }

    /// Read all token embeddings back from the GPU buffer.
    ///
    /// - Returns: 3D array [batchSize][sequenceLength][dimensions]
    public func toTokens() -> [[[Float]]] {
        guard batchSize > 0 && sequenceLength > 0 && dimensions > 0 else { return [] }

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        let flat = Array(UnsafeBufferPointer(start: ptr, count: totalElements))

        var result: [[[Float]]] = []
        result.reserveCapacity(batchSize)

        var offset = 0
        for _ in 0..<batchSize {
            var sequence: [[Float]] = []
            sequence.reserveCapacity(sequenceLength)
            for _ in 0..<sequenceLength {
                let end = offset + dimensions
                sequence.append(Array(flat[offset..<end]))
                offset = end
            }
            result.append(sequence)
        }

        return result
    }

    /// Access a single sequence by batch index (zero-copy read).
    ///
    /// - Parameter batchIndex: Index of the sequence to access (0..<batchSize)
    /// - Returns: 2D array [sequenceLength][dimensions] for the sequence
    public func sequence(at batchIndex: Int) -> [[Float]] {
        precondition(batchIndex >= 0 && batchIndex < batchSize, "Batch index out of range")

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        let batchOffset = batchIndex * sequenceLength * dimensions

        var result: [[Float]] = []
        result.reserveCapacity(sequenceLength)

        for t in 0..<sequenceLength {
            let tokenOffset = batchOffset + t * dimensions
            result.append(Array(UnsafeBufferPointer(start: ptr + tokenOffset, count: dimensions)))
        }

        return result
    }

    /// Access a single token embedding by batch and sequence index.
    ///
    /// - Parameters:
    ///   - batchIndex: Index of the sequence (0..<batchSize)
    ///   - tokenIndex: Index of the token (0..<sequenceLength)
    /// - Returns: The token embedding vector
    public func token(batch batchIndex: Int, token tokenIndex: Int) -> [Float] {
        precondition(batchIndex >= 0 && batchIndex < batchSize, "Batch index out of range")
        precondition(tokenIndex >= 0 && tokenIndex < sequenceLength, "Token index out of range")

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        let offset = batchIndex * elementsPerSequence + tokenIndex * dimensions
        return Array(UnsafeBufferPointer(start: ptr + offset, count: dimensions))
    }

    // MARK: - Write Operations

    /// Write token embeddings to the tensor buffer.
    ///
    /// - Parameter tokens: 3D array [batchSize][sequenceLength][dimensions]
    /// - Throws: `EmbedKitError.dimensionMismatch` if dimensions don't match
    public func write(tokens: [[[Float]]]) throws {
        guard tokens.count == batchSize else {
            throw EmbedKitError.dimensionMismatch(expected: batchSize, got: tokens.count)
        }

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        var offset = 0

        for batch in tokens {
            guard batch.count == sequenceLength else {
                throw EmbedKitError.dimensionMismatch(expected: sequenceLength, got: batch.count)
            }
            for token in batch {
                guard token.count == dimensions else {
                    throw EmbedKitError.dimensionMismatch(expected: dimensions, got: token.count)
                }
                token.withUnsafeBufferPointer { src in
                    (ptr + offset).update(from: src.baseAddress!, count: dimensions)
                }
                offset += dimensions
            }
        }
    }

    /// Write flat data to the tensor buffer.
    ///
    /// - Parameter data: Flat row-major float array
    /// - Throws: `EmbedKitError.dimensionMismatch` if size doesn't match
    public func write(data: [Float]) throws {
        guard data.count == totalElements else {
            throw EmbedKitError.dimensionMismatch(expected: totalElements, got: data.count)
        }

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        data.withUnsafeBufferPointer { src in
            ptr.update(from: src.baseAddress!, count: totalElements)
        }
    }

    /// Clear the tensor buffer to zeros.
    public func clear() {
        memset(buffer.contents(), 0, sizeInBytes)
    }
    #endif
}

// MARK: - Tensor Parameter Structures

/// Parameters for tensor-based pooling operations.
///
/// **Memory Layout**: 16 bytes (4 × Int32), 16-byte aligned
/// Matches Metal shader `TensorPoolingParams` struct.
@frozen
public struct TensorPoolingParams: Sendable {
    /// Number of sequences in the batch.
    public let batchSize: Int32

    /// Number of tokens per sequence.
    public let sequenceLength: Int32

    /// Embedding dimensionality.
    public let dimensions: Int32

    /// Pooling strategy (0=mean, 1=max, 2=cls).
    public let poolingStrategy: Int32

    public init(batchSize: Int, sequenceLength: Int, dimensions: Int, strategy: PoolingStrategy = .mean) {
        self.batchSize = Int32(batchSize)
        self.sequenceLength = Int32(sequenceLength)
        self.dimensions = Int32(dimensions)
        self.poolingStrategy = Int32(strategy.metalIndex)
    }
}

/// Parameters for tensor-based normalization operations.
///
/// **Memory Layout**: 16 bytes (4 × Int32), 16-byte aligned
@frozen
public struct TensorNormalizationParams: Sendable {
    /// Number of vectors in the batch.
    public let batchSize: Int32

    /// Dimensionality of each vector.
    public let dimensions: Int32

    /// Whether to normalize (1) or skip (0).
    public let shouldNormalize: Int32

    /// Padding for 16-byte alignment.
    private let _padding: Int32

    public init(batchSize: Int, dimensions: Int, shouldNormalize: Bool = true) {
        self.batchSize = Int32(batchSize)
        self.dimensions = Int32(dimensions)
        self.shouldNormalize = shouldNormalize ? 1 : 0
        self._padding = 0
    }
}

/// Parameters for V2 tensor normalization kernels.
///
/// **Memory Layout**: 16 bytes (4 × Int32), 16-byte aligned
/// Matches Metal shader `TensorNormParams` struct in MetalCommon.h.
@frozen
public struct TensorNormParams: Sendable {
    /// Number of vectors in the batch.
    public let batchSize: Int32

    /// Dimensionality of each vector.
    public let dimensions: Int32

    /// Padding for 16-byte alignment.
    private let _padding0: Int32

    /// Padding for 16-byte alignment.
    private let _padding1: Int32

    public init(batchSize: Int, dimensions: Int) {
        self.batchSize = Int32(batchSize)
        self.dimensions = Int32(dimensions)
        self._padding0 = 0
        self._padding1 = 0
    }
}

/// Parameters for tensor-based similarity operations.
///
/// **Memory Layout**: 16 bytes (4 × Int32), 16-byte aligned
@frozen
public struct TensorSimilarityParams: Sendable {
    /// Number of query vectors.
    public let queryBatchSize: Int32

    /// Number of key vectors.
    public let keyBatchSize: Int32

    /// Dimensionality of vectors.
    public let dimensions: Int32

    /// Similarity metric: 0=cosine, 1=dot, 2=euclidean
    public let metric: Int32

    /// Creates tensor similarity parameters.
    ///
    /// - Parameters:
    ///   - queryBatchSize: Number of query vectors
    ///   - keyBatchSize: Number of key vectors
    ///   - dimensions: Vector dimensionality
    ///   - metric: Similarity metric (0=cosine, 1=dot, 2=euclidean). Default is 0 (cosine).
    public init(queryBatchSize: Int, keyBatchSize: Int, dimensions: Int, metric: Int = 0) {
        self.queryBatchSize = Int32(queryBatchSize)
        self.keyBatchSize = Int32(keyBatchSize)
        self.dimensions = Int32(dimensions)
        self.metric = Int32(metric)
    }
}

/// Parameters for fused pooling + normalization operations.
///
/// Combines pooling and L2 normalization in single dispatch.
///
/// **Memory Layout**: 32 bytes (8 × Int32), 16-byte aligned
@frozen
public struct FusedPoolNormParams: Sendable {
    /// Number of sequences in the batch.
    public let batchSize: Int32

    /// Number of tokens per sequence.
    public let sequenceLength: Int32

    /// Embedding dimensionality.
    public let dimensions: Int32

    /// Pooling strategy (0=mean, 1=max, 2=cls).
    public let poolingStrategy: Int32

    /// Whether to apply L2 normalization (1=yes, 0=no).
    public let normalize: Int32

    /// Padding for 32-byte alignment.
    private let _padding0: Int32
    private let _padding1: Int32
    private let _padding2: Int32

    public init(
        batchSize: Int,
        sequenceLength: Int,
        dimensions: Int,
        strategy: PoolingStrategy = .mean,
        normalize: Bool = true
    ) {
        self.batchSize = Int32(batchSize)
        self.sequenceLength = Int32(sequenceLength)
        self.dimensions = Int32(dimensions)
        self.poolingStrategy = Int32(strategy.metalIndex)
        self.normalize = normalize ? 1 : 0
        self._padding0 = 0
        self._padding1 = 0
        self._padding2 = 0
    }
}

/// Parameters for complete embedding pipeline.
///
/// Fused: token embeddings → pooling → normalization → similarity
///
/// **Memory Layout**: 32 bytes (8 × Int32), 16-byte aligned
@frozen
public struct EmbeddingPipelineParams: Sendable {
    /// Number of sequences in the batch.
    public let batchSize: Int32

    /// Number of tokens per sequence.
    public let sequenceLength: Int32

    /// Embedding dimensionality.
    public let dimensions: Int32

    /// Pooling strategy (0=mean, 1=max, 2=cls).
    public let poolingStrategy: Int32

    /// Whether to apply L2 normalization.
    public let normalize: Int32

    /// Whether to compute similarity matrix.
    public let computeSimilarity: Int32

    /// Padding for 32-byte alignment.
    private let _padding0: Int32
    private let _padding1: Int32

    public init(
        batchSize: Int,
        sequenceLength: Int,
        dimensions: Int,
        strategy: PoolingStrategy = .mean,
        normalize: Bool = true,
        computeSimilarity: Bool = true
    ) {
        self.batchSize = Int32(batchSize)
        self.sequenceLength = Int32(sequenceLength)
        self.dimensions = Int32(dimensions)
        self.poolingStrategy = Int32(strategy.metalIndex)
        self.normalize = normalize ? 1 : 0
        self.computeSimilarity = computeSimilarity ? 1 : 0
        self._padding0 = 0
        self._padding1 = 0
    }
}

/// Similarity metric for tensor operations.
public enum TensorSimilarityMetric: Int32, Sendable {
    case cosine = 0
    case dotProduct = 1
    case euclidean = 2
}

// MARK: - Pooling Strategy Extension

/// Extension to provide integer index for Metal shader compatibility.
extension PoolingStrategy {
    /// Integer index for Metal shader parameter.
    var metalIndex: Int {
        switch self {
        case .mean: return 0
        case .max: return 1
        case .cls: return 2
        case .attention: return 3
        }
    }
}
