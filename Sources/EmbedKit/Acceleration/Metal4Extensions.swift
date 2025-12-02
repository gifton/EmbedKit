// EmbedKit - Metal 4 Extensions
//
// Convenience extensions for Metal 4 APIs.
// Provides device-level factory methods for tensor creation and configuration.

import Foundation

#if canImport(Metal)
@preconcurrency import Metal

// MARK: - MTLDevice Extensions

extension MTLDevice {

    // MARK: - Tensor Creation

    /// Create an embedding tensor for a batch of embeddings.
    ///
    /// - Parameters:
    ///   - batchSize: Number of embeddings
    ///   - dimensions: Dimensions per embedding
    /// - Returns: Empty EmbeddingTensor ready for data
    /// - Throws: `EmbedKitError.metalTensorFailed` if creation fails
    public func makeEmbeddingTensor(
        batchSize: Int,
        dimensions: Int
    ) throws -> EmbeddingTensor {
        try EmbeddingTensor(
            batchSize: batchSize,
            dimensions: dimensions,
            device: self
        )
    }

    /// Create an embedding tensor from existing embedding vectors.
    ///
    /// - Parameter embeddings: Array of embedding vectors
    /// - Returns: EmbeddingTensor populated with the data
    /// - Throws: `EmbedKitError` if creation fails
    public func makeEmbeddingTensor(
        embeddings: [[Float]]
    ) throws -> EmbeddingTensor {
        try EmbeddingTensor(embeddings: embeddings, device: self)
    }

    /// Create a token embedding tensor.
    ///
    /// - Parameters:
    ///   - batchSize: Number of sequences
    ///   - sequenceLength: Tokens per sequence
    ///   - dimensions: Dimensions per token
    /// - Returns: Empty TokenEmbeddingTensor ready for data
    /// - Throws: `EmbedKitError.metalTensorFailed` if creation fails
    public func makeTokenEmbeddingTensor(
        batchSize: Int,
        sequenceLength: Int,
        dimensions: Int
    ) throws -> TokenEmbeddingTensor {
        try TokenEmbeddingTensor(
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            dimensions: dimensions,
            device: self
        )
    }

    // MARK: - Tensor Parameter Buffers

    /// Create a buffer for pooling parameters.
    ///
    /// - Parameter params: Pooling parameters
    /// - Returns: Buffer containing the parameters
    public func makePoolingParamsBuffer(_ params: TensorPoolingParams) -> MTLBuffer? {
        var p = params
        return makeBuffer(
            bytes: &p,
            length: MemoryLayout<TensorPoolingParams>.stride,
            options: .storageModeShared
        )
    }

    /// Create a buffer for normalization parameters.
    ///
    /// - Parameter params: Normalization parameters
    /// - Returns: Buffer containing the parameters
    public func makeNormalizationParamsBuffer(_ params: TensorNormalizationParams) -> MTLBuffer? {
        var p = params
        return makeBuffer(
            bytes: &p,
            length: MemoryLayout<TensorNormalizationParams>.stride,
            options: .storageModeShared
        )
    }

    /// Create a buffer for similarity parameters.
    ///
    /// - Parameter params: Similarity parameters
    /// - Returns: Buffer containing the parameters
    public func makeSimilarityParamsBuffer(_ params: TensorSimilarityParams) -> MTLBuffer? {
        var p = params
        return makeBuffer(
            bytes: &p,
            length: MemoryLayout<TensorSimilarityParams>.stride,
            options: .storageModeShared
        )
    }

    /// Create a buffer for fused pool+norm parameters.
    ///
    /// - Parameter params: Fused operation parameters
    /// - Returns: Buffer containing the parameters
    public func makeFusedPoolNormParamsBuffer(_ params: FusedPoolNormParams) -> MTLBuffer? {
        var p = params
        return makeBuffer(
            bytes: &p,
            length: MemoryLayout<FusedPoolNormParams>.stride,
            options: .storageModeShared
        )
    }

    // MARK: - Similarity Matrix

    /// Create a similarity matrix buffer.
    ///
    /// - Parameters:
    ///   - queryBatchSize: Number of query embeddings
    ///   - keyBatchSize: Number of key embeddings
    /// - Returns: Buffer for storing similarity scores [Q x K]
    public func makeSimilarityMatrixBuffer(
        queryBatchSize: Int,
        keyBatchSize: Int
    ) -> MTLBuffer? {
        let size = queryBatchSize * keyBatchSize * MemoryLayout<Float>.stride
        return makeBuffer(length: size, options: .storageModeShared)
    }

    // MARK: - Attention Mask

    /// Create an attention mask buffer.
    ///
    /// - Parameters:
    ///   - batchSize: Number of sequences
    ///   - sequenceLength: Tokens per sequence
    ///   - defaultValue: Default mask value (1 = attend, 0 = ignore)
    /// - Returns: Buffer initialized with the default value
    public func makeAttentionMaskBuffer(
        batchSize: Int,
        sequenceLength: Int,
        defaultValue: Int32 = 1
    ) -> MTLBuffer? {
        let count = batchSize * sequenceLength
        let size = count * MemoryLayout<Int32>.stride
        guard let buffer = makeBuffer(length: size, options: .storageModeShared) else {
            return nil
        }

        let ptr = buffer.contents().bindMemory(to: Int32.self, capacity: count)
        for i in 0..<count {
            ptr[i] = defaultValue
        }

        return buffer
    }
}

// MARK: - MTLCommandQueue Extensions

extension MTLCommandQueue {

    /// Execute a simple compute operation with a single buffer.
    ///
    /// Convenience method for common single-dispatch patterns.
    ///
    /// - Parameters:
    ///   - pipeline: Compute pipeline state
    ///   - buffer: Input/output buffer
    ///   - threadgroups: Grid dimensions
    ///   - threadsPerGroup: Threads per threadgroup
    /// - Returns: True if encoding succeeded
    @discardableResult
    public func executeCompute(
        pipeline: MTLComputePipelineState,
        buffer: MTLBuffer,
        threadgroups: MTLSize,
        threadsPerGroup: MTLSize
    ) async -> Bool {
        guard let cmd = makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else {
            return false
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        cmd.commit()
        await cmd.completed()
        return true
    }
}

// MARK: - Compute Dispatch Helpers

/// Helper for calculating optimal threadgroup sizes.
public struct ThreadgroupCalculator {

    /// Calculate optimal threadgroup size for a given pipeline and dimension.
    ///
    /// - Parameters:
    ///   - pipeline: Compute pipeline state
    ///   - totalThreads: Total threads needed
    /// - Returns: Tuple of (threadgroups, threadsPerThreadgroup)
    public static func calculate(
        pipeline: MTLComputePipelineState,
        totalThreads: Int
    ) -> (threadgroups: MTLSize, threadsPerGroup: MTLSize) {
        let maxThreadsPerGroup = pipeline.maxTotalThreadsPerThreadgroup
        let threadExecutionWidth = pipeline.threadExecutionWidth

        // Use thread execution width for best performance
        let threadsPerGroup = min(maxThreadsPerGroup, max(threadExecutionWidth, 64))
        let threadgroupCount = (totalThreads + threadsPerGroup - 1) / threadsPerGroup

        return (
            MTLSize(width: threadgroupCount, height: 1, depth: 1),
            MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        )
    }

    /// Calculate 2D threadgroup size for batch operations.
    ///
    /// - Parameters:
    ///   - pipeline: Compute pipeline state
    ///   - batchSize: Number of items
    ///   - dimensions: Dimension size
    /// - Returns: Tuple of (threadgroups, threadsPerThreadgroup)
    public static func calculate2D(
        pipeline: MTLComputePipelineState,
        batchSize: Int,
        dimensions: Int
    ) -> (threadgroups: MTLSize, threadsPerGroup: MTLSize) {
        let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
        let width = min(32, dimensions)
        let height = min(maxThreads / width, batchSize)

        let groupsX = (dimensions + width - 1) / width
        let groupsY = (batchSize + height - 1) / height

        return (
            MTLSize(width: groupsX, height: groupsY, depth: 1),
            MTLSize(width: width, height: height, depth: 1)
        )
    }
}

#endif
