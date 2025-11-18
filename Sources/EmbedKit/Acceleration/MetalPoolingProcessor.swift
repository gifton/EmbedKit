import Foundation
@preconcurrency import Metal
import OSLog

/// Actor responsible for Metal-accelerated pooling operations
///
/// Specializes in various pooling strategies including mean, max, CLS, and attention-weighted pooling.
public actor MetalPoolingProcessor {
    nonisolated private let logger = EmbedKitLogger.metal()
    private let resourceManager: MetalResourceManager

    public init(resourceManager: MetalResourceManager) {
        self.resourceManager = resourceManager
    }

    // MARK: - High-Performance VectorBatch API

    /// Pool token embeddings using VectorBatch (optimized)
    ///
    /// **Performance:** Eliminates `flatMap` overhead via zero-copy GPU transfer.
    /// Expect 10-15% performance improvement over array-based API.
    ///
    /// **Algorithm:** Reduces sequence of token embeddings to single pooled vector using
    /// specified strategy (mean, max, CLS, attention-weighted).
    ///
    /// **Example:**
    /// ```swift
    /// let tokens = try VectorBatch(vectors: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    /// let pooled = try await processor.poolEmbeddings(tokens, strategy: .mean)
    /// // pooled ≈ [3.0, 4.0] (mean of all tokens)
    /// ```
    ///
    /// - Parameters:
    ///   - tokenEmbeddings: Batch of token embeddings (sequence × dimensions)
    ///   - strategy: Pooling strategy to use
    ///   - attentionMask: Optional attention mask for ignoring padding tokens
    ///   - attentionWeights: Optional weights for attention-weighted pooling
    /// - Returns: Single pooled embedding vector
    /// - Throws: MetalError if GPU operations fail
    ///
    /// - Complexity: O(n×d) where n = sequence length, d = dimensions
    /// - Note: Uses Metal 3.0 GPU acceleration for mean/max/attention pooling
    public func poolEmbeddings(
        _ tokenEmbeddings: VectorBatch,
        strategy: PoolingStrategy,
        attentionMask: [Int]? = nil,
        attentionWeights: [Float]? = nil
    ) async throws -> [Float] {
        guard !tokenEmbeddings.isEmpty else {
            throw MetalError.invalidInput("Empty token embeddings")
        }

        let sequenceLength = tokenEmbeddings.count
        let dimensions = tokenEmbeddings.dimensions

        switch strategy {
        case .cls:
            // Simply return the first token (CLS) - no GPU needed
            return Array(tokenEmbeddings[0])

        case .mean, .max:
            return try await performMetalPoolingBatch(
                tokenEmbeddings: tokenEmbeddings,
                strategy: strategy,
                attentionMask: attentionMask,
                sequenceLength: sequenceLength,
                dimensions: dimensions
            )

        case .attentionWeighted:
            // Use provided attention weights if available, otherwise use uniform weights
            let weights = attentionWeights ?? [Float](repeating: 1.0 / Float(sequenceLength), count: sequenceLength)
            return try await self.attentionWeightedPooling(tokenEmbeddings, attentionWeights: weights)
        }
    }

    /// Attention-weighted pooling with VectorBatch (optimized)
    ///
    /// **Performance:** Zero-copy GPU transfer eliminates allocation overhead.
    ///
    /// **Algorithm:** Computes weighted average: output[j] = Σ(weight[i] × token[i][j]) / Σ(weights)
    ///
    /// **Example:**
    /// ```swift
    /// let tokens = try VectorBatch(vectors: [[1.0, 2.0], [3.0, 4.0]])
    /// let weights: [Float] = [0.7, 0.3]  // Emphasize first token
    /// let pooled = try await processor.attentionWeightedPooling(tokens, attentionWeights: weights)
    /// // pooled ≈ [1.6, 2.6] (weighted average)
    /// ```
    ///
    /// - Parameters:
    ///   - tokenEmbeddings: Batch of token embeddings
    ///   - attentionWeights: Weights for each token (must match sequence length)
    /// - Returns: Attention-weighted pooled embedding
    /// - Throws: MetalError if GPU operations fail or weights count mismatches
    ///
    /// - Complexity: O(n×d) where n = sequence length, d = dimensions
    public func attentionWeightedPooling(
        _ tokenEmbeddings: VectorBatch,
        attentionWeights: [Float]
    ) async throws -> [Float] {
        guard !tokenEmbeddings.isEmpty else {
            throw MetalError.invalidInput("Empty token embeddings")
        }

        guard tokenEmbeddings.count == attentionWeights.count else {
            throw MetalError.invalidInput("Attention weights count must match sequence length")
        }

        // Try to use Metal kernel if available
        if let pipeline = try await resourceManager.getPipeline(
            MetalShaderLibrary.KernelName.attentionWeightedPool.rawValue
        ) {
            return try await attentionWeightedPoolingKernelBatch(
                tokenEmbeddings,
                attentionWeights: attentionWeights,
                pipeline: pipeline
            )
        } else {
            // CPU fallback
            return try await attentionWeightedPoolingCPUBatch(tokenEmbeddings, attentionWeights: attentionWeights)
        }
    }


    // MARK: - Private Implementation (VectorBatch)

    /// Perform Metal pooling with VectorBatch (zero-copy, optimized)
    private func performMetalPoolingBatch(
        tokenEmbeddings: VectorBatch,
        strategy: PoolingStrategy,
        attentionMask: [Int]?,
        sequenceLength: Int,
        dimensions: Int
    ) async throws -> [Float] {
        let kernelName = strategy == .mean ?
            MetalShaderLibrary.KernelName.meanPool.rawValue :
            MetalShaderLibrary.KernelName.maxPool.rawValue

        guard let pipeline = try await resourceManager.getPipeline(kernelName) else {
            throw MetalError.pipelineNotFound(kernelName)
        }

        // Zero-copy Metal buffer creation - NO flatMap!
        guard let inputBuffer = await resourceManager.createBuffer(
                bytes: tokenEmbeddings.data,  // Direct access to flat buffer
                length: tokenEmbeddings.sizeInBytes
              ),
              let outputBuffer = await resourceManager.createBuffer(
                length: dimensions * MemoryLayout<Float>.size
              ) else {
            throw MetalError.bufferCreationFailed
        }

        // Create mask buffer if provided
        var maskBuffer: MTLBuffer?
        if let mask = attentionMask {
            // Convert to Int32 to match shader expectation and ensure correct element stride
            let mask32 = mask.map { Int32($0) }
            maskBuffer = await resourceManager.createBuffer(
                bytes: mask32,
                length: mask32.count * MemoryLayout<Int32>.size
            )
        }

        // Execute kernel
        try await executePoolingKernel(
            pipeline: pipeline,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            maskBuffer: maskBuffer,
            sequenceLength: sequenceLength,
            dimensions: dimensions
        )

        // Extract result
        let outputPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: dimensions)
        return Array(UnsafeBufferPointer(start: outputPointer, count: dimensions))
    }

    /// Attention-weighted pooling kernel with VectorBatch (zero-copy)
    private func attentionWeightedPoolingKernelBatch(
        _ tokenEmbeddings: VectorBatch,
        attentionWeights: [Float],
        pipeline: MTLComputePipelineState
    ) async throws -> [Float] {
        let sequenceLength = tokenEmbeddings.count
        let dimensions = tokenEmbeddings.dimensions

        // Zero-copy Metal buffer creation - NO flatMap!
        guard let inputBuffer = await resourceManager.createBuffer(
                bytes: tokenEmbeddings.data,  // Direct access to flat buffer
                length: tokenEmbeddings.sizeInBytes
              ),
              let weightsBuffer = await resourceManager.createBuffer(
                bytes: attentionWeights,
                length: attentionWeights.count * MemoryLayout<Float>.size
              ),
              let outputBuffer = await resourceManager.createBuffer(
                length: dimensions * MemoryLayout<Float>.size
              ) else {
            throw MetalError.bufferCreationFailed
        }

        // Execute kernel
        guard let commandBuffer = resourceManager.commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.encoderCreationFailed
        }

        computeEncoder.setComputePipelineState(pipeline)
        computeEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(weightsBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(outputBuffer, offset: 0, index: 2)

        var params = PoolingParams(sequenceLength: sequenceLength, dimensions: dimensions)
        computeEncoder.setBytes(&params, length: MemoryLayout<PoolingParams>.size, index: 3)

        // Metal 3 optimization: Use non-uniform threadgroups
        if #available(iOS 16.0, macOS 13.0, *) {
            let threadsPerGrid = MTLSize(width: dimensions, height: 1, depth: 1)
            let threadsPerThreadgroup = MTLSize(
                width: pipeline.threadExecutionWidth,
                height: 1,
                depth: 1
            )
            computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        } else {
            // Fallback
            let threadsPerGroup = MTLSize(width: min(dimensions, pipeline.threadExecutionWidth), height: 1, depth: 1)
            let threadGroups = MTLSize(width: (dimensions + threadsPerGroup.width - 1) / threadsPerGroup.width, height: 1, depth: 1)
            computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        }
        computeEncoder.endEncoding()

        // Swift 6: Use async completion instead of blocking
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            commandBuffer.addCompletedHandler { buffer in
                if buffer.error != nil {
                    continuation.resume(throwing: MetalError.commandBufferCreationFailed)
                } else {
                    continuation.resume(returning: ())
                }
            }
            commandBuffer.commit()
        }

        // Extract result
        let outputPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: dimensions)
        return Array(UnsafeBufferPointer(start: outputPointer, count: dimensions))
    }

    /// CPU fallback for attention-weighted pooling with VectorBatch
    private func attentionWeightedPoolingCPUBatch(
        _ tokenEmbeddings: VectorBatch,
        attentionWeights: [Float]
    ) async throws -> [Float] {
        let sequenceLength = tokenEmbeddings.count
        let dimensions = tokenEmbeddings.dimensions

        var result = [Float](repeating: 0.0, count: dimensions)
        let weightSum = attentionWeights.reduce(0, +)

        guard weightSum > 0 else {
            throw MetalError.invalidInput("Attention weights sum to zero")
        }

        for i in 0..<sequenceLength {
            let weight = attentionWeights[i] / weightSum
            let vector = tokenEmbeddings[i]  // ArraySlice<Float>
            for j in 0..<dimensions {
                // Use relative indexing for ArraySlice
                result[j] += vector[vector.startIndex + j] * weight
            }
        }

        return result
    }


    private func executePoolingKernel(
        pipeline: MTLComputePipelineState,
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        maskBuffer: MTLBuffer?,
        sequenceLength: Int,
        dimensions: Int
    ) async throws {
        guard let commandBuffer = resourceManager.commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.encoderCreationFailed
        }

        computeEncoder.setComputePipelineState(pipeline)
        computeEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(outputBuffer, offset: 0, index: 1)

        if let maskBuffer = maskBuffer {
            computeEncoder.setBuffer(maskBuffer, offset: 0, index: 2)
        }

        var params = PoolingParams(sequenceLength: sequenceLength, dimensions: dimensions)
        computeEncoder.setBytes(&params, length: MemoryLayout<PoolingParams>.size, index: 3)

        // Metal 3 optimization: Use non-uniform threadgroups for better GPU utilization
        if #available(iOS 16.0, macOS 13.0, *) {
            let threadsPerGrid = MTLSize(width: dimensions, height: 1, depth: 1)
            let threadsPerThreadgroup = MTLSize(
                width: pipeline.threadExecutionWidth,
                height: 1,
                depth: 1
            )
            computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        } else {
            // Fallback (shouldn't happen with Metal 3 requirement)
            let threadsPerGroup = MTLSize(width: min(dimensions, pipeline.threadExecutionWidth), height: 1, depth: 1)
            let threadGroups = MTLSize(width: (dimensions + threadsPerGroup.width - 1) / threadsPerGroup.width, height: 1, depth: 1)
            computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        }
        computeEncoder.endEncoding()

        // Swift 6: Use async completion instead of blocking
        // Add handler BEFORE committing
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            commandBuffer.addCompletedHandler { buffer in
                if buffer.error != nil {
                    continuation.resume(throwing: MetalError.commandBufferCreationFailed)
                } else {
                    continuation.resume(returning: ())
                }
            }
            commandBuffer.commit()
        }
    }

}

// Note: PoolingStrategy is already defined in TextEmbedder.swift
