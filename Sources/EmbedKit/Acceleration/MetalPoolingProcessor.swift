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
    
    /// Pool token embeddings using the specified strategy
    ///
    /// - Parameters:
    ///   - tokenEmbeddings: Array of token embeddings to pool
    ///   - strategy: Pooling strategy to use
    ///   - attentionMask: Optional attention mask for ignoring padding tokens
    /// - Returns: Single pooled embedding vector
    /// - Throws: MetalError if GPU operations fail
    public func poolEmbeddings(
        _ tokenEmbeddings: [[Float]],
        strategy: PoolingStrategy,
        attentionMask: [Int]? = nil
    ) async throws -> [Float] {
        guard !tokenEmbeddings.isEmpty else {
            throw MetalError.invalidInput("Empty token embeddings")
        }
        
        let sequenceLength = tokenEmbeddings.count
        let dimensions = tokenEmbeddings[0].count
        
        switch strategy {
        case .cls:
            // Simply return the first token (CLS)
            return tokenEmbeddings[0]
            
        case .mean, .max:
            return try await performMetalPooling(
                tokenEmbeddings: tokenEmbeddings,
                strategy: strategy,
                attentionMask: attentionMask,
                sequenceLength: sequenceLength,
                dimensions: dimensions
            )
            
        case .attentionWeighted:
            // Use attention-weighted pooling if attention weights are available
            // For now, use uniform weights (equivalent to mean pooling)
            let uniformWeights = [Float](repeating: 1.0 / Float(sequenceLength), count: sequenceLength)
            return try await attentionWeightedPooling(tokenEmbeddings, attentionWeights: uniformWeights)
        }
    }
    
    /// Attention-weighted pooling implementation
    ///
    /// - Parameters:
    ///   - tokenEmbeddings: Array of token embeddings
    ///   - attentionWeights: Weights for each token
    /// - Returns: Attention-weighted pooled embedding
    /// - Throws: MetalError if GPU operations fail
    public func attentionWeightedPooling(
        _ tokenEmbeddings: [[Float]],
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
            return try await attentionWeightedPoolingKernel(
                tokenEmbeddings,
                attentionWeights: attentionWeights,
                pipeline: pipeline
            )
        } else {
            // CPU fallback
            return try await attentionWeightedPoolingCPU(tokenEmbeddings, attentionWeights: attentionWeights)
        }
    }
    
    // MARK: - Private Implementation
    
    private func performMetalPooling(
        tokenEmbeddings: [[Float]],
        strategy: PoolingStrategy,
        attentionMask: [Int]?,
        sequenceLength: Int,
        dimensions: Int
    ) async throws -> [Float] {
        // Use Metal kernel for pooling
        let kernelName = strategy == .mean ? 
            MetalShaderLibrary.KernelName.meanPool.rawValue : 
            MetalShaderLibrary.KernelName.maxPool.rawValue
            
        guard let pipeline = try await resourceManager.getPipeline(kernelName) else {
            throw MetalError.pipelineNotFound(kernelName)
        }
        
        // Flatten input
        let flatInput = tokenEmbeddings.flatMap { $0 }
        
        // Create buffers with optimized storage
        guard let inputBuffer = await resourceManager.createBuffer(
                bytes: flatInput,
                length: flatInput.count * MemoryLayout<Float>.size
              ),
              let outputBuffer = await resourceManager.createBuffer(
                length: dimensions * MemoryLayout<Float>.size
              ) else {
            throw MetalError.bufferCreationFailed
        }
        
        // Create mask buffer if provided
        var maskBuffer: MTLBuffer?
        if let mask = attentionMask {
            maskBuffer = await resourceManager.createBuffer(
                bytes: mask,
                length: mask.count * MemoryLayout<Int32>.size
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
        
        var params = PoolingParams(sequenceLength: Int32(sequenceLength), dimensions: Int32(dimensions))
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
        
        commandBuffer.commit()
        
        // Swift 6: Use async completion instead of blocking
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            commandBuffer.addCompletedHandler { buffer in
                if buffer.error != nil {
                    continuation.resume(throwing: MetalError.commandBufferCreationFailed)
                } else {
                    continuation.resume(returning: ())
                }
            }
        }
    }
    
    private func attentionWeightedPoolingKernel(
        _ tokenEmbeddings: [[Float]],
        attentionWeights: [Float],
        pipeline: MTLComputePipelineState
    ) async throws -> [Float] {
        let sequenceLength = tokenEmbeddings.count
        let dimensions = tokenEmbeddings[0].count
        
        // Flatten input
        let flatInput = tokenEmbeddings.flatMap { $0 }
        
        // Create buffers with optimized storage
        guard let inputBuffer = await resourceManager.createBuffer(
                bytes: flatInput,
                length: flatInput.count * MemoryLayout<Float>.size
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
        
        var params = PoolingParams(sequenceLength: Int32(sequenceLength), dimensions: Int32(dimensions))
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
        
        commandBuffer.commit()
        
        // Swift 6: Use async completion instead of blocking
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            commandBuffer.addCompletedHandler { buffer in
                if buffer.error != nil {
                    continuation.resume(throwing: MetalError.commandBufferCreationFailed)
                } else {
                    continuation.resume(returning: ())
                }
            }
        }
        
        // Extract result
        let outputPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: dimensions)
        return Array(UnsafeBufferPointer(start: outputPointer, count: dimensions))
    }
    
    /// CPU fallback for attention-weighted pooling
    private func attentionWeightedPoolingCPU(
        _ tokenEmbeddings: [[Float]],
        attentionWeights: [Float]
    ) async throws -> [Float] {
        let sequenceLength = tokenEmbeddings.count
        let dimensions = tokenEmbeddings[0].count
        
        var result = [Float](repeating: 0.0, count: dimensions)
        let weightSum = attentionWeights.reduce(0, +)
        
        guard weightSum > 0 else {
            throw MetalError.invalidInput("Attention weights sum to zero")
        }
        
        for i in 0..<sequenceLength {
            let weight = attentionWeights[i] / weightSum
            for j in 0..<dimensions {
                result[j] += tokenEmbeddings[i][j] * weight
            }
        }
        
        return result
    }
}

// Note: PoolingStrategy is already defined in TextEmbedder.swift
// We'll use that definition to avoid conflicts