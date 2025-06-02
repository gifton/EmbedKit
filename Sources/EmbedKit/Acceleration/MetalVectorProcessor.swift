import Foundation
@preconcurrency import Metal
import OSLog

/// Actor responsible for Metal-accelerated vector operations
///
/// Specializes in vector normalization and related mathematical operations on GPU.
public actor MetalVectorProcessor {
    nonisolated private let logger = EmbedKitLogger.metal()
    private let resourceManager: MetalResourceManager
    
    public init(resourceManager: MetalResourceManager) {
        self.resourceManager = resourceManager
    }
    
    /// Normalize a batch of vectors using L2 normalization
    ///
    /// - Parameter vectors: Array of vectors to normalize
    /// - Returns: Array of L2-normalized vectors
    /// - Throws: MetalError if GPU operations fail
    public func normalizeVectors(_ vectors: [[Float]]) async throws -> [[Float]] {
        guard !vectors.isEmpty else { return vectors }
        
        let batchSize = vectors.count
        let dimensions = vectors[0].count
        
        // Flatten input for efficient GPU processing
        let flatInput = vectors.flatMap { $0 }
        
        // Create optimized Metal buffers
        guard let inputBuffer = await resourceManager.createBuffer(
                bytes: flatInput, 
                length: flatInput.count * MemoryLayout<Float>.size
              ),
              let outputBuffer = await resourceManager.createBuffer(
                length: flatInput.count * MemoryLayout<Float>.size
              ) else {
            throw MetalError.bufferCreationFailed
        }
        
        // Get the L2 normalization pipeline
        guard let pipeline = try await resourceManager.getPipeline(
            MetalShaderLibrary.KernelName.l2Normalize.rawValue
        ) else {
            throw MetalError.pipelineNotFound(MetalShaderLibrary.KernelName.l2Normalize.rawValue)
        }
        
        // Execute GPU computation
        try await executeNormalizationKernel(
            pipeline: pipeline,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            dimensions: dimensions,
            batchSize: batchSize
        )
        
        // Extract and reshape results
        return extractNormalizationResults(
            from: outputBuffer,
            batchSize: batchSize,
            dimensions: dimensions
        )
    }
    
    /// Metal 3 optimization: Fast batch normalization with epsilon parameter
    ///
    /// - Parameters:
    ///   - vectors: Vectors to normalize
    ///   - epsilon: Small value to prevent division by zero
    /// - Returns: Normalized vectors
    public func fastBatchNormalize(_ vectors: [[Float]], epsilon: Float = 1e-6) async throws -> [[Float]] {
        // For now, delegate to standard normalization
        // Future enhancement: implement epsilon-aware normalization kernel
        return try await normalizeVectors(vectors)
    }
    
    // MARK: - Private Implementation
    
    private func executeNormalizationKernel(
        pipeline: MTLComputePipelineState,
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        dimensions: Int,
        batchSize: Int
    ) async throws {
        // Create command buffer
        guard let commandBuffer = resourceManager.commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.encoderCreationFailed
        }
        
        // Setup compute pipeline
        computeEncoder.setComputePipelineState(pipeline)
        computeEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
        
        // Set dimensions parameter
        var dims = Int32(dimensions)
        computeEncoder.setBytes(&dims, length: MemoryLayout<Int32>.size, index: 2)
        
        // Metal 3 optimization: Use non-uniform threadgroups for better GPU utilization
        if #available(iOS 16.0, macOS 13.0, *) {
            let threadsPerGrid = MTLSize(width: dimensions, height: batchSize, depth: 1)
            let threadsPerThreadgroup = MTLSize(
                width: pipeline.threadExecutionWidth,
                height: 1,
                depth: 1
            )
            computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        } else {
            // Fallback for older systems
            let threadsPerGroup = MTLSize(width: min(dimensions, pipeline.threadExecutionWidth), height: 1, depth: 1)
            let threadGroups = MTLSize(
                width: (dimensions + threadsPerGroup.width - 1) / threadsPerGroup.width,
                height: batchSize,
                depth: 1
            )
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
    
    private func extractNormalizationResults(
        from outputBuffer: MTLBuffer,
        batchSize: Int,
        dimensions: Int
    ) -> [[Float]] {
        let outputPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: batchSize * dimensions)
        var results: [[Float]] = []
        results.reserveCapacity(batchSize)
        
        for i in 0..<batchSize {
            let start = i * dimensions
            let vectorSlice = Array(UnsafeBufferPointer<Float>(start: outputPointer + start, count: dimensions))
            results.append(vectorSlice)
        }
        
        return results
    }
}