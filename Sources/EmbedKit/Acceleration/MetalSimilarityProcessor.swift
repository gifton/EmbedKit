import Foundation
@preconcurrency import Metal
@preconcurrency import MetalPerformanceShaders
import OSLog

/// Actor responsible for Metal-accelerated similarity calculations
///
/// Specializes in cosine similarity computations and matrix operations using both
/// custom Metal kernels and Metal Performance Shaders (MPS) fallbacks.
public actor MetalSimilarityProcessor {
    nonisolated private let logger = EmbedKitLogger.metal()
    private let resourceManager: MetalResourceManager
    
    public init(resourceManager: MetalResourceManager) {
        self.resourceManager = resourceManager
    }
    
    /// Calculate cosine similarity matrix between two sets of vectors
    ///
    /// - Parameters:
    ///   - queries: Query vectors for similarity calculation
    ///   - keys: Key vectors to compare against
    /// - Returns: Matrix of cosine similarities between queries and keys
    /// - Throws: MetalError if GPU operations fail
    public func cosineSimilarityMatrix(queries: [[Float]], keys: [[Float]]) async throws -> [[Float]] {
        guard !queries.isEmpty && !keys.isEmpty else {
            throw MetalError.invalidInput("Empty input vectors")
        }
        
        let dimensions = queries[0].count
        
        guard keys[0].count == dimensions else {
            throw MetalError.dimensionMismatch
        }
        
        // Try to use custom Metal kernel for cosine similarity if available
        if let pipeline = try await resourceManager.getPipeline(
            MetalShaderLibrary.KernelName.cosineSimilarity.rawValue
        ) {
            return try await cosineSimilarityKernel(queries: queries, keys: keys, pipeline: pipeline)
        } else {
            // Fallback to MPS matrix multiplication
            return try await cosineSimilarityMPS(queries: queries, keys: keys)
        }
    }
    
    /// Calculate similarity between a single query and multiple keys
    ///
    /// - Parameters:
    ///   - query: Single query vector
    ///   - keys: Array of key vectors to compare against
    /// - Returns: Array of similarity scores
    /// - Throws: MetalError if GPU operations fail
    public func cosineSimilarity(query: [Float], keys: [[Float]]) async throws -> [Float] {
        let similarities = try await cosineSimilarityMatrix(queries: [query], keys: keys)
        return similarities[0]
    }
    
    /// Calculate cosine similarity between two vectors
    ///
    /// - Parameters:
    ///   - vectorA: First vector for similarity calculation
    ///   - vectorB: Second vector for similarity calculation
    /// - Returns: Cosine similarity score between -1 and 1
    /// - Throws: MetalError if GPU operations fail or vectors have different dimensions
    public func cosineSimilarity(_ vectorA: [Float], _ vectorB: [Float]) async throws -> Float {
        guard vectorA.count == vectorB.count else {
            throw MetalError.dimensionMismatch
        }
        
        guard !vectorA.isEmpty else {
            throw MetalError.invalidInput("Empty input vectors")
        }
        
        // For single vector pairs, use optimized kernel if available
        if let pipeline = try await resourceManager.getPipeline(
            MetalShaderLibrary.KernelName.cosineSimilarity.rawValue
        ) {
            return try await cosineSimilarityKernelSingle(
                vectorA: vectorA,
                vectorB: vectorB,
                pipeline: pipeline
            )
        } else {
            // Fallback to matrix calculation
            let similarities = try await cosineSimilarityMatrix(queries: [vectorA], keys: [vectorB])
            return similarities[0][0]
        }
    }
    
    /// Calculate cosine similarities for multiple vector pairs in batch
    ///
    /// - Parameters:
    ///   - vectorPairs: Array of (vectorA, vectorB) tuples to compute similarities for
    /// - Returns: Array of cosine similarity scores for each pair
    /// - Throws: MetalError if GPU operations fail or vectors have mismatched dimensions
    public func cosineSimilarityBatch(_ vectorPairs: [([Float], [Float])]) async throws -> [Float] {
        guard !vectorPairs.isEmpty else {
            return []
        }
        
        let dimensions = vectorPairs[0].0.count
        
        // Validate all pairs have same dimensions
        for (vectorA, vectorB) in vectorPairs {
            guard vectorA.count == dimensions && vectorB.count == dimensions else {
                throw MetalError.dimensionMismatch
            }
        }
        
        // Try to use batch kernel for better performance
        if let pipeline = try await resourceManager.getPipeline(
            MetalShaderLibrary.KernelName.cosineSimilarityBatch.rawValue
        ) {
            return try await cosineSimilarityKernelBatch(
                vectorPairs: vectorPairs,
                dimensions: dimensions,
                pipeline: pipeline
            )
        } else {
            // Fallback to sequential processing
            var results: [Float] = []
            results.reserveCapacity(vectorPairs.count)
            
            for (vectorA, vectorB) in vectorPairs {
                let similarity = try await cosineSimilarity(vectorA, vectorB)
                results.append(similarity)
            }
            
            return results
        }
    }
    
    // MARK: - Private Implementation
    
    /// Custom Metal kernel implementation for cosine similarity
    private func cosineSimilarityKernel(
        queries: [[Float]], 
        keys: [[Float]], 
        pipeline: MTLComputePipelineState
    ) async throws -> [[Float]] {
        let queryCount = queries.count
        let keyCount = keys.count
        let dimensions = queries[0].count
        
        // Flatten inputs
        let flatQueries = queries.flatMap { $0 }
        let flatKeys = keys.flatMap { $0 }
        
        // Create buffers with optimized storage
        guard let queryBuffer = await resourceManager.createBuffer(
                bytes: flatQueries,
                length: flatQueries.count * MemoryLayout<Float>.size
              ),
              let keyBuffer = await resourceManager.createBuffer(
                bytes: flatKeys,
                length: flatKeys.count * MemoryLayout<Float>.size
              ),
              let resultBuffer = await resourceManager.createBuffer(
                length: queryCount * keyCount * MemoryLayout<Float>.size
              ) else {
            throw MetalError.bufferCreationFailed
        }
        
        // Execute kernel
        try await executeSimilarityKernel(
            pipeline: pipeline,
            queryBuffer: queryBuffer,
            keyBuffer: keyBuffer,
            resultBuffer: resultBuffer,
            queryCount: queryCount,
            keyCount: keyCount,
            dimensions: dimensions
        )
        
        // Extract and reshape results
        return extractSimilarityResults(
            from: resultBuffer,
            queryCount: queryCount,
            keyCount: keyCount
        )
    }
    
    private func executeSimilarityKernel(
        pipeline: MTLComputePipelineState,
        queryBuffer: MTLBuffer,
        keyBuffer: MTLBuffer,
        resultBuffer: MTLBuffer,
        queryCount: Int,
        keyCount: Int,
        dimensions: Int
    ) async throws {
        // Create command buffer
        guard let commandBuffer = resourceManager.commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.encoderCreationFailed
        }
        
        computeEncoder.setComputePipelineState(pipeline)
        computeEncoder.setBuffer(queryBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(keyBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(resultBuffer, offset: 0, index: 2)
        
        var params = SimilarityParams(
            queryCount: Int32(queryCount),
            keyCount: Int32(keyCount),
            dimensions: Int32(dimensions)
        )
        computeEncoder.setBytes(&params, length: MemoryLayout<SimilarityParams>.size, index: 3)
        
        // Metal 3 optimization: Use non-uniform threadgroups
        if #available(iOS 16.0, macOS 13.0, *) {
            let threadsPerGrid = MTLSize(width: keyCount, height: queryCount, depth: 1)
            let threadsPerThreadgroup = MTLSize(
                width: pipeline.threadExecutionWidth,
                height: 1,
                depth: 1
            )
            computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        } else {
            // Fallback
            let threadsPerGroup = MTLSize(width: min(keyCount, pipeline.threadExecutionWidth), height: 1, depth: 1)
            let threadGroups = MTLSize(
                width: (keyCount + threadsPerGroup.width - 1) / threadsPerGroup.width,
                height: queryCount,
                depth: 1
            )
            computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        }
        computeEncoder.endEncoding()
        
        // Swift 6: Use async completion instead of blocking
        // Add completion handler BEFORE committing
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
    
    private func extractSimilarityResults(
        from resultBuffer: MTLBuffer,
        queryCount: Int,
        keyCount: Int
    ) -> [[Float]] {
        let outputPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: queryCount * keyCount)
        var results: [[Float]] = []
        results.reserveCapacity(queryCount)
        
        for i in 0..<queryCount {
            let start = i * keyCount
            let row = Array(UnsafeBufferPointer(start: outputPointer + start, count: keyCount))
            results.append(row)
        }
        
        return results
    }
    
    /// Optimized single vector cosine similarity calculation
    private func cosineSimilarityKernelSingle(
        vectorA: [Float],
        vectorB: [Float],
        pipeline: MTLComputePipelineState
    ) async throws -> Float {
        let dimensions = vectorA.count
        
        // Create buffers
        guard let vectorABuffer = await resourceManager.createBuffer(
                bytes: vectorA,
                length: vectorA.count * MemoryLayout<Float>.size
              ),
              let vectorBBuffer = await resourceManager.createBuffer(
                bytes: vectorB,
                length: vectorB.count * MemoryLayout<Float>.size
              ),
              let resultBuffer = await resourceManager.createBuffer(
                length: MemoryLayout<Float>.size
              ) else {
            throw MetalError.bufferCreationFailed
        }
        
        // Create command buffer
        guard let commandBuffer = resourceManager.commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.encoderCreationFailed
        }
        
        computeEncoder.setComputePipelineState(pipeline)
        computeEncoder.setBuffer(vectorABuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(vectorBBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(resultBuffer, offset: 0, index: 2)
        
        var params = SimilarityParams(
            queryCount: 1,
            keyCount: 1,
            dimensions: Int32(dimensions)
        )
        computeEncoder.setBytes(&params, length: MemoryLayout<SimilarityParams>.size, index: 3)
        
        // Dispatch single thread since we're computing one similarity value
        let threadsPerGrid = MTLSize(width: 1, height: 1, depth: 1)
        let threadsPerThreadgroup = MTLSize(width: 1, height: 1, depth: 1)
        computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        computeEncoder.endEncoding()
        
        // Wait for completion - add handler BEFORE committing
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
        let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: 1)
        return resultPointer[0]
    }
    
    /// Batch cosine similarity kernel implementation
    private func cosineSimilarityKernelBatch(
        vectorPairs: [([Float], [Float])],
        dimensions: Int,
        pipeline: MTLComputePipelineState
    ) async throws -> [Float] {
        let pairCount = vectorPairs.count
        
        // Flatten input vectors
        var flatVectorsA: [Float] = []
        var flatVectorsB: [Float] = []
        flatVectorsA.reserveCapacity(pairCount * dimensions)
        flatVectorsB.reserveCapacity(pairCount * dimensions)
        
        for (vectorA, vectorB) in vectorPairs {
            flatVectorsA.append(contentsOf: vectorA)
            flatVectorsB.append(contentsOf: vectorB)
        }
        
        // Create buffers
        guard let vectorsABuffer = await resourceManager.createBuffer(
                bytes: flatVectorsA,
                length: flatVectorsA.count * MemoryLayout<Float>.size
              ),
              let vectorsBBuffer = await resourceManager.createBuffer(
                bytes: flatVectorsB,
                length: flatVectorsB.count * MemoryLayout<Float>.size
              ),
              let resultBuffer = await resourceManager.createBuffer(
                length: pairCount * MemoryLayout<Float>.size
              ) else {
            throw MetalError.bufferCreationFailed
        }
        
        // Create command buffer
        guard let commandBuffer = resourceManager.commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.encoderCreationFailed
        }
        
        computeEncoder.setComputePipelineState(pipeline)
        computeEncoder.setBuffer(vectorsABuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(vectorsBBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(resultBuffer, offset: 0, index: 2)
        
        var params = BatchSimilarityParams(
            pairCount: Int32(pairCount),
            dimensions: Int32(dimensions)
        )
        computeEncoder.setBytes(&params, length: MemoryLayout<BatchSimilarityParams>.size, index: 3)
        
        // Dispatch threads for batch processing
        let threadsPerGrid = MTLSize(width: pairCount, height: 1, depth: 1)
        let threadsPerThreadgroup = MTLSize(
            width: min(pairCount, pipeline.threadExecutionWidth),
            height: 1,
            depth: 1
        )
        computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        computeEncoder.endEncoding()
        
        // Wait for completion - add handler BEFORE committing
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
        
        // Extract results
        let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: pairCount)
        return Array(UnsafeBufferPointer(start: resultPointer, count: pairCount))
    }
    
    /// MPS-based fallback implementation
    private func cosineSimilarityMPS(queries: [[Float]], keys: [[Float]]) async throws -> [[Float]] {
        let queryCount = queries.count
        let keyCount = keys.count
        let dimensions = queries[0].count
        
        // Use MPS matrix multiplication for efficiency
        let mpsHandler = MPSMatrixMultiplication(
            device: resourceManager.device,
            transposeLeft: false,
            transposeRight: true,
            resultRows: queryCount,
            resultColumns: keyCount,
            interiorColumns: dimensions,
            alpha: 1.0,
            beta: 0.0
        )
        
        // Flatten inputs
        let flatQueries = queries.flatMap { $0 }
        let flatKeys = keys.flatMap { $0 }
        
        // Create buffers with optimized storage
        guard let queryBuffer = await resourceManager.createBuffer(
                bytes: flatQueries,
                length: flatQueries.count * MemoryLayout<Float>.size
              ),
              let keyBuffer = await resourceManager.createBuffer(
                bytes: flatKeys,
                length: flatKeys.count * MemoryLayout<Float>.size
              ),
              let resultBuffer = await resourceManager.createBuffer(
                length: queryCount * keyCount * MemoryLayout<Float>.size
              ) else {
            throw MetalError.bufferCreationFailed
        }
        
        // Create matrices with modern descriptors
        let queryDesc = MPSMatrixDescriptor(
            rows: queryCount,
            columns: dimensions,
            rowBytes: dimensions * MemoryLayout<Float>.size,
            dataType: .float32
        )
        let keyDesc = MPSMatrixDescriptor(
            rows: keyCount,
            columns: dimensions,
            rowBytes: dimensions * MemoryLayout<Float>.size,
            dataType: .float32
        )
        let resultDesc = MPSMatrixDescriptor(
            rows: queryCount,
            columns: keyCount,
            rowBytes: keyCount * MemoryLayout<Float>.size,
            dataType: .float32
        )
        
        let queryMatrix = MPSMatrix(buffer: queryBuffer, descriptor: queryDesc)
        let keyMatrix = MPSMatrix(buffer: keyBuffer, descriptor: keyDesc)
        let resultMatrix = MPSMatrix(buffer: resultBuffer, descriptor: resultDesc)
        
        // Execute multiplication
        guard let commandBuffer = resourceManager.commandQueue.makeCommandBuffer() else {
            throw MetalError.commandBufferCreationFailed
        }
        
        mpsHandler.encode(
            commandBuffer: commandBuffer,
            leftMatrix: queryMatrix,
            rightMatrix: keyMatrix,
            resultMatrix: resultMatrix
        )
        
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
        
        // Extract results
        return extractSimilarityResults(
            from: resultBuffer,
            queryCount: queryCount,
            keyCount: keyCount
        )
    }
}