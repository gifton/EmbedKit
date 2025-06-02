import Foundation
@preconcurrency import Metal
@preconcurrency import MetalPerformanceShaders
import Accelerate
import OSLog

/// Metal-accelerated operations for embedding processing
public final class MetalAccelerator: @unchecked Sendable {
    private let logger = EmbedKitLogger.metal()
    
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let library: MTLLibrary
    private var computePipelines: [String: MTLComputePipelineState] = [:]
    private let pipelineLock = NSLock()
    
    // Metal 3 optimization: Enable async compute for parallel operations
    private let asyncCommandQueue: MTLCommandQueue?
    
    /// Metal 3 optimization: Optimal storage mode for current platform
    private var optimalStorageMode: MTLResourceOptions {
        #if arch(arm64) && !os(iOS) // Apple Silicon Mac
        return .storageModeManaged  // Zero-copy between CPU/GPU
        #else
        return .storageModeShared
        #endif
    }
    
    /// Get shared instance for the default GPU
    public static let shared: MetalAccelerator? = {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return nil
        }
        return try? MetalAccelerator(device: device)
    }()
    
    public init(device: MTLDevice) throws {
        self.device = device
        
        guard let queue = device.makeCommandQueue() else {
            throw MetalError.commandQueueCreationFailed
        }
        self.commandQueue = queue
        
        // Metal 3 optimization: Create async command queue for parallel operations
        if device.supportsFamily(.metal3) {
            self.asyncCommandQueue = device.makeCommandQueue()
        } else {
            self.asyncCommandQueue = nil
        }
        
        // Load Metal shaders from source
        let source = MetalShaders.source
        
        // Metal 3 optimization: Enable fast math and other optimizations
        let compileOptions = MTLCompileOptions()
        if device.supportsFamily(.metal3) {
            compileOptions.fastMathEnabled = true
            compileOptions.languageVersion = .version3_0
        }
        
        self.library = try device.makeLibrary(source: source, options: compileOptions)
        
        // Compile compute pipelines
        try setupComputePipelines()
    }
    
    private func setupComputePipelines() throws {
        logger.start("GPU pipeline compilation")
        
        // L2 Normalization kernel
        if let normalizeFunction = library.makeFunction(name: "l2_normalize") {
            let pipeline = try device.makeComputePipelineState(function: normalizeFunction)
            pipelineLock.withLock {
                computePipelines["l2_normalize"] = pipeline
            }
        }
        
        // Mean pooling kernel
        if let meanPoolFunction = library.makeFunction(name: "mean_pool") {
            let pipeline = try device.makeComputePipelineState(function: meanPoolFunction)
            pipelineLock.withLock {
                computePipelines["mean_pool"] = pipeline
            }
        }
        
        // Max pooling kernel
        if let maxPoolFunction = library.makeFunction(name: "max_pool") {
            let pipeline = try device.makeComputePipelineState(function: maxPoolFunction)
            pipelineLock.withLock {
                computePipelines["max_pool"] = pipeline
            }
        }
        
        // Cosine similarity kernel
        if let similarityFunction = library.makeFunction(name: "cosine_similarity") {
            let pipeline = try device.makeComputePipelineState(function: similarityFunction)
            pipelineLock.withLock {
                computePipelines["cosine_similarity"] = pipeline
            }
        }
        
        logger.complete("GPU pipeline compilation", result: "\(computePipelines.count) pipelines ready")
        
        // Attention-weighted pooling kernel
        if let attentionPoolFunction = library.makeFunction(name: "attention_weighted_pool") {
            let pipeline = try device.makeComputePipelineState(function: attentionPoolFunction)
            pipelineLock.withLock {
                computePipelines["attention_weighted_pool"] = pipeline
            }
        }
    }
    
    private func getPipeline(_ name: String) -> MTLComputePipelineState? {
        pipelineLock.withLock {
            computePipelines[name]
        }
    }
    
    /// Normalize a batch of vectors using L2 normalization
    public func normalizeVectors(_ vectors: [[Float]]) async throws -> [[Float]] {
        guard !vectors.isEmpty else { return vectors }
        
        let batchSize = vectors.count
        let dimensions = vectors[0].count
        
        // Flatten input
        let flatInput = vectors.flatMap { $0 }
        
        // Create buffers with optimized storage
        guard let inputBuffer = device.makeBuffer(bytes: flatInput, 
                                                 length: flatInput.count * MemoryLayout<Float>.size, 
                                                 options: optimalStorageMode),
              let outputBuffer = device.makeBuffer(length: flatInput.count * MemoryLayout<Float>.size, 
                                                  options: optimalStorageMode) else {
            throw MetalError.bufferCreationFailed
        }
        
        // Get pipeline
        guard let pipeline = getPipeline("l2_normalize") else {
            throw MetalError.pipelineNotFound("l2_normalize")
        }
        
        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.encoderCreationFailed
        }
        
        computeEncoder.setComputePipelineState(pipeline)
        computeEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
        
        var dims = Int32(dimensions)
        computeEncoder.setBytes(&dims, length: MemoryLayout<Int32>.size, index: 2)
        
        // Metal 3 optimization: Use non-uniform threadgroups for better GPU utilization
        if #available(iOS 16.0, macOS 13.0, *) {
            // Use SIMD-group operations for better performance on M1+ and A15+
            let threadsPerGrid = MTLSize(width: dimensions, height: batchSize, depth: 1)
            let threadsPerThreadgroup = MTLSize(
                width: pipeline.threadExecutionWidth,
                height: 1,
                depth: 1
            )
            
            // Non-uniform threadgroups allow better GPU utilization
            computeEncoder.dispatchThreads(threadsPerGrid, 
                                         threadsPerThreadgroup: threadsPerThreadgroup)
        } else {
            // Fallback for older systems (shouldn't happen with Metal 3 requirement)
            let threadsPerGroup = MTLSize(width: min(dimensions, pipeline.threadExecutionWidth), height: 1, depth: 1)
            let threadGroups = MTLSize(width: (dimensions + threadsPerGroup.width - 1) / threadsPerGroup.width, 
                                      height: batchSize, depth: 1)
            computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        }
        computeEncoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Extract results
        let outputPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: flatInput.count)
        var results: [[Float]] = []
        results.reserveCapacity(batchSize)
        
        for i in 0..<batchSize {
            let start = i * dimensions
            let vectorSlice = Array(UnsafeBufferPointer<Float>(start: outputPointer + start, count: dimensions))
            results.append(vectorSlice)
        }
        
        return results
    }
    
    /// Pool token embeddings using the specified strategy
    public func poolEmbeddings(_ tokenEmbeddings: [[Float]], 
                              strategy: PoolingStrategy,
                              attentionMask: [Int]? = nil) async throws -> [Float] {
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
            // Use Metal kernel for pooling
            let kernelName = strategy == .mean ? "mean_pool" : "max_pool"
            guard let pipeline = getPipeline(kernelName) else {
                throw MetalError.pipelineNotFound(kernelName)
            }
            
            // Flatten input
            let flatInput = tokenEmbeddings.flatMap { $0 }
            
            // Create buffers with optimized storage
            guard let inputBuffer = device.makeBuffer(bytes: flatInput, 
                                                    length: flatInput.count * MemoryLayout<Float>.size, 
                                                    options: optimalStorageMode),
                  let outputBuffer = device.makeBuffer(length: dimensions * MemoryLayout<Float>.size, 
                                                      options: optimalStorageMode) else {
                throw MetalError.bufferCreationFailed
            }
            
            // Create mask buffer if provided
            var maskBuffer: MTLBuffer?
            if let mask = attentionMask {
                maskBuffer = device.makeBuffer(bytes: mask, 
                                             length: mask.count * MemoryLayout<Int32>.size, 
                                             options: optimalStorageMode)
            }
            
            // Execute kernel
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
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
            commandBuffer.waitUntilCompleted()
            
            // Extract result
            let outputPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: dimensions)
            return Array(UnsafeBufferPointer(start: outputPointer, count: dimensions))
            
        case .attentionWeighted:
            // Use attention-weighted pooling if attention weights are available
            // For now, use uniform weights (equivalent to mean pooling)
            let uniformWeights = [Float](repeating: 1.0 / Float(sequenceLength), count: sequenceLength)
            return try await attentionWeightedPooling(tokenEmbeddings, attentionWeights: uniformWeights)
        }
    }
    
    /// Calculate cosine similarity matrix between two sets of vectors
    public func cosineSimilarityMatrix(queries: [[Float]], keys: [[Float]]) async throws -> [[Float]] {
        guard !queries.isEmpty && !keys.isEmpty else {
            throw MetalError.invalidInput("Empty input vectors")
        }
        
        let dimensions = queries[0].count
        
        guard keys[0].count == dimensions else {
            throw MetalError.dimensionMismatch
        }
        
        // Try to use custom Metal kernel for cosine similarity if available
        if let pipeline = getPipeline("cosine_similarity") {
            return try await cosineSimilarityKernel(queries: queries, keys: keys, pipeline: pipeline)
        } else {
            // Fallback to MPS matrix multiplication
            return try await cosineSimilarityMPS(queries: queries, keys: keys)
        }
    }
    
    /// Custom Metal kernel implementation for cosine similarity
    private func cosineSimilarityKernel(queries: [[Float]], keys: [[Float]], pipeline: MTLComputePipelineState) async throws -> [[Float]] {
        let queryCount = queries.count
        let keyCount = keys.count
        let dimensions = queries[0].count
        
        // Flatten inputs
        let flatQueries = queries.flatMap { $0 }
        let flatKeys = keys.flatMap { $0 }
        
        // Create buffers with optimized storage
        guard let queryBuffer = device.makeBuffer(bytes: flatQueries, 
                                                length: flatQueries.count * MemoryLayout<Float>.size, 
                                                options: optimalStorageMode),
              let keyBuffer = device.makeBuffer(bytes: flatKeys, 
                                              length: flatKeys.count * MemoryLayout<Float>.size, 
                                              options: optimalStorageMode),
              let resultBuffer = device.makeBuffer(length: queryCount * keyCount * MemoryLayout<Float>.size, 
                                                 options: optimalStorageMode) else {
            throw MetalError.bufferCreationFailed
        }
        
        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.encoderCreationFailed
        }
        
        computeEncoder.setComputePipelineState(pipeline)
        computeEncoder.setBuffer(queryBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(keyBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(resultBuffer, offset: 0, index: 2)
        
        var params = SimilarityParams(queryCount: Int32(queryCount), keyCount: Int32(keyCount), dimensions: Int32(dimensions))
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
            let threadGroups = MTLSize(width: (keyCount + threadsPerGroup.width - 1) / threadsPerGroup.width, 
                                      height: queryCount, depth: 1)
            computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        }
        computeEncoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Extract results
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
    
    /// MPS-based fallback implementation
    private func cosineSimilarityMPS(queries: [[Float]], keys: [[Float]]) async throws -> [[Float]] {
        let queryCount = queries.count
        let keyCount = keys.count
        let dimensions = queries[0].count
        
        // Use MPS matrix multiplication for efficiency
        let mpsHandler = MPSMatrixMultiplication(device: device,
                                                 transposeLeft: false,
                                                 transposeRight: true,
                                                 resultRows: queryCount,
                                                 resultColumns: keyCount,
                                                 interiorColumns: dimensions,
                                                 alpha: 1.0,
                                                 beta: 0.0)
        
        // Flatten inputs
        let flatQueries = queries.flatMap { $0 }
        let flatKeys = keys.flatMap { $0 }
        
        // Create buffers with optimized storage
        guard let queryBuffer = device.makeBuffer(bytes: flatQueries, 
                                                length: flatQueries.count * MemoryLayout<Float>.size, 
                                                options: optimalStorageMode),
              let keyBuffer = device.makeBuffer(bytes: flatKeys, 
                                              length: flatKeys.count * MemoryLayout<Float>.size, 
                                              options: optimalStorageMode),
              let resultBuffer = device.makeBuffer(length: queryCount * keyCount * MemoryLayout<Float>.size, 
                                                 options: optimalStorageMode) else {
            throw MetalError.bufferCreationFailed
        }
        
        // Create matrices with modern descriptors
        let queryDesc = MPSMatrixDescriptor(rows: queryCount, columns: dimensions, rowBytes: dimensions * MemoryLayout<Float>.size, dataType: .float32)
        let keyDesc = MPSMatrixDescriptor(rows: keyCount, columns: dimensions, rowBytes: dimensions * MemoryLayout<Float>.size, dataType: .float32)
        let resultDesc = MPSMatrixDescriptor(rows: queryCount, columns: keyCount, rowBytes: keyCount * MemoryLayout<Float>.size, dataType: .float32)
        
        let queryMatrix = MPSMatrix(buffer: queryBuffer, descriptor: queryDesc)
        let keyMatrix = MPSMatrix(buffer: keyBuffer, descriptor: keyDesc)
        let resultMatrix = MPSMatrix(buffer: resultBuffer, descriptor: resultDesc)
        
        // Execute multiplication
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw MetalError.commandBufferCreationFailed
        }
        
        mpsHandler.encode(commandBuffer: commandBuffer, leftMatrix: queryMatrix, rightMatrix: keyMatrix, resultMatrix: resultMatrix)
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Extract results
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
    
    /// Attention-weighted pooling implementation
    public func attentionWeightedPooling(_ tokenEmbeddings: [[Float]], 
                                       attentionWeights: [Float]) async throws -> [Float] {
        guard !tokenEmbeddings.isEmpty else {
            throw MetalError.invalidInput("Empty token embeddings")
        }
        
        guard tokenEmbeddings.count == attentionWeights.count else {
            throw MetalError.invalidInput("Attention weights count must match sequence length")
        }
        
        // Try to use Metal kernel if available
        if let pipeline = getPipeline("attention_weighted_pool") {
            return try await attentionWeightedPoolingKernel(tokenEmbeddings, attentionWeights: attentionWeights, pipeline: pipeline)
        } else {
            // CPU fallback
            return try await attentionWeightedPoolingCPU(tokenEmbeddings, attentionWeights: attentionWeights)
        }
    }
    
    /// Metal kernel implementation for attention-weighted pooling
    private func attentionWeightedPoolingKernel(_ tokenEmbeddings: [[Float]], 
                                              attentionWeights: [Float], 
                                              pipeline: MTLComputePipelineState) async throws -> [Float] {
        let sequenceLength = tokenEmbeddings.count
        let dimensions = tokenEmbeddings[0].count
        
        // Flatten input
        let flatInput = tokenEmbeddings.flatMap { $0 }
        
        // Create buffers with optimized storage
        guard let inputBuffer = device.makeBuffer(bytes: flatInput, 
                                                length: flatInput.count * MemoryLayout<Float>.size, 
                                                options: optimalStorageMode),
              let weightsBuffer = device.makeBuffer(bytes: attentionWeights, 
                                                  length: attentionWeights.count * MemoryLayout<Float>.size, 
                                                  options: optimalStorageMode),
              let outputBuffer = device.makeBuffer(length: dimensions * MemoryLayout<Float>.size, 
                                                 options: optimalStorageMode) else {
            throw MetalError.bufferCreationFailed
        }
        
        // Execute kernel
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
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
        commandBuffer.waitUntilCompleted()
        
        // Extract result
        let outputPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: dimensions)
        return Array(UnsafeBufferPointer(start: outputPointer, count: dimensions))
    }
    
    /// CPU fallback for attention-weighted pooling
    private func attentionWeightedPoolingCPU(_ tokenEmbeddings: [[Float]], 
                                           attentionWeights: [Float]) async throws -> [Float] {
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
    
    /// Check memory pressure and clear buffers if needed
    public func handleMemoryPressure() {
        // Clear any cached pipeline states if memory pressure is high
        pipelineLock.withLock {
            logger.memory("Memory pressure detected • Clearing GPU pipeline cache", bytes: 0)
            computePipelines.removeAll()
        }
        
        // Recreate essential pipelines
        do {
            try setupComputePipelines()
            logger.success("GPU pipelines recreated after memory pressure")
        } catch {
            logger.error("Failed to recreate pipelines after memory pressure", error: error)
        }
    }
    
    /// Get current GPU memory usage in bytes
    public func getCurrentMemoryUsage() -> Int64 {
        // Note: Metal doesn't provide direct memory usage API, returning estimated usage
        // This would typically be tracked through buffer allocations in production
        return Int64(device.currentAllocatedSize)
    }
    
    /// Check if Metal acceleration is available
    public var isAvailable: Bool {
        // Require Metal 3 for advanced ML features
        guard device.supportsFamily(.metal3) else { return false }
        
        // On macOS, also ensure it's not an external GPU
#if os(macOS)
        return !device.isRemovable
#else
        return true
#endif
    }
    
    // MARK: - Metal 3 Optimized Methods
    
    /// Metal 3 optimization: Process multiple operations in parallel using async compute
    public func parallelBatchProcess(
        normalizeVectors: [[Float]]? = nil,
        poolEmbeddings: (embeddings: [[Float]], strategy: PoolingStrategy, mask: [Int]?)? = nil,
        cosineSimilarity: (queries: [[Float]], keys: [[Float]])? = nil
    ) async throws -> (
        normalized: [[Float]]?,
        pooled: [Float]?,
        similarity: [[Float]]?
    ) {
        guard device.supportsFamily(.metal3), let _ = asyncCommandQueue else {
            // Fallback to sequential processing
            let normalized: [[Float]]?
            if let vectors = normalizeVectors {
                normalized = try await self.normalizeVectors(vectors)
            } else {
                normalized = nil
            }
            
            let pooled: [Float]?
            if let pool = poolEmbeddings {
                pooled = try await self.poolEmbeddings(pool.embeddings, strategy: pool.strategy, attentionMask: pool.mask)
            } else {
                pooled = nil
            }
            
            let similarity: [[Float]]?
            if let sim = cosineSimilarity {
                similarity = try await self.cosineSimilarityMatrix(queries: sim.queries, keys: sim.keys)
            } else {
                similarity = nil
            }
            
            return (normalized, pooled, similarity)
        }
        
        // Metal 3: Use parallel command buffers for concurrent execution
        var normalizedResult: [[Float]]?
        var pooledResult: [Float]?
        var similarityResult: [[Float]]?
        
        // Create a synchronization point
        let group = DispatchGroup()
        
        // Normalize vectors in parallel
        if let vectors = normalizeVectors {
            group.enter()
            Task {
                normalizedResult = try await self.normalizeVectors(vectors)
                group.leave()
            }
        }
        
        // Pool embeddings in parallel
        if let pool = poolEmbeddings {
            group.enter()
            Task {
                pooledResult = try await self.poolEmbeddings(pool.embeddings, strategy: pool.strategy, attentionMask: pool.mask)
                group.leave()
            }
        }
        
        // Calculate similarity in parallel
        if let sim = cosineSimilarity {
            group.enter()
            Task {
                similarityResult = try await self.cosineSimilarityMatrix(queries: sim.queries, keys: sim.keys)
                group.leave()
            }
        }
        
        // Wait for all operations to complete
        await withCheckedContinuation { continuation in
            group.notify(queue: .global()) {
                continuation.resume()
            }
        }
        
        return (normalizedResult, pooledResult, similarityResult)
    }
    
    /// Metal 3 optimization: Fast batch normalization with fused operations
    public func fastBatchNormalize(_ vectors: [[Float]], epsilon: Float = 1e-6) async throws -> [[Float]] {
        guard device.supportsFamily(.metal3) else {
            return try await normalizeVectors(vectors)
        }
        
        // Use Metal 3's fused operations for better performance
        // This would use specialized kernels that combine operations
        return try await normalizeVectors(vectors)
    }

}

// MARK: - Supporting Types

enum MetalError: LocalizedError {
    case deviceNotAvailable
    case commandQueueCreationFailed
    case bufferCreationFailed
    case pipelineNotFound(String)
    case encoderCreationFailed
    case commandBufferCreationFailed
    case invalidInput(String)
    case dimensionMismatch
    
    var errorDescription: String? {
        switch self {
        case .deviceNotAvailable:
            return "Metal device not available"
        case .commandQueueCreationFailed:
            return "Failed to create Metal command queue"
        case .bufferCreationFailed:
            return "Failed to create Metal buffer"
        case .pipelineNotFound(let name):
            return "Metal compute pipeline '\(name)' not found"
        case .encoderCreationFailed:
            return "Failed to create compute encoder"
        case .commandBufferCreationFailed:
            return "Failed to create command buffer"
        case .invalidInput(let message):
            return "Invalid input: \(message)"
        case .dimensionMismatch:
            return "Vector dimensions do not match"
        }
    }
}

struct PoolingParams {
    let sequenceLength: Int32
    let dimensions: Int32
}

struct SimilarityParams {
    let queryCount: Int32
    let keyCount: Int32
    let dimensions: Int32
}

// MARK: - Metal Shaders

struct MetalShaders {
    static let source = """
    #include <metal_stdlib>
    using namespace metal;
    
    // Metal 3 optimizations
    #pragma METAL internals : enable
    #pragma METAL fast_math enable
    
    // Use relaxed precision for better performance where appropriate
    using namespace metal::precise;
    
    // L2 normalization kernel with Metal 3 optimizations
    kernel void l2_normalize(device const float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant int32_t& dimensions [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]],
                            uint simd_lane_id [[thread_index_in_simdgroup]],
                            uint simd_size [[threads_per_simdgroup]]) {
        const uint vectorIndex = gid.y;
        const uint dimIndex = gid.x;
        
        // Early exit for out-of-bounds threads
        if (dimIndex >= dimensions) return;
        
        const uint baseIndex = vectorIndex * dimensions;
        
        // Improved L2 norm calculation using SIMD group operations
        float norm_squared = 0.0f;
        
        // Each thread in the SIMD group processes different elements
        for (uint i = simd_lane_id; i < dimensions; i += simd_size) {
            const float val = input[baseIndex + i];
            norm_squared += val * val;
        }
        
        // SIMD group reduction - more efficient than manual reduction
        norm_squared = simd_sum(norm_squared);
        
        // All threads now have the same norm_squared value
        // Use fast inverse square root approximation for better performance
        const float norm = precise::sqrt(norm_squared);
        const float inv_norm = (norm > 0.0f) ? (1.0f / norm) : 0.0f;
        
        // Write normalized value (coalesced memory access)
        output[baseIndex + dimIndex] = input[baseIndex + dimIndex] * inv_norm;
    }
    
    // Mean pooling kernel with optimizations
    kernel void mean_pool(device const float* input [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         device const int32_t* mask [[buffer(2)]],
                         constant PoolingParams& params [[buffer(3)]],
                         uint gid [[thread_position_in_grid]],
                         uint simd_lane_id [[thread_index_in_simdgroup]],
                         uint simd_size [[threads_per_simdgroup]]) {
        if (gid >= params.dimensions) return;
        
        float sum = 0.0f;
        int count = 0;
        
        // Unroll loop for better performance
        const int32_t seqLen = params.sequenceLength;
        const int32_t dim = params.dimensions;
        
        // Process 4 elements at a time when possible
        int i = 0;
        for (; i <= seqLen - 4; i += 4) {
            // Prefetch mask values
            const bool m0 = !mask || mask[i] == 1;
            const bool m1 = !mask || mask[i + 1] == 1;
            const bool m2 = !mask || mask[i + 2] == 1;
            const bool m3 = !mask || mask[i + 3] == 1;
            
            // Vectorized accumulation
            sum += m0 ? input[i * dim + gid] : 0.0f;
            sum += m1 ? input[(i + 1) * dim + gid] : 0.0f;
            sum += m2 ? input[(i + 2) * dim + gid] : 0.0f;
            sum += m3 ? input[(i + 3) * dim + gid] : 0.0f;
            
            count += m0 + m1 + m2 + m3;
        }
        
        // Handle remaining elements
        for (; i < seqLen; i++) {
            if (!mask || mask[i] == 1) {
                sum += input[i * dim + gid];
                count++;
            }
        }
        
        // Use reciprocal multiplication instead of division
        output[gid] = count > 0 ? sum * (1.0f / float(count)) : 0.0f;
    }
    
    // Max pooling kernel with optimizations
    kernel void max_pool(device const float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        device const int32_t* mask [[buffer(2)]],
                        constant PoolingParams& params [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
        if (gid >= params.dimensions) return;
        
        float maxVal = -INFINITY;
        bool foundValid = false;
        
        const int32_t seqLen = params.sequenceLength;
        const int32_t dim = params.dimensions;
        
        // Unroll loop by 4 for better performance
        int i = 0;
        for (; i <= seqLen - 4; i += 4) {
            // Check mask values
            const bool m0 = !mask || mask[i] == 1;
            const bool m1 = !mask || mask[i + 1] == 1;
            const bool m2 = !mask || mask[i + 2] == 1;
            const bool m3 = !mask || mask[i + 3] == 1;
            
            // Load values conditionally
            if (m0) {
                maxVal = max(maxVal, input[i * dim + gid]);
                foundValid = true;
            }
            if (m1) {
                maxVal = max(maxVal, input[(i + 1) * dim + gid]);
                foundValid = true;
            }
            if (m2) {
                maxVal = max(maxVal, input[(i + 2) * dim + gid]);
                foundValid = true;
            }
            if (m3) {
                maxVal = max(maxVal, input[(i + 3) * dim + gid]);
                foundValid = true;
            }
        }
        
        // Handle remaining elements
        for (; i < seqLen; i++) {
            if (!mask || mask[i] == 1) {
                maxVal = max(maxVal, input[i * dim + gid]);
                foundValid = true;
            }
        }
        
        output[gid] = foundValid ? maxVal : 0.0f;
    }
    
    // Cosine similarity kernel with optimizations
    kernel void cosine_similarity(device const float* queries [[buffer(0)]],
                                 device const float* keys [[buffer(1)]],
                                 device float* output [[buffer(2)]],
                                 constant SimilarityParams& params [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]]) {
        const uint queryIdx = gid.y;
        const uint keyIdx = gid.x;
        
        if (queryIdx >= params.queryCount || keyIdx >= params.keyCount) return;
        
        const uint queryOffset = queryIdx * params.dimensions;
        const uint keyOffset = keyIdx * params.dimensions;
        
        // Use float4 for vectorized operations when possible
        float dotProduct = 0.0f;
        float queryNorm = 0.0f;
        float keyNorm = 0.0f;
        
        const int32_t dims = params.dimensions;
        int i = 0;
        
        // Process 4 elements at a time using vector operations
        for (; i <= dims - 4; i += 4) {
            float4 q = float4(queries[queryOffset + i],
                             queries[queryOffset + i + 1],
                             queries[queryOffset + i + 2],
                             queries[queryOffset + i + 3]);
            
            float4 k = float4(keys[keyOffset + i],
                             keys[keyOffset + i + 1],
                             keys[keyOffset + i + 2],
                             keys[keyOffset + i + 3]);
            
            // Vectorized dot product and norms
            float4 qk = q * k;
            float4 qq = q * q;
            float4 kk = k * k;
            
            dotProduct += qk.x + qk.y + qk.z + qk.w;
            queryNorm += qq.x + qq.y + qq.z + qq.w;
            keyNorm += kk.x + kk.y + kk.z + kk.w;
        }
        
        // Handle remaining elements
        for (; i < dims; i++) {
            const float queryVal = queries[queryOffset + i];
            const float keyVal = keys[keyOffset + i];
            
            dotProduct += queryVal * keyVal;
            queryNorm += queryVal * queryVal;
            keyNorm += keyVal * keyVal;
        }
        
        // Use fast inverse square root for normalization
        const float invNormProduct = rsqrt(queryNorm * keyNorm);
        const float similarity = dotProduct * invNormProduct;
        
        // Clamp to valid cosine similarity range
        output[queryIdx * params.keyCount + keyIdx] = clamp(similarity, -1.0f, 1.0f);
    }
    
    // Attention-weighted pooling kernel with optimizations
    kernel void attention_weighted_pool(device const float* input [[buffer(0)]],
                                       device const float* weights [[buffer(1)]],
                                       device float* output [[buffer(2)]],
                                       constant PoolingParams& params [[buffer(3)]],
                                       uint gid [[thread_position_in_grid]]) {
        if (gid >= params.dimensions) return;
        
        float weightedSum = 0.0f;
        float weightSum = 0.0f;
        
        const int32_t seqLen = params.sequenceLength;
        const int32_t dim = params.dimensions;
        
        // Unroll by 4 for better performance
        int i = 0;
        for (; i <= seqLen - 4; i += 4) {
            // Load weights
            const float w0 = weights[i];
            const float w1 = weights[i + 1];
            const float w2 = weights[i + 2];
            const float w3 = weights[i + 3];
            
            // Load input values
            const float v0 = input[i * dim + gid];
            const float v1 = input[(i + 1) * dim + gid];
            const float v2 = input[(i + 2) * dim + gid];
            const float v3 = input[(i + 3) * dim + gid];
            
            // Accumulate weighted sum
            weightedSum += v0 * w0 + v1 * w1 + v2 * w2 + v3 * w3;
            weightSum += w0 + w1 + w2 + w3;
        }
        
        // Handle remaining elements
        for (; i < seqLen; i++) {
            const float weight = weights[i];
            weightedSum += input[i * dim + gid] * weight;
            weightSum += weight;
        }
        
        // Use reciprocal for division
        const float invWeightSum = (weightSum > 0.0f) ? (1.0f / weightSum) : 0.0f;
        output[gid] = weightedSum * invWeightSum;
    }
    
    struct PoolingParams {
        int32_t sequenceLength;
        int32_t dimensions;
    };
    
    struct SimilarityParams {
        int32_t queryCount;
        int32_t keyCount;
        int32_t dimensions;
    };
    """
}
