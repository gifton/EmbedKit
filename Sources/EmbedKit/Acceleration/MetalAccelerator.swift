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
        
        // Load Metal shaders from source
        let source = MetalShaders.source
        self.library = try device.makeLibrary(source: source, options: nil)
        
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
        
        // Create buffers
        guard let inputBuffer = device.makeBuffer(bytes: flatInput, length: flatInput.count * MemoryLayout<Float>.size, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: flatInput.count * MemoryLayout<Float>.size, options: .storageModeShared) else {
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
        
        // Calculate thread groups
        let threadsPerGroup = MTLSize(width: min(dimensions, pipeline.threadExecutionWidth), height: 1, depth: 1)
        let threadGroups = MTLSize(width: (dimensions + threadsPerGroup.width - 1) / threadsPerGroup.width, 
                                  height: batchSize, depth: 1)
        
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        computeEncoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Extract results
        let outputPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: flatInput.count)
        var results: [[Float]] = []
        results.reserveCapacity(batchSize)
        
        for i in 0..<batchSize {
            let start = i * dimensions
            let vectorSlice = Array(UnsafeBufferPointer(start: outputPointer + start, count: dimensions))
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
            
            // Create buffers
            guard let inputBuffer = device.makeBuffer(bytes: flatInput, length: flatInput.count * MemoryLayout<Float>.size, options: .storageModeShared),
                  let outputBuffer = device.makeBuffer(length: dimensions * MemoryLayout<Float>.size, options: .storageModeShared) else {
                throw MetalError.bufferCreationFailed
            }
            
            // Create mask buffer if provided
            var maskBuffer: MTLBuffer?
            if let mask = attentionMask {
                maskBuffer = device.makeBuffer(bytes: mask, length: mask.count * MemoryLayout<Int32>.size, options: .storageModeShared)
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
            
            let threadsPerGroup = MTLSize(width: min(dimensions, pipeline.threadExecutionWidth), height: 1, depth: 1)
            let threadGroups = MTLSize(width: (dimensions + threadsPerGroup.width - 1) / threadsPerGroup.width, height: 1, depth: 1)
            
            computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
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
        
        // Create buffers
        guard let queryBuffer = device.makeBuffer(bytes: flatQueries, length: flatQueries.count * MemoryLayout<Float>.size, options: .storageModeShared),
              let keyBuffer = device.makeBuffer(bytes: flatKeys, length: flatKeys.count * MemoryLayout<Float>.size, options: .storageModeShared),
              let resultBuffer = device.makeBuffer(length: queryCount * keyCount * MemoryLayout<Float>.size, options: .storageModeShared) else {
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
        
        // Calculate thread groups
        let threadsPerGroup = MTLSize(width: min(keyCount, pipeline.threadExecutionWidth), height: 1, depth: 1)
        let threadGroups = MTLSize(width: (keyCount + threadsPerGroup.width - 1) / threadsPerGroup.width, 
                                  height: queryCount, depth: 1)
        
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
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
        
        // Create buffers
        guard let queryBuffer = device.makeBuffer(bytes: flatQueries, length: flatQueries.count * MemoryLayout<Float>.size, options: .storageModeShared),
              let keyBuffer = device.makeBuffer(bytes: flatKeys, length: flatKeys.count * MemoryLayout<Float>.size, options: .storageModeShared),
              let resultBuffer = device.makeBuffer(length: queryCount * keyCount * MemoryLayout<Float>.size, options: .storageModeShared) else {
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
        
        // Create buffers
        guard let inputBuffer = device.makeBuffer(bytes: flatInput, length: flatInput.count * MemoryLayout<Float>.size, options: .storageModeShared),
              let weightsBuffer = device.makeBuffer(bytes: attentionWeights, length: attentionWeights.count * MemoryLayout<Float>.size, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: dimensions * MemoryLayout<Float>.size, options: .storageModeShared) else {
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
        
        let threadsPerGroup = MTLSize(width: min(dimensions, pipeline.threadExecutionWidth), height: 1, depth: 1)
        let threadGroups = MTLSize(width: (dimensions + threadsPerGroup.width - 1) / threadsPerGroup.width, height: 1, depth: 1)
        
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
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
    
    // L2 normalization kernel
    kernel void l2_normalize(device const float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant int32_t& dimensions [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
        uint vectorIndex = gid.y;
        uint dimIndex = gid.x;
        
        if (dimIndex >= dimensions) return;
        
        uint baseIndex = vectorIndex * dimensions;
        
        // Calculate L2 norm
        float norm = 0.0;
        for (int i = 0; i < dimensions; i++) {
            float val = input[baseIndex + i];
            norm += val * val;
        }
        norm = sqrt(norm);
        
        // Normalize
        if (norm > 0) {
            output[baseIndex + dimIndex] = input[baseIndex + dimIndex] / norm;
        } else {
            output[baseIndex + dimIndex] = 0.0;
        }
    }
    
    // Mean pooling kernel
    kernel void mean_pool(device const float* input [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         device const int32_t* mask [[buffer(2)]],
                         constant PoolingParams& params [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
        if (gid >= params.dimensions) return;
        
        float sum = 0.0;
        int count = 0;
        
        for (int i = 0; i < params.sequenceLength; i++) {
            if (!mask || mask[i] == 1) {
                sum += input[i * params.dimensions + gid];
                count++;
            }
        }
        
        output[gid] = count > 0 ? sum / float(count) : 0.0;
    }
    
    // Max pooling kernel
    kernel void max_pool(device const float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        device const int32_t* mask [[buffer(2)]],
                        constant PoolingParams& params [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
        if (gid >= params.dimensions) return;
        
        float maxVal = -INFINITY;
        bool foundValid = false;
        
        for (int i = 0; i < params.sequenceLength; i++) {
            if (!mask || mask[i] == 1) {
                float val = input[i * params.dimensions + gid];
                maxVal = max(maxVal, val);
                foundValid = true;
            }
        }
        
        output[gid] = foundValid ? maxVal : 0.0;
    }
    
    // Cosine similarity kernel
    kernel void cosine_similarity(device const float* queries [[buffer(0)]],
                                 device const float* keys [[buffer(1)]],
                                 device float* output [[buffer(2)]],
                                 constant SimilarityParams& params [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]]) {
        uint queryIdx = gid.y;
        uint keyIdx = gid.x;
        
        if (queryIdx >= params.queryCount || keyIdx >= params.keyCount) return;
        
        uint queryOffset = queryIdx * params.dimensions;
        uint keyOffset = keyIdx * params.dimensions;
        
        // Calculate dot product
        float dotProduct = 0.0;
        float queryNorm = 0.0;
        float keyNorm = 0.0;
        
        for (int i = 0; i < params.dimensions; i++) {
            float queryVal = queries[queryOffset + i];
            float keyVal = keys[keyOffset + i];
            
            dotProduct += queryVal * keyVal;
            queryNorm += queryVal * queryVal;
            keyNorm += keyVal * keyVal;
        }
        
        // Calculate cosine similarity
        float similarity = 0.0;
        float normProduct = sqrt(queryNorm * keyNorm);
        if (normProduct > 0.0) {
            similarity = dotProduct / normProduct;
        }
        
        output[queryIdx * params.keyCount + keyIdx] = similarity;
    }
    
    // Attention-weighted pooling kernel
    kernel void attention_weighted_pool(device const float* input [[buffer(0)]],
                                       device const float* weights [[buffer(1)]],
                                       device float* output [[buffer(2)]],
                                       constant PoolingParams& params [[buffer(3)]],
                                       uint gid [[thread_position_in_grid]]) {
        if (gid >= params.dimensions) return;
        
        float weightedSum = 0.0;
        float weightSum = 0.0;
        
        for (int i = 0; i < params.sequenceLength; i++) {
            float weight = weights[i];
            weightedSum += input[i * params.dimensions + gid] * weight;
            weightSum += weight;
        }
        
        output[gid] = weightSum > 0.0 ? weightedSum / weightSum : 0.0;
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
