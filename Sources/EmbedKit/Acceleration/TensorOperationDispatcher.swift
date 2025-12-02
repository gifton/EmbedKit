// EmbedKit - Tensor Operation Dispatcher
//
// Type-safe operation dispatching for managed tensors with Metal 4 integration.
// Provides a high-level interface for GPU compute operations on ManagedTensor instances.
//
// Metal 4.0 (iOS 26+ / macOS 26+)

import Foundation

#if canImport(Metal)
@preconcurrency import Metal

// MARK: - Tensor Operation Types

/// Enumeration of supported tensor operations.
public enum TensorOperationType: String, Sendable, CaseIterable {
    /// L2 normalization
    case normalize

    /// Mean pooling
    case meanPool

    /// Max pooling
    case maxPool

    /// CLS token pooling
    case clsPool

    /// Attention-weighted pooling
    case attentionPool

    /// Fused pooling + normalization
    case fusedPoolNormalize

    /// Cosine similarity matrix
    case cosineSimilarity

    /// Dot product similarity (for pre-normalized vectors)
    case dotSimilarity

    /// Element-wise addition
    case add

    /// Element-wise multiplication
    case multiply

    /// Scale by constant
    case scale
}

/// Configuration for a tensor operation.
public struct TensorOperationConfig: Sendable {
    /// Operation type
    public let operation: TensorOperationType

    /// Pooling strategy (for pooling operations)
    public let poolingStrategy: PoolingStrategy

    /// Whether to normalize output
    public let normalize: Bool

    /// Optional attention mask
    public let mask: [Int32]?

    /// Optional attention weights (for attention pooling)
    public let weights: [Float]?

    /// Scale factor (for scale operation)
    public let scaleFactor: Float

    /// Default configuration for normalization
    public static func normalizeConfig() -> TensorOperationConfig {
        TensorOperationConfig(
            operation: .normalize,
            poolingStrategy: .mean,
            normalize: true,
            mask: nil,
            weights: nil,
            scaleFactor: 1.0
        )
    }

    /// Configuration for mean pooling with normalization
    public static func meanPoolNormalize(mask: [Int32]? = nil) -> TensorOperationConfig {
        TensorOperationConfig(
            operation: .fusedPoolNormalize,
            poolingStrategy: .mean,
            normalize: true,
            mask: mask,
            weights: nil,
            scaleFactor: 1.0
        )
    }

    /// Configuration for max pooling with normalization
    public static func maxPoolNormalize(mask: [Int32]? = nil) -> TensorOperationConfig {
        TensorOperationConfig(
            operation: .fusedPoolNormalize,
            poolingStrategy: .max,
            normalize: true,
            mask: mask,
            weights: nil,
            scaleFactor: 1.0
        )
    }

    /// Configuration for CLS pooling with normalization
    public static func clsPoolNormalize() -> TensorOperationConfig {
        TensorOperationConfig(
            operation: .fusedPoolNormalize,
            poolingStrategy: .cls,
            normalize: true,
            mask: nil,
            weights: nil,
            scaleFactor: 1.0
        )
    }

    /// Configuration for attention-weighted pooling
    public static func attentionPoolNormalize(weights: [Float]) -> TensorOperationConfig {
        TensorOperationConfig(
            operation: .attentionPool,
            poolingStrategy: .attention,
            normalize: true,
            mask: nil,
            weights: weights,
            scaleFactor: 1.0
        )
    }

    /// Configuration for cosine similarity
    public static func cosineSimilarity() -> TensorOperationConfig {
        TensorOperationConfig(
            operation: .cosineSimilarity,
            poolingStrategy: .mean,
            normalize: false,
            mask: nil,
            weights: nil,
            scaleFactor: 1.0
        )
    }

    /// Full initializer
    public init(
        operation: TensorOperationType,
        poolingStrategy: PoolingStrategy = .mean,
        normalize: Bool = true,
        mask: [Int32]? = nil,
        weights: [Float]? = nil,
        scaleFactor: Float = 1.0
    ) {
        self.operation = operation
        self.poolingStrategy = poolingStrategy
        self.normalize = normalize
        self.mask = mask
        self.weights = weights
        self.scaleFactor = scaleFactor
    }
}

/// Result of a tensor operation.
public struct TensorOperationResult: Sendable {
    /// Output tensor (if operation produces a tensor)
    public let outputTensor: ManagedTensor?

    /// Execution time in seconds
    public let executionTime: TimeInterval

    /// Whether the operation used GPU
    public let usedGPU: Bool

    /// Operation that was executed
    public let operation: TensorOperationType

    /// Any warnings or notes
    public let notes: String?
}

// MARK: - Tensor Operation Dispatcher

/// Actor for dispatching GPU operations on managed tensors.
///
/// `TensorOperationDispatcher` provides a high-level interface for executing
/// GPU compute operations on `ManagedTensor` instances, integrating with
/// `TensorStorageManager` for lifecycle management.
///
/// ## Features
/// - Type-safe operation dispatching
/// - Automatic buffer management
/// - Integration with TensorStorageManager
/// - Metal 4 unified encoding support
/// - Performance tracking
///
/// ## Usage
/// ```swift
/// let dispatcher = TensorOperationDispatcher(
///     accelerator: accelerator,
///     storageManager: storageManager
/// )
///
/// // Pool and normalize token embeddings
/// let result = try await dispatcher.execute(
///     input: tokenEmbeddings,
///     output: pooledEmbeddings,
///     config: .meanPoolNormalize(),
///     inputShape: (batchSize: 32, sequenceLength: 128, dimensions: 384)
/// )
/// ```
public actor TensorOperationDispatcher {

    /// The Metal accelerator for GPU operations
    private let accelerator: MetalAccelerator

    /// The storage manager for tensor lifecycle
    private let storageManager: TensorStorageManager

    /// The Metal device
    private let device: MTLDevice

    /// The command queue
    private let commandQueue: MTLCommandQueue

    /// Statistics tracking
    private var stats: Statistics = Statistics()

    // MARK: - Types

    /// Input shape descriptor for operations.
    public struct InputShape: Sendable {
        public let batchSize: Int
        public let sequenceLength: Int
        public let dimensions: Int

        public init(batchSize: Int, sequenceLength: Int, dimensions: Int) {
            self.batchSize = batchSize
            self.sequenceLength = sequenceLength
            self.dimensions = dimensions
        }

        /// Create shape for embeddings (2D: batch x dimensions)
        public static func embedding(batchSize: Int, dimensions: Int) -> InputShape {
            InputShape(batchSize: batchSize, sequenceLength: 1, dimensions: dimensions)
        }

        /// Create shape for token embeddings (3D: batch x sequence x dimensions)
        public static func tokenEmbedding(batchSize: Int, sequenceLength: Int, dimensions: Int) -> InputShape {
            InputShape(batchSize: batchSize, sequenceLength: sequenceLength, dimensions: dimensions)
        }

        /// Total element count
        public var elementCount: Int {
            batchSize * sequenceLength * dimensions
        }
    }

    /// Dispatcher statistics.
    public struct Statistics: Sendable {
        /// Total operations dispatched
        public var totalOperations: Int = 0

        /// Operations executed on GPU
        public var gpuOperations: Int = 0

        /// Operations executed on CPU (fallback)
        public var cpuOperations: Int = 0

        /// Total GPU execution time
        public var totalGPUTime: TimeInterval = 0

        /// Total CPU execution time
        public var totalCPUTime: TimeInterval = 0

        /// Average GPU operation time
        public var averageGPUTime: TimeInterval {
            gpuOperations > 0 ? totalGPUTime / Double(gpuOperations) : 0
        }
    }

    // MARK: - Initialization

    /// Initialize the operation dispatcher.
    ///
    /// - Parameters:
    ///   - accelerator: Metal accelerator for GPU operations
    ///   - storageManager: Storage manager for tensor lifecycle
    public init(
        accelerator: MetalAccelerator,
        storageManager: TensorStorageManager
    ) async {
        self.accelerator = accelerator
        self.storageManager = storageManager
        self.device = storageManager.device
        self.commandQueue = storageManager.device.makeCommandQueue()!
    }

    // MARK: - Operation Execution

    /// Execute a tensor operation.
    ///
    /// - Parameters:
    ///   - input: Input tensor
    ///   - output: Output tensor (must be pre-allocated with correct size)
    ///   - config: Operation configuration
    ///   - inputShape: Shape of the input data
    /// - Returns: Operation result with timing information
    public func execute(
        input: ManagedTensor,
        output: ManagedTensor,
        config: TensorOperationConfig,
        inputShape: InputShape
    ) async throws -> TensorOperationResult {
        let startTime = Date()

        // Mark tensors as accessed
        await storageManager.markAccessed(input)
        await storageManager.markAccessed(output)

        // Dispatch based on operation type
        let usedGPU: Bool
        var notes: String? = nil

        switch config.operation {
        case .normalize:
            usedGPU = try await executeNormalize(
                input: input,
                output: output,
                batchSize: inputShape.batchSize,
                dimensions: inputShape.dimensions
            )

        case .meanPool, .maxPool, .clsPool:
            usedGPU = try await executePooling(
                input: input,
                output: output,
                config: config,
                inputShape: inputShape
            )

        case .fusedPoolNormalize:
            usedGPU = try await executeFusedPoolNormalize(
                input: input,
                output: output,
                config: config,
                inputShape: inputShape
            )

        case .attentionPool:
            guard let weights = config.weights else {
                throw EmbedKitError.invalidConfiguration("Attention pooling requires weights")
            }
            usedGPU = try await executeAttentionPool(
                input: input,
                output: output,
                weights: weights,
                inputShape: inputShape,
                normalize: config.normalize
            )

        case .cosineSimilarity, .dotSimilarity:
            notes = "Similarity operations require separate query/key inputs - use executeSimilarity method"
            usedGPU = false

        case .add, .multiply, .scale:
            notes = "Element-wise operations not yet implemented"
            usedGPU = false
        }

        let executionTime = Date().timeIntervalSince(startTime)

        // Update statistics
        stats.totalOperations += 1
        if usedGPU {
            stats.gpuOperations += 1
            stats.totalGPUTime += executionTime
        } else {
            stats.cpuOperations += 1
            stats.totalCPUTime += executionTime
        }

        return TensorOperationResult(
            outputTensor: output,
            executionTime: executionTime,
            usedGPU: usedGPU,
            operation: config.operation,
            notes: notes
        )
    }

    /// Execute a similarity operation between two tensors.
    ///
    /// - Parameters:
    ///   - queries: Query tensor
    ///   - keys: Key tensor
    ///   - output: Output tensor for similarity matrix
    ///   - queryCount: Number of query vectors
    ///   - keyCount: Number of key vectors
    ///   - dimensions: Vector dimensions
    ///   - normalized: Whether vectors are already normalized
    /// - Returns: Operation result
    public func executeSimilarity(
        queries: ManagedTensor,
        keys: ManagedTensor,
        output: ManagedTensor,
        queryCount: Int,
        keyCount: Int,
        dimensions: Int,
        normalized: Bool = true
    ) async throws -> TensorOperationResult {
        let startTime = Date()

        // Mark tensors as accessed
        await storageManager.markAccessed(queries)
        await storageManager.markAccessed(keys)
        await storageManager.markAccessed(output)

        // Check if GPU is available
        let isGPUAvailable = await accelerator.isAvailable

        let usedGPU: Bool

        if isGPUAvailable && queryCount * keyCount >= 64 {
            // Use GPU for larger matrices
            try await dispatchSimilarityGPU(
                queries: queries,
                keys: keys,
                output: output,
                queryCount: queryCount,
                keyCount: keyCount,
                dimensions: dimensions,
                normalized: normalized
            )
            usedGPU = true
        } else {
            // CPU fallback
            executeSimilarityCPU(
                queries: queries,
                keys: keys,
                output: output,
                queryCount: queryCount,
                keyCount: keyCount,
                dimensions: dimensions,
                normalized: normalized
            )
            usedGPU = false
        }

        let executionTime = Date().timeIntervalSince(startTime)

        // Update statistics
        stats.totalOperations += 1
        if usedGPU {
            stats.gpuOperations += 1
            stats.totalGPUTime += executionTime
        } else {
            stats.cpuOperations += 1
            stats.totalCPUTime += executionTime
        }

        return TensorOperationResult(
            outputTensor: output,
            executionTime: executionTime,
            usedGPU: usedGPU,
            operation: normalized ? .dotSimilarity : .cosineSimilarity,
            notes: nil
        )
    }

    // MARK: - Private Implementation

    /// Execute L2 normalization.
    private func executeNormalize(
        input: ManagedTensor,
        output: ManagedTensor,
        batchSize: Int,
        dimensions: Int
    ) async throws -> Bool {
        let isGPUAvailable = await accelerator.isAvailable

        if isGPUAvailable && batchSize * dimensions >= 1024 {
            // Read from input, normalize, write to output using GPU
            let inputPtr = input.buffer.contents().bindMemory(to: Float.self, capacity: batchSize * dimensions)
            let inputArray = Array(UnsafeBufferPointer(start: inputPtr, count: batchSize * dimensions))

            // Reshape to 2D for accelerator
            var vectors: [[Float]] = []
            for b in 0..<batchSize {
                let start = b * dimensions
                vectors.append(Array(inputArray[start..<start + dimensions]))
            }

            let normalized = await accelerator.l2Normalize(vectors)

            // Write to output
            let outputPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: batchSize * dimensions)
            for (b, vec) in normalized.enumerated() {
                for (d, val) in vec.enumerated() {
                    outputPtr[b * dimensions + d] = val
                }
            }

            return true
        } else {
            // CPU fallback
            executeNormalizeCPU(input: input, output: output, batchSize: batchSize, dimensions: dimensions)
            return false
        }
    }

    /// CPU implementation of L2 normalization.
    private func executeNormalizeCPU(
        input: ManagedTensor,
        output: ManagedTensor,
        batchSize: Int,
        dimensions: Int
    ) {
        let inputPtr = input.buffer.contents().bindMemory(to: Float.self, capacity: batchSize * dimensions)
        let outputPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: batchSize * dimensions)

        for b in 0..<batchSize {
            let offset = b * dimensions

            // Compute norm
            var sumSquares: Double = 0
            for d in 0..<dimensions {
                let val = Double(inputPtr[offset + d])
                sumSquares += val * val
            }
            let norm = Float(max(1e-12, sqrt(sumSquares)))

            // Normalize
            for d in 0..<dimensions {
                outputPtr[offset + d] = inputPtr[offset + d] / norm
            }
        }
    }

    /// Execute pooling operation.
    private func executePooling(
        input: ManagedTensor,
        output: ManagedTensor,
        config: TensorOperationConfig,
        inputShape: InputShape
    ) async throws -> Bool {
        // Delegate to fused operation with normalize=false for standalone pooling
        var modifiedConfig = config
        modifiedConfig = TensorOperationConfig(
            operation: .fusedPoolNormalize,
            poolingStrategy: config.poolingStrategy,
            normalize: false,
            mask: config.mask,
            weights: config.weights,
            scaleFactor: config.scaleFactor
        )

        return try await executeFusedPoolNormalize(
            input: input,
            output: output,
            config: modifiedConfig,
            inputShape: inputShape
        )
    }

    /// Execute fused pooling + normalization.
    private func executeFusedPoolNormalize(
        input: ManagedTensor,
        output: ManagedTensor,
        config: TensorOperationConfig,
        inputShape: InputShape
    ) async throws -> Bool {
        let isGPUAvailable = await accelerator.isAvailable
        let workloadSize = inputShape.batchSize * inputShape.sequenceLength * inputShape.dimensions

        if isGPUAvailable && workloadSize >= 4096 {
            // Read from input buffer
            let inputPtr = input.buffer.contents().bindMemory(to: Float.self, capacity: workloadSize)
            let embeddings = Array(UnsafeBufferPointer(start: inputPtr, count: workloadSize))

            // Execute via accelerator
            let results = await accelerator.tensorPoolNormalize(
                embeddings: embeddings,
                batchSize: inputShape.batchSize,
                sequenceLength: inputShape.sequenceLength,
                dimensions: inputShape.dimensions,
                masks: config.mask,
                strategy: config.poolingStrategy,
                normalize: config.normalize
            )

            // Write to output
            let outputPtr = output.buffer.contents().bindMemory(
                to: Float.self,
                capacity: inputShape.batchSize * inputShape.dimensions
            )
            for (b, vec) in results.enumerated() {
                for (d, val) in vec.enumerated() {
                    outputPtr[b * inputShape.dimensions + d] = val
                }
            }

            return true
        } else {
            // CPU fallback
            executeFusedPoolNormalizeCPU(
                input: input,
                output: output,
                config: config,
                inputShape: inputShape
            )
            return false
        }
    }

    /// CPU implementation of fused pool + normalize.
    private func executeFusedPoolNormalizeCPU(
        input: ManagedTensor,
        output: ManagedTensor,
        config: TensorOperationConfig,
        inputShape: InputShape
    ) {
        let inputPtr = input.buffer.contents().bindMemory(
            to: Float.self,
            capacity: inputShape.elementCount
        )
        let outputPtr = output.buffer.contents().bindMemory(
            to: Float.self,
            capacity: inputShape.batchSize * inputShape.dimensions
        )

        let elementsPerSequence = inputShape.sequenceLength * inputShape.dimensions

        for b in 0..<inputShape.batchSize {
            let seqStart = b * elementsPerSequence
            let maskOffset = b * inputShape.sequenceLength

            // Pool
            var pooled = [Float](repeating: 0, count: inputShape.dimensions)

            switch config.poolingStrategy {
            case .mean:
                var count = 0
                for t in 0..<inputShape.sequenceLength {
                    let isValid = config.mask == nil || config.mask![maskOffset + t] == 1
                    if isValid {
                        for d in 0..<inputShape.dimensions {
                            pooled[d] += inputPtr[seqStart + t * inputShape.dimensions + d]
                        }
                        count += 1
                    }
                }
                if count > 0 {
                    let scale = 1.0 / Float(count)
                    for d in 0..<inputShape.dimensions {
                        pooled[d] *= scale
                    }
                }

            case .max:
                pooled = [Float](repeating: -.greatestFiniteMagnitude, count: inputShape.dimensions)
                for t in 0..<inputShape.sequenceLength {
                    let isValid = config.mask == nil || config.mask![maskOffset + t] == 1
                    if isValid {
                        for d in 0..<inputShape.dimensions {
                            pooled[d] = max(pooled[d], inputPtr[seqStart + t * inputShape.dimensions + d])
                        }
                    }
                }

            case .cls:
                for d in 0..<inputShape.dimensions {
                    pooled[d] = inputPtr[seqStart + d]
                }

            case .attention:
                // Fall back to mean without weights
                var count = 0
                for t in 0..<inputShape.sequenceLength {
                    for d in 0..<inputShape.dimensions {
                        pooled[d] += inputPtr[seqStart + t * inputShape.dimensions + d]
                    }
                    count += 1
                }
                if count > 0 {
                    let scale = 1.0 / Float(count)
                    for d in 0..<inputShape.dimensions {
                        pooled[d] *= scale
                    }
                }
            }

            // Normalize if requested
            if config.normalize {
                var sumSquares: Double = 0
                for d in 0..<inputShape.dimensions {
                    sumSquares += Double(pooled[d]) * Double(pooled[d])
                }
                let norm = Float(max(1e-12, sqrt(sumSquares)))
                for d in 0..<inputShape.dimensions {
                    pooled[d] /= norm
                }
            }

            // Write output
            for d in 0..<inputShape.dimensions {
                outputPtr[b * inputShape.dimensions + d] = pooled[d]
            }
        }
    }

    /// Execute attention pooling.
    private func executeAttentionPool(
        input: ManagedTensor,
        output: ManagedTensor,
        weights: [Float],
        inputShape: InputShape,
        normalize: Bool
    ) async throws -> Bool {
        let isGPUAvailable = await accelerator.isAvailable
        let workloadSize = inputShape.batchSize * inputShape.sequenceLength * inputShape.dimensions

        if isGPUAvailable && workloadSize >= 4096 {
            // Read from input buffer
            let inputPtr = input.buffer.contents().bindMemory(to: Float.self, capacity: workloadSize)
            let embeddings = Array(UnsafeBufferPointer(start: inputPtr, count: workloadSize))

            // Execute via accelerator
            let results = await accelerator.tensorAttentionPoolNormalize(
                embeddings: embeddings,
                weights: weights,
                batchSize: inputShape.batchSize,
                sequenceLength: inputShape.sequenceLength,
                dimensions: inputShape.dimensions,
                normalize: normalize
            )

            // Write to output
            let outputPtr = output.buffer.contents().bindMemory(
                to: Float.self,
                capacity: inputShape.batchSize * inputShape.dimensions
            )
            for (b, vec) in results.enumerated() {
                for (d, val) in vec.enumerated() {
                    outputPtr[b * inputShape.dimensions + d] = val
                }
            }

            return true
        } else {
            // CPU fallback
            executeAttentionPoolCPU(
                input: input,
                output: output,
                weights: weights,
                inputShape: inputShape,
                normalize: normalize
            )
            return false
        }
    }

    /// CPU implementation of attention pooling.
    private func executeAttentionPoolCPU(
        input: ManagedTensor,
        output: ManagedTensor,
        weights: [Float],
        inputShape: InputShape,
        normalize: Bool
    ) {
        let inputPtr = input.buffer.contents().bindMemory(
            to: Float.self,
            capacity: inputShape.elementCount
        )
        let outputPtr = output.buffer.contents().bindMemory(
            to: Float.self,
            capacity: inputShape.batchSize * inputShape.dimensions
        )

        let elementsPerSequence = inputShape.sequenceLength * inputShape.dimensions

        for b in 0..<inputShape.batchSize {
            let seqStart = b * elementsPerSequence
            let weightStart = b * inputShape.sequenceLength

            // Compute weight sum
            var weightSum: Float = 0
            for t in 0..<inputShape.sequenceLength {
                weightSum += weights[weightStart + t]
            }
            let invWeightSum = weightSum > 1e-12 ? 1.0 / weightSum : 0.0

            // Compute weighted average
            var pooled = [Float](repeating: 0, count: inputShape.dimensions)
            for t in 0..<inputShape.sequenceLength {
                let weight = weights[weightStart + t]
                for d in 0..<inputShape.dimensions {
                    pooled[d] += inputPtr[seqStart + t * inputShape.dimensions + d] * weight
                }
            }
            for d in 0..<inputShape.dimensions {
                pooled[d] *= invWeightSum
            }

            // Normalize if requested
            if normalize {
                var sumSquares: Double = 0
                for d in 0..<inputShape.dimensions {
                    sumSquares += Double(pooled[d]) * Double(pooled[d])
                }
                let norm = Float(max(1e-12, sqrt(sumSquares)))
                for d in 0..<inputShape.dimensions {
                    pooled[d] /= norm
                }
            }

            // Write output
            for d in 0..<inputShape.dimensions {
                outputPtr[b * inputShape.dimensions + d] = pooled[d]
            }
        }
    }

    /// Dispatch similarity computation to GPU.
    private func dispatchSimilarityGPU(
        queries: ManagedTensor,
        keys: ManagedTensor,
        output: ManagedTensor,
        queryCount: Int,
        keyCount: Int,
        dimensions: Int,
        normalized: Bool
    ) async throws {
        // Read query and key buffers
        let queryPtr = queries.buffer.contents().bindMemory(to: Float.self, capacity: queryCount * dimensions)
        let keyPtr = keys.buffer.contents().bindMemory(to: Float.self, capacity: keyCount * dimensions)

        let queryArray = Array(UnsafeBufferPointer(start: queryPtr, count: queryCount * dimensions))
        let keyArray = Array(UnsafeBufferPointer(start: keyPtr, count: keyCount * dimensions))

        // Execute via accelerator
        let results = await accelerator.tensorSimilarityMatrix(
            queries: queryArray,
            keys: keyArray,
            queryBatchSize: queryCount,
            keyBatchSize: keyCount,
            dimensions: dimensions,
            normalized: normalized
        )

        // Write to output
        let outputPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: queryCount * keyCount)
        for (q, row) in results.enumerated() {
            for (k, val) in row.enumerated() {
                outputPtr[q * keyCount + k] = val
            }
        }
    }

    /// CPU implementation of similarity computation.
    private func executeSimilarityCPU(
        queries: ManagedTensor,
        keys: ManagedTensor,
        output: ManagedTensor,
        queryCount: Int,
        keyCount: Int,
        dimensions: Int,
        normalized: Bool
    ) {
        let queryPtr = queries.buffer.contents().bindMemory(to: Float.self, capacity: queryCount * dimensions)
        let keyPtr = keys.buffer.contents().bindMemory(to: Float.self, capacity: keyCount * dimensions)
        let outputPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: queryCount * keyCount)

        for q in 0..<queryCount {
            let qOffset = q * dimensions

            for k in 0..<keyCount {
                let kOffset = k * dimensions

                if normalized {
                    // Dot product for normalized vectors
                    var dot: Float = 0
                    for d in 0..<dimensions {
                        dot += queryPtr[qOffset + d] * keyPtr[kOffset + d]
                    }
                    outputPtr[q * keyCount + k] = dot
                } else {
                    // Full cosine similarity
                    var dot: Float = 0
                    var normQ: Float = 0
                    var normK: Float = 0
                    for d in 0..<dimensions {
                        let qVal = queryPtr[qOffset + d]
                        let kVal = keyPtr[kOffset + d]
                        dot += qVal * kVal
                        normQ += qVal * qVal
                        normK += kVal * kVal
                    }
                    let denom = sqrt(max(normQ, 1e-12)) * sqrt(max(normK, 1e-12))
                    outputPtr[q * keyCount + k] = dot / denom
                }
            }
        }
    }

    // MARK: - Statistics

    /// Get current statistics.
    public func getStatistics() -> Statistics {
        stats
    }

    /// Reset statistics.
    public func resetStatistics() {
        stats = Statistics()
    }
}

// MARK: - Convenience Extensions

extension TensorStorageManager {

    /// Create an operation dispatcher for this storage manager.
    ///
    /// - Parameter accelerator: Metal accelerator to use
    /// - Returns: A new operation dispatcher
    public func createOperationDispatcher(
        accelerator: MetalAccelerator
    ) async -> TensorOperationDispatcher {
        await TensorOperationDispatcher(accelerator: accelerator, storageManager: self)
    }
}

#endif
