import Foundation
@preconcurrency import Metal
import OSLog

/// Slim coordinator for Metal-accelerated operations
/// 
/// This refactored MetalAccelerator now acts as a lightweight coordinator that delegates
/// to specialized processor actors for different types of operations:
/// - MetalVectorProcessor: Vector normalization and mathematical operations
/// - MetalPoolingProcessor: Pooling strategies (mean, max, CLS, attention-weighted)
/// - MetalSimilarityProcessor: Cosine similarity and matrix operations
/// - MetalResourceManager: Device, queue, and pipeline management
///
/// Benefits of this architecture:
/// - Single Responsibility Principle: Each component has a focused purpose
/// - Better testability: Components can be tested in isolation
/// - Improved maintainability: Changes to one operation type don't affect others
/// - Enhanced flexibility: Easy to add new operation types or swap implementations
public actor MetalAccelerator {
    nonisolated private let logger = EmbedKitLogger.metal()
    
    // Specialized processor components
    private let resourceManager: MetalResourceManager
    private let vectorProcessor: MetalVectorProcessor
    private let poolingProcessor: MetalPoolingProcessor
    private let similarityProcessor: MetalSimilarityProcessor
    
    /// Get shared instance for the default GPU
    public static let shared: MetalAccelerator? = {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return nil
        }
        guard let accelerator = try? MetalAccelerator(device: device) else {
            return nil
        }
        return accelerator
    }()
    
    public init(device: MTLDevice) throws {
        // Initialize resource manager first
        self.resourceManager = try MetalResourceManager(device: device)
        
        // Initialize specialized processors
        self.vectorProcessor = MetalVectorProcessor(resourceManager: resourceManager)
        self.poolingProcessor = MetalPoolingProcessor(resourceManager: resourceManager)
        self.similarityProcessor = MetalSimilarityProcessor(resourceManager: resourceManager)
        
        logger.info("MetalAccelerator coordinator initialized with specialized processors")
    }
    
    /// Ensure pipelines are ready for use
    public func setupPipelines() async throws {
        try await resourceManager.setupPipelines()
    }
    
    // MARK: - Vector Operations
    
    /// Normalize a batch of vectors using L2 normalization
    ///
    /// Delegates to MetalVectorProcessor for specialized vector operations.
    public func normalizeVectors(_ vectors: [[Float]]) async throws -> [[Float]] {
        return try await vectorProcessor.normalizeVectors(vectors)
    }
    
    /// Metal 3 optimization: Fast batch normalization with epsilon parameter
    public func fastBatchNormalize(_ vectors: [[Float]], epsilon: Float = 1e-6) async throws -> [[Float]] {
        return try await vectorProcessor.fastBatchNormalize(vectors, epsilon: epsilon)
    }
    
    // MARK: - Pooling Operations
    
    /// Pool token embeddings using the specified strategy
    ///
    /// Delegates to MetalPoolingProcessor for specialized pooling operations.
    public func poolEmbeddings(
        _ tokenEmbeddings: [[Float]],
        strategy: PoolingStrategy,
        attentionMask: [Int]? = nil
    ) async throws -> [Float] {
        return try await poolingProcessor.poolEmbeddings(
            tokenEmbeddings,
            strategy: strategy,
            attentionMask: attentionMask
        )
    }
    
    /// Attention-weighted pooling implementation
    public func attentionWeightedPooling(
        _ tokenEmbeddings: [[Float]],
        attentionWeights: [Float]
    ) async throws -> [Float] {
        return try await poolingProcessor.attentionWeightedPooling(
            tokenEmbeddings,
            attentionWeights: attentionWeights
        )
    }
    
    // MARK: - Similarity Operations
    
    /// Calculate cosine similarity matrix between two sets of vectors
    ///
    /// Delegates to MetalSimilarityProcessor for specialized similarity calculations.
    public func cosineSimilarityMatrix(queries: [[Float]], keys: [[Float]]) async throws -> [[Float]] {
        return try await similarityProcessor.cosineSimilarityMatrix(queries: queries, keys: keys)
    }
    
    /// Calculate similarity between a single query and multiple keys
    public func cosineSimilarity(query: [Float], keys: [[Float]]) async throws -> [Float] {
        return try await similarityProcessor.cosineSimilarity(query: query, keys: keys)
    }
    
    // MARK: - Resource Management
    
    /// Handle memory pressure by delegating to resource manager
    public func handleMemoryPressure() async {
        await resourceManager.handleMemoryPressure()
    }
    
    /// Get current GPU memory usage in bytes
    nonisolated public func getCurrentMemoryUsage() -> Int64 {
        return resourceManager.getCurrentMemoryUsage()
    }
    
    /// Check if Metal acceleration is available
    nonisolated public var isAvailable: Bool {
        return resourceManager.isAvailable
    }
    
    // MARK: - Metal 3 Optimizations
    
    /// Metal 3 optimization: Process multiple operations in parallel using async compute
    ///
    /// This method demonstrates how the coordinator can orchestrate multiple specialized
    /// processors working in parallel for complex multi-operation workloads.
    public func parallelBatchProcess(
        normalizeVectors: [[Float]]? = nil,
        poolEmbeddings: (embeddings: [[Float]], strategy: PoolingStrategy, mask: [Int]?)? = nil,
        cosineSimilarity: (queries: [[Float]], keys: [[Float]])? = nil
    ) async throws -> (
        normalized: [[Float]]?,
        pooled: [Float]?,
        similarity: [[Float]]?
    ) {
        // Check if device supports Metal 3 for parallel processing
        guard resourceManager.isAvailable else {
            // Fallback to sequential processing
            return try await sequentialBatchProcess(
                normalizeVectors: normalizeVectors,
                poolEmbeddings: poolEmbeddings,
                cosineSimilarity: cosineSimilarity
            )
        }
        
        // Metal 3: Use parallel execution across specialized processors
        return try await withThrowingTaskGroup(of: BatchResult.self) { group in
            var normalized: [[Float]]?
            var pooled: [Float]?
            var similarity: [[Float]]?
            
            // Submit parallel tasks to specialized processors
            if let vectors = normalizeVectors {
                group.addTask {
                    let result = try await self.vectorProcessor.normalizeVectors(vectors)
                    return .normalized(result)
                }
            }
            
            if let pool = poolEmbeddings {
                group.addTask {
                    let result = try await self.poolingProcessor.poolEmbeddings(
                        pool.embeddings,
                        strategy: pool.strategy,
                        attentionMask: pool.mask
                    )
                    return .pooled(result)
                }
            }
            
            if let sim = cosineSimilarity {
                group.addTask {
                    let result = try await self.similarityProcessor.cosineSimilarityMatrix(
                        queries: sim.queries,
                        keys: sim.keys
                    )
                    return .similarity(result)
                }
            }
            
            // Collect results as they complete
            for try await result in group {
                switch result {
                case .normalized(let vectors):
                    normalized = vectors
                case .pooled(let vector):
                    pooled = vector
                case .similarity(let matrix):
                    similarity = matrix
                }
            }
            
            return (normalized, pooled, similarity)
        }
    }
    
    // MARK: - Private Implementation
    
    private func sequentialBatchProcess(
        normalizeVectors: [[Float]]? = nil,
        poolEmbeddings: (embeddings: [[Float]], strategy: PoolingStrategy, mask: [Int]?)? = nil,
        cosineSimilarity: (queries: [[Float]], keys: [[Float]])? = nil
    ) async throws -> (
        normalized: [[Float]]?,
        pooled: [Float]?,
        similarity: [[Float]]?
    ) {
        let normalized: [[Float]]?
        if let vectors = normalizeVectors {
            normalized = try await self.normalizeVectors(vectors)
        } else {
            normalized = nil
        }
        
        let pooled: [Float]?
        if let pool = poolEmbeddings {
            pooled = try await self.poolEmbeddings(
                pool.embeddings,
                strategy: pool.strategy,
                attentionMask: pool.mask
            )
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
}

// MARK: - Supporting Types

/// Result types for parallel batch processing
private enum BatchResult {
    case normalized([[Float]])
    case pooled([Float])
    case similarity([[Float]])
}

/// Metal error types
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

/// Factory for creating MetalAccelerator instances
public extension MetalAccelerator {
    /// Create a MetalAccelerator instance with custom resource manager
    static func create(with resourceManager: MetalResourceManager) -> MetalAccelerator? {
        // This would require some refactoring to support dependency injection
        // For now, we use the standard initializer
        return shared
    }
}
