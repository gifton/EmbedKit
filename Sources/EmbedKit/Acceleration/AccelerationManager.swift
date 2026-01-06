// EmbedKit - Acceleration Manager
// GPU-accelerated distance and normalization operations using Metal4

import Foundation
import VectorCore
import VectorAccelerate

// MARK: - Acceleration Manager

/// Actor that provides GPU-accelerated vector operations.
///
/// `AccelerationManager` uses VectorAccelerate's Metal4 kernels for
/// GPU-accelerated distance computation and normalization.
///
/// ## Example Usage
/// ```swift
/// let manager = try await AccelerationManager.create()
///
/// // Compute batch distances
/// let distances = try await manager.batchDistance(
///     from: queryVector,
///     to: candidateVectors,
///     metric: .cosine
/// )
///
/// // Normalize vectors
/// let normalized = try await manager.normalizeBatch(vectors)
/// ```
public actor AccelerationManager {

    // MARK: - Shared Instance

    /// Shared instance for convenience.
    public static func shared() async throws -> AccelerationManager {
        try await SharedAccelerationManager.instance
    }

    // MARK: - Properties

    /// Metal4 context for GPU operations.
    private let context: Metal4Context

    /// Distance provider for batch distance computation.
    private let distanceProvider: UniversalKernelDistanceProvider

    /// Normalization kernel.
    private let normalizationKernel: L2NormalizationKernel

    /// UMAP gradient kernel for dimensionality reduction.
    private let umapKernel: UMAPGradientKernel

    /// Statistics tracking.
    private var stats: MutableStatistics

    // MARK: - Initialization

    /// Create an acceleration manager with a new Metal4Context.
    public init() async throws {
        self.context = try await Metal4ContextManager.shared()
        self.distanceProvider = try await UniversalKernelDistanceProvider(context: context)
        self.normalizationKernel = try await L2NormalizationKernel(context: context)
        self.umapKernel = try await UMAPGradientKernel(context: context)
        self.stats = MutableStatistics()
    }

    /// Create an acceleration manager with a shared Metal4Context.
    public init(context: Metal4Context) async throws {
        self.context = context
        self.distanceProvider = try await UniversalKernelDistanceProvider(context: context)
        self.normalizationKernel = try await L2NormalizationKernel(context: context)
        self.umapKernel = try await UMAPGradientKernel(context: context)
        self.stats = MutableStatistics()
    }

    /// Factory method for creating an acceleration manager.
    public static func create() async throws -> AccelerationManager {
        try await AccelerationManager()
    }

    /// Factory method with shared context.
    public static func create(context: Metal4Context) async throws -> AccelerationManager {
        try await AccelerationManager(context: context)
    }

    // MARK: - Distance Operations

    /// Compute batch distances from a query to multiple candidates.
    ///
    /// Uses GPU-accelerated distance computation via Metal4 kernels.
    ///
    /// - Parameters:
    ///   - query: Query vector
    ///   - candidates: Candidate vectors to compute distances to
    ///   - metric: Distance metric to use
    /// - Returns: Array of distances, one per candidate
    public func batchDistance(
        from query: [Float],
        to candidates: [[Float]],
        metric: SupportedDistanceMetric
    ) async throws -> [Float] {
        guard !candidates.isEmpty else { return [] }

        // Validate dimensions
        let dimension = query.count
        for candidate in candidates {
            guard candidate.count == dimension else {
                throw AccelerationError.dimensionMismatch(
                    expected: dimension,
                    got: candidate.count
                )
            }
        }

        let start = CFAbsoluteTimeGetCurrent()

        // Use VectorAccelerate's UniversalKernelDistanceProvider
        let queryVec = DynamicVector(query)
        let candidateVecs = candidates.map { DynamicVector($0) }

        let distances = try await distanceProvider.batchDistance(
            from: queryVec,
            to: candidateVecs,
            metric: metric
        )

        stats.gpuOperations += 1
        stats.gpuTimeTotal += CFAbsoluteTimeGetCurrent() - start

        return distances
    }

    /// Compute distance between two vectors.
    public func distance(
        from a: [Float],
        to b: [Float],
        metric: SupportedDistanceMetric
    ) async throws -> Float {
        guard a.count == b.count else {
            throw AccelerationError.dimensionMismatch(expected: a.count, got: b.count)
        }

        let start = CFAbsoluteTimeGetCurrent()

        let vecA = DynamicVector(a)
        let vecB = DynamicVector(b)

        let result = try await distanceProvider.distance(from: vecA, to: vecB, metric: metric)

        stats.gpuOperations += 1
        stats.gpuTimeTotal += CFAbsoluteTimeGetCurrent() - start

        return result
    }

    // MARK: - Vector Operations

    /// Normalize a batch of vectors using GPU.
    public func normalizeBatch(_ vectors: [[Float]]) async throws -> [[Float]] {
        guard !vectors.isEmpty else { return [] }

        let start = CFAbsoluteTimeGetCurrent()

        let result = try await normalizationKernel.normalize(vectors)
        let normalized = result.asArrays()

        stats.gpuOperations += 1
        stats.gpuTimeTotal += CFAbsoluteTimeGetCurrent() - start

        return normalized
    }

    /// Normalize a single vector.
    ///
    /// For single vectors, uses CPU (GPU overhead not worth it).
    public nonisolated func normalize(_ vector: [Float]) -> [Float] {
        AccelerateBLAS.normalize(vector)
    }

    // MARK: - UMAP Projection

    /// Project high-dimensional embeddings to lower dimensions using UMAP.
    ///
    /// UMAP (Uniform Manifold Approximation and Projection) preserves local
    /// neighborhood structure while reducing dimensionality, making it ideal
    /// for visualizing embedding spaces.
    ///
    /// - Parameters:
    ///   - embeddings: Array of high-dimensional vectors (all same dimension)
    ///   - config: UMAP configuration parameters
    /// - Returns: Array of low-dimensional projections
    /// - Throws: `AccelerationError` if projection fails
    ///
    /// ## Example
    /// ```swift
    /// let points = try await accelerationManager.umapProject(
    ///     embeddings: vectors,
    ///     config: .visualization2D()
    /// )
    /// // points[i] = [x, y] for 2D projection
    /// ```
    public func umapProject(
        embeddings: [[Float]],
        config: UMAPConfiguration
    ) async throws -> [[Float]] {
        guard !embeddings.isEmpty else { return [] }

        let n = embeddings.count
        let sourceDim = embeddings[0].count

        // Validate dimensions
        for (i, emb) in embeddings.enumerated() {
            guard emb.count == sourceDim else {
                throw AccelerationError.dimensionMismatch(expected: sourceDim, got: emb.count)
            }
            // Check for NaN/Inf
            if emb.contains(where: { $0.isNaN || $0.isInfinite }) {
                throw AccelerationError.invalidInput("Embedding at index \(i) contains NaN or Inf values")
            }
        }

        // Validate config
        try config.validate()

        // Need at least k+1 points for k neighbors
        guard n > config.neighbors else {
            throw AccelerationError.invalidInput(
                "Need at least \(config.neighbors + 1) points for \(config.neighbors) neighbors, got \(n)"
            )
        }

        let start = CFAbsoluteTimeGetCurrent()
        let targetDim = config.targetDimension

        // Step 1: Build k-NN graph using L2 distance
        let edges = try await buildKNNGraph(
            embeddings: embeddings,
            k: config.neighbors
        )

        // Step 2: Sort edges by source for UMAP kernel
        let sortedEdges = umapKernel.sortEdgesBySource(edges)

        // Step 3: Initialize random low-dimensional embedding
        var lowDimEmbedding = initializeRandomEmbedding(n: n, d: targetDim)

        // Step 4: Run optimization epochs
        for epoch in 0..<config.iterations {
            let params = config.toKernelParameters(epoch: epoch, totalEpochs: config.iterations)
            try await umapKernel.optimizeEpoch(
                embedding: &lowDimEmbedding,
                edges: sortedEdges,
                params: params
            )
        }

        stats.gpuOperations += 1
        stats.gpuTimeTotal += CFAbsoluteTimeGetCurrent() - start

        return lowDimEmbedding
    }

    /// Compute UMAP gradients for custom optimization loops.
    ///
    /// This lower-level API provides gradients without applying them,
    /// useful for custom optimizers or when you need to inspect gradients.
    ///
    /// - Parameters:
    ///   - embeddings: High-dimensional embeddings [N, D]
    ///   - lowDimEmbeddings: Current low-dimensional embedding [N, targetDim]
    ///   - neighbors: Number of nearest neighbors for graph
    /// - Returns: Gradients for each point in low-dimensional space
    public func umapGradient(
        embeddings: [[Float]],
        lowDimEmbeddings: [[Float]],
        neighbors: Int = 15
    ) async throws -> [[Float]] {
        guard !embeddings.isEmpty else { return [] }
        guard embeddings.count == lowDimEmbeddings.count else {
            throw AccelerationError.dimensionMismatch(
                expected: embeddings.count,
                got: lowDimEmbeddings.count
            )
        }

        let n = embeddings.count
        let d = lowDimEmbeddings[0].count

        // Build k-NN graph
        let edges = try await buildKNNGraph(embeddings: embeddings, k: neighbors)
        let sortedEdges = umapKernel.sortEdgesBySource(edges)

        // Compute segment info
        let (starts, counts) = umapKernel.computeSegments(edges: sortedEdges, n: n)

        // Create GPU buffers
        let device = context.device.rawDevice
        let flatLowDim = lowDimEmbeddings.flatMap { $0 }

        guard let embeddingBuffer = device.makeBuffer(
            bytes: flatLowDim,
            length: flatLowDim.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw AccelerationError.gpuOperationFailed("Failed to allocate embedding buffer")
        }

        guard let edgeBuffer = device.makeBuffer(
            bytes: sortedEdges,
            length: sortedEdges.count * MemoryLayout<UMAPEdge>.size,
            options: .storageModeShared
        ) else {
            throw AccelerationError.gpuOperationFailed("Failed to allocate edge buffer")
        }

        guard let startsBuffer = device.makeBuffer(
            bytes: starts,
            length: starts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ),
        let countsBuffer = device.makeBuffer(
            bytes: counts,
            length: counts.count * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw AccelerationError.gpuOperationFailed("Failed to allocate segment buffers")
        }

        // Compute gradients
        let gradientBuffer = try await umapKernel.computeGradients(
            embedding: embeddingBuffer,
            edges: edgeBuffer,
            segmentStarts: startsBuffer,
            segmentCounts: countsBuffer,
            n: n,
            d: d,
            edgeCount: sortedEdges.count,
            params: .default
        )

        // Read back gradients
        let gradPtr = gradientBuffer.contents().bindMemory(to: Float.self, capacity: n * d)
        var gradients: [[Float]] = []
        for i in 0..<n {
            var row = [Float](repeating: 0, count: d)
            for j in 0..<d {
                row[j] = gradPtr[i * d + j]
            }
            gradients.append(row)
        }

        return gradients
    }

    // MARK: - Private UMAP Helpers

    /// Build k-NN graph from embeddings using L2 distance.
    private func buildKNNGraph(
        embeddings: [[Float]],
        k: Int
    ) async throws -> [UMAPEdge] {
        let n = embeddings.count
        var edges: [UMAPEdge] = []
        edges.reserveCapacity(n * k)

        // For each point, find k nearest neighbors using GPU distance
        for i in 0..<n {
            let query = embeddings[i]
            let distances = try await batchDistance(
                from: query,
                to: embeddings,
                metric: .euclidean
            )

            // Get k nearest (excluding self)
            let indexed = distances.enumerated()
                .filter { $0.offset != i }
                .sorted { $0.element < $1.element }
                .prefix(k)

            // Convert distances to weights (similarity)
            // Using exponential decay: w = exp(-d / sigma) where sigma is median distance
            let neighborDistances = indexed.map { $0.element }
            let sigma = max(neighborDistances.sorted()[neighborDistances.count / 2], 1e-6)

            for (targetIdx, dist) in indexed {
                let weight = exp(-dist / sigma)
                edges.append(UMAPEdge(source: i, target: targetIdx, weight: weight))
            }
        }

        return edges
    }

    /// Initialize random low-dimensional embedding.
    private nonisolated func initializeRandomEmbedding(n: Int, d: Int) -> [[Float]] {
        // Initialize with small random values (scaled by 1e-4 for stability)
        var embedding: [[Float]] = []
        embedding.reserveCapacity(n)
        for _ in 0..<n {
            var row = [Float](repeating: 0, count: d)
            for j in 0..<d {
                row[j] = Float.random(in: -1e-4...1e-4)
            }
            embedding.append(row)
        }
        return embedding
    }

    // MARK: - Statistics

    /// Get current acceleration statistics.
    public func statistics() -> AccelerationStatistics {
        AccelerationStatistics(
            gpuOperations: stats.gpuOperations,
            gpuTimeTotal: stats.gpuTimeTotal
        )
    }

    /// Reset statistics.
    public func resetStatistics() {
        stats = MutableStatistics()
    }

    // MARK: - GPU Availability

    /// Whether GPU acceleration is available.
    /// Always true for AccelerationManager since it requires Metal4.
    public nonisolated var isGPUAvailable: Bool {
        true
    }
}

// MARK: - Statistics Types

/// Statistics about GPU usage in acceleration operations.
public struct AccelerationStatistics: Sendable {
    /// Number of operations executed on GPU.
    public let gpuOperations: Int

    /// Total time spent on GPU operations.
    public let gpuTimeTotal: TimeInterval

    /// Average time per GPU operation.
    public var averageGPUTime: TimeInterval {
        guard gpuOperations > 0 else { return 0 }
        return gpuTimeTotal / Double(gpuOperations)
    }

    public init(
        gpuOperations: Int = 0,
        gpuTimeTotal: TimeInterval = 0
    ) {
        self.gpuOperations = gpuOperations
        self.gpuTimeTotal = gpuTimeTotal
    }
}

/// Internal mutable statistics for tracking.
private struct MutableStatistics {
    var gpuOperations: Int = 0
    var gpuTimeTotal: TimeInterval = 0
}

// MARK: - Errors

/// Errors from acceleration operations.
public enum AccelerationError: Error, LocalizedError, Sendable {
    case gpuNotAvailable
    case gpuOperationFailed(String)
    case dimensionMismatch(expected: Int, got: Int)
    case invalidInput(String)
    case initializationFailed(Error)

    public var errorDescription: String? {
        switch self {
        case .gpuNotAvailable:
            return "GPU acceleration is not available on this device"
        case .gpuOperationFailed(let reason):
            return "GPU operation failed: \(reason)"
        case .dimensionMismatch(let expected, let got):
            return "Vector dimension mismatch: expected \(expected), got \(got)"
        case .invalidInput(let reason):
            return "Invalid input: \(reason)"
        case .initializationFailed(let error):
            return "Failed to initialize GPU acceleration: \(error.localizedDescription)"
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .gpuNotAvailable:
            return "GPU acceleration requires Metal 4 support (Apple Silicon)."
        case .gpuOperationFailed:
            return "Try reducing batch size or input dimensions."
        case .dimensionMismatch:
            return "Ensure all vectors have the same dimensions."
        case .invalidInput:
            return "Verify input data is valid (not empty, no NaN/Inf)."
        case .initializationFailed:
            return "Ensure Metal 4 is available on this device."
        }
    }
}

// MARK: - Shared Instance Helper

/// Actor for thread-safe singleton management.
private actor SharedAccelerationManager {
    static let holder = SharedAccelerationManager()
    private var manager: AccelerationManager?
    private var initError: Error?

    static var instance: AccelerationManager {
        get async throws {
            if let existing = await holder.manager {
                return existing
            }
            if let error = await holder.initError {
                throw error
            }

            do {
                let new = try await AccelerationManager()
                await holder.setManager(new)
                return new
            } catch {
                await holder.setError(error)
                throw error
            }
        }
    }

    private func setManager(_ manager: AccelerationManager) {
        self.manager = manager
    }

    private func setError(_ error: Error) {
        self.initError = error
    }
}
