// EmbedKit - Acceleration Manager
// GPU-accelerated distance and normalization operations using Metal4

import Foundation
import VectorCore
import VectorAccelerate

// MARK: - GPU Health Preset

/// Configuration preset for GPU health monitoring.
///
/// These presets control how aggressively the system disables GPU operations
/// after failures and how quickly it attempts recovery.
public enum GPUHealthPreset: Sendable {
    /// Balanced preset - 3 failures before disable, 5 min recovery.
    /// Suitable for most production workloads.
    case `default`

    /// Aggressive preset - 2 failures before disable, 10 min recovery.
    /// Use for critical workloads where GPU stability is paramount.
    case aggressive

    /// Lenient preset - 5 failures before disable, 3 min recovery.
    /// Use for development or when GPU failures are expected to be transient.
    case lenient

    /// Convert to VectorAccelerate's configuration.
    internal func toConfiguration() -> GPUHealthMonitorConfiguration {
        switch self {
        case .default:
            return .default
        case .aggressive:
            return .aggressive
        case .lenient:
            return .lenient
        }
    }
}

// MARK: - GPU Decision Profile

/// Configuration profile for GPU/CPU routing decisions.
///
/// These profiles control when operations are routed to GPU vs CPU based on
/// workload characteristics like vector count, dimension, and batch size.
public enum GPUDecisionProfile: Sendable {
    /// Balanced profile - suitable for most workloads.
    /// Uses default thresholds that balance GPU/CPU based on operation complexity.
    case balanced

    /// Batch-optimized profile - lower thresholds for batch workloads.
    /// Routes more operations to GPU, assuming batch processing benefits from parallelism.
    case batchOptimized

    /// Real-time optimized profile - higher thresholds for latency-sensitive operations.
    /// More conservative GPU routing to avoid latency spikes from GPU overhead.
    case realTimeOptimized

    /// Always use GPU (current behavior).
    /// Bypasses decision engine and always routes to GPU (health permitting).
    case alwaysGPU

    /// Always use CPU.
    /// Forces CPU execution, useful for debugging or when GPU is unreliable.
    case alwaysCPU

    /// Convert to VectorAccelerate's activation thresholds.
    internal func toThresholds() -> GPUActivationThresholds? {
        switch self {
        case .balanced:
            return GPUActivationThresholds()
        case .batchOptimized:
            return .batchOptimized
        case .realTimeOptimized:
            return .realTimeOptimized
        case .alwaysGPU, .alwaysCPU:
            return nil // No thresholds needed
        }
    }

    /// Whether this profile always uses GPU (bypasses decision engine).
    internal var alwaysUsesGPU: Bool {
        self == .alwaysGPU
    }

    /// Whether this profile always uses CPU (bypasses decision engine).
    internal var alwaysUsesCPU: Bool {
        self == .alwaysCPU
    }
}

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

    /// GPU health monitor for fault tolerance.
    private let healthMonitor: GPUHealthMonitor

    /// GPU decision engine for adaptive GPU/CPU routing.
    /// Nil when using `.alwaysGPU` or `.alwaysCPU` profiles.
    private let decisionEngine: GPUDecisionEngine?

    /// Current decision profile for GPU/CPU routing.
    private var decisionProfile: GPUDecisionProfile

    /// Statistics tracking.
    private var stats: MutableStatistics

    // MARK: - Initialization

    /// Create an acceleration manager with a new Metal4Context.
    ///
    /// - Parameters:
    ///   - healthPreset: Health monitoring configuration preset (default: `.default`)
    ///   - decisionProfile: GPU/CPU routing decision profile (default: `.alwaysGPU`)
    public init(
        healthPreset: GPUHealthPreset = .default,
        decisionProfile: GPUDecisionProfile = .alwaysGPU
    ) async throws {
        self.context = try await Metal4ContextManager.shared()
        self.distanceProvider = try await UniversalKernelDistanceProvider(context: context)
        self.normalizationKernel = try await L2NormalizationKernel(context: context)
        self.umapKernel = try await UMAPGradientKernel(context: context)
        self.healthMonitor = GPUHealthMonitor(configuration: healthPreset.toConfiguration())
        self.decisionProfile = decisionProfile

        // Initialize decision engine only for adaptive profiles
        if let thresholds = decisionProfile.toThresholds() {
            self.decisionEngine = await GPUDecisionEngine(context: context, thresholds: thresholds)
        } else {
            self.decisionEngine = nil
        }

        self.stats = MutableStatistics()
    }

    /// Create an acceleration manager with a shared Metal4Context.
    ///
    /// - Parameters:
    ///   - context: Shared Metal4Context for GPU operations
    ///   - healthPreset: Health monitoring configuration preset (default: `.default`)
    ///   - decisionProfile: GPU/CPU routing decision profile (default: `.alwaysGPU`)
    public init(
        context: Metal4Context,
        healthPreset: GPUHealthPreset = .default,
        decisionProfile: GPUDecisionProfile = .alwaysGPU
    ) async throws {
        self.context = context
        self.distanceProvider = try await UniversalKernelDistanceProvider(context: context)
        self.normalizationKernel = try await L2NormalizationKernel(context: context)
        self.umapKernel = try await UMAPGradientKernel(context: context)
        self.healthMonitor = GPUHealthMonitor(configuration: healthPreset.toConfiguration())
        self.decisionProfile = decisionProfile

        // Initialize decision engine only for adaptive profiles
        if let thresholds = decisionProfile.toThresholds() {
            self.decisionEngine = await GPUDecisionEngine(context: context, thresholds: thresholds)
        } else {
            self.decisionEngine = nil
        }

        self.stats = MutableStatistics()
    }

    /// Factory method for creating an acceleration manager.
    ///
    /// - Parameters:
    ///   - healthPreset: Health monitoring configuration preset (default: `.default`)
    ///   - decisionProfile: GPU/CPU routing decision profile (default: `.alwaysGPU`)
    public static func create(
        healthPreset: GPUHealthPreset = .default,
        decisionProfile: GPUDecisionProfile = .alwaysGPU
    ) async throws -> AccelerationManager {
        try await AccelerationManager(healthPreset: healthPreset, decisionProfile: decisionProfile)
    }

    /// Factory method with shared context.
    ///
    /// - Parameters:
    ///   - context: Shared Metal4Context for GPU operations
    ///   - healthPreset: Health monitoring configuration preset (default: `.default`)
    ///   - decisionProfile: GPU/CPU routing decision profile (default: `.alwaysGPU`)
    public static func create(
        context: Metal4Context,
        healthPreset: GPUHealthPreset = .default,
        decisionProfile: GPUDecisionProfile = .alwaysGPU
    ) async throws -> AccelerationManager {
        try await AccelerationManager(context: context, healthPreset: healthPreset, decisionProfile: decisionProfile)
    }

    // MARK: - Distance Operations

    /// Compute batch distances from a query to multiple candidates.
    ///
    /// Uses GPU-accelerated distance computation via Metal4 kernels with
    /// automatic CPU fallback when GPU health is degraded or when the
    /// decision engine determines CPU is more efficient.
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

        let healthOperation = "batch_distance"
        let gpuOp = gpuOperation(for: metric)
        let candidateCount = candidates.count

        // Step 1: Check if we should fallback to CPU due to health
        if await healthMonitor.shouldFallbackToCPU(operation: healthOperation) {
            return cpuBatchDistance(from: query, to: candidates, metric: metric)
        }

        // Step 2: Check if decision engine recommends CPU
        let useGPU = await shouldUseGPU(
            operation: gpuOp,
            candidateCount: candidateCount,
            dimension: dimension
        )

        if !useGPU {
            return cpuBatchDistance(from: query, to: candidates, metric: metric)
        }

        // Step 3: Execute GPU operation
        let start = CFAbsoluteTimeGetCurrent()

        do {
            // Use VectorAccelerate's UniversalKernelDistanceProvider
            let queryVec = DynamicVector(query)
            let candidateVecs = candidates.map { DynamicVector($0) }

            let distances = try await distanceProvider.batchDistance(
                from: queryVec,
                to: candidateVecs,
                metric: metric
            )

            let gpuTime = CFAbsoluteTimeGetCurrent() - start

            await healthMonitor.recordSuccess(operation: healthOperation)
            stats.gpuOperations += 1
            stats.gpuTimeTotal += gpuTime

            // Step 4: Record performance for adaptive learning
            let estimatedCPU = estimateCPUTime(candidateCount: candidateCount, dimension: dimension)
            await recordPerformance(operation: gpuOp, cpuTime: estimatedCPU, gpuTime: gpuTime)

            return distances
        } catch {
            await healthMonitor.recordFailure(operation: healthOperation, error: error)
            // Fallback to CPU on failure
            return cpuBatchDistance(from: query, to: candidates, metric: metric)
        }
    }

    /// CPU fallback for batch distance computation.
    private nonisolated func cpuBatchDistance(
        from query: [Float],
        to candidates: [[Float]],
        metric: SupportedDistanceMetric
    ) -> [Float] {
        switch metric {
        case .cosine:
            return AccelerateBLAS.batchCosineDistance(query: query, candidates: candidates)
        case .euclidean:
            return AccelerateBLAS.batchEuclideanDistance(query: query, candidates: candidates)
        case .dotProduct:
            return candidates.map { -AccelerateBLAS.dotProduct(query, $0) }
        case .manhattan:
            return AccelerateBLAS.batchManhattanDistance(query: query, candidates: candidates)
        case .chebyshev:
            return AccelerateBLAS.batchChebyshevDistance(query: query, candidates: candidates)
        }
    }

    /// Compute distance between two vectors.
    ///
    /// Uses GPU-accelerated distance computation with automatic CPU fallback
    /// when GPU health is degraded or when the decision engine determines
    /// CPU is more efficient (typical for single-vector operations).
    public func distance(
        from a: [Float],
        to b: [Float],
        metric: SupportedDistanceMetric
    ) async throws -> Float {
        guard a.count == b.count else {
            throw AccelerationError.dimensionMismatch(expected: a.count, got: b.count)
        }

        let healthOperation = "distance"
        let gpuOp = gpuOperation(for: metric)
        let dimension = a.count

        // Step 1: Check if we should fallback to CPU due to health
        if await healthMonitor.shouldFallbackToCPU(operation: healthOperation) {
            return cpuDistance(from: a, to: b, metric: metric)
        }

        // Step 2: Check if decision engine recommends CPU
        // Single-vector distance typically benefits from CPU
        let useGPU = await shouldUseGPU(
            operation: gpuOp,
            candidateCount: 1,
            dimension: dimension
        )

        if !useGPU {
            return cpuDistance(from: a, to: b, metric: metric)
        }

        // Step 3: Execute GPU operation
        let start = CFAbsoluteTimeGetCurrent()

        do {
            let vecA = DynamicVector(a)
            let vecB = DynamicVector(b)

            let result = try await distanceProvider.distance(from: vecA, to: vecB, metric: metric)

            let gpuTime = CFAbsoluteTimeGetCurrent() - start

            await healthMonitor.recordSuccess(operation: healthOperation)
            stats.gpuOperations += 1
            stats.gpuTimeTotal += gpuTime

            // Step 4: Record performance for adaptive learning
            let estimatedCPU = estimateCPUTime(candidateCount: 1, dimension: dimension)
            await recordPerformance(operation: gpuOp, cpuTime: estimatedCPU, gpuTime: gpuTime)

            return result
        } catch {
            await healthMonitor.recordFailure(operation: healthOperation, error: error)
            return cpuDistance(from: a, to: b, metric: metric)
        }
    }

    /// CPU fallback for single distance computation.
    private nonisolated func cpuDistance(
        from a: [Float],
        to b: [Float],
        metric: SupportedDistanceMetric
    ) -> Float {
        switch metric {
        case .cosine:
            return AccelerateBLAS.cosineDistance(a, b)
        case .euclidean:
            return AccelerateBLAS.euclideanDistance(a, b)
        case .dotProduct:
            return -AccelerateBLAS.dotProduct(a, b)
        case .manhattan:
            return AccelerateBLAS.manhattanDistance(a, b)
        case .chebyshev:
            return AccelerateBLAS.chebyshevDistance(a, b)
        }
    }

    // MARK: - Vector Operations

    /// Normalize a batch of vectors using GPU.
    ///
    /// Uses GPU-accelerated normalization with automatic CPU fallback
    /// when GPU health is degraded or when the decision engine determines
    /// CPU is more efficient.
    public func normalizeBatch(_ vectors: [[Float]]) async throws -> [[Float]] {
        guard !vectors.isEmpty else { return [] }

        let healthOperation = "normalize_batch"
        let vectorCount = vectors.count
        let dimension = vectors[0].count

        // Step 1: Check if we should fallback to CPU due to health
        if await healthMonitor.shouldFallbackToCPU(operation: healthOperation) {
            return cpuNormalizeBatch(vectors)
        }

        // Step 2: Check if decision engine recommends CPU
        let useGPU = await shouldUseGPU(
            operation: .normalization,
            vectorCount: vectorCount,
            candidateCount: vectorCount,
            dimension: dimension
        )

        if !useGPU {
            return cpuNormalizeBatch(vectors)
        }

        // Step 3: Execute GPU operation
        let start = CFAbsoluteTimeGetCurrent()

        do {
            let result = try await normalizationKernel.normalize(vectors)
            let normalized = result.asArrays()

            let gpuTime = CFAbsoluteTimeGetCurrent() - start

            await healthMonitor.recordSuccess(operation: healthOperation)
            stats.gpuOperations += 1
            stats.gpuTimeTotal += gpuTime

            // Step 4: Record performance for adaptive learning
            let estimatedCPU = estimateCPUTime(candidateCount: vectorCount, dimension: dimension)
            await recordPerformance(operation: .normalization, cpuTime: estimatedCPU, gpuTime: gpuTime)

            return normalized
        } catch {
            await healthMonitor.recordFailure(operation: healthOperation, error: error)
            return cpuNormalizeBatch(vectors)
        }
    }

    /// CPU fallback for batch normalization.
    private nonisolated func cpuNormalizeBatch(_ vectors: [[Float]]) -> [[Float]] {
        vectors.map { AccelerateBLAS.normalize($0) }
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
    /// - Throws: `AccelerationError` if projection fails or GPU is unhealthy
    ///
    /// - Note: UMAP requires GPU acceleration. If GPU health is degraded,
    ///   this operation will fail with `gpuNotAvailable` error.
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

        let operation = "umap_project"

        // UMAP requires GPU - check health first
        if await healthMonitor.shouldFallbackToCPU(operation: operation) {
            throw AccelerationError.gpuNotAvailable
        }

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

        do {
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

            await healthMonitor.recordSuccess(operation: operation)
            stats.gpuOperations += 1
            stats.gpuTimeTotal += CFAbsoluteTimeGetCurrent() - start

            return lowDimEmbedding
        } catch {
            await healthMonitor.recordFailure(operation: operation, error: error)
            throw error
        }
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
    /// - Throws: `AccelerationError` if GPU is unhealthy (no CPU fallback)
    public func umapGradient(
        embeddings: [[Float]],
        lowDimEmbeddings: [[Float]],
        neighbors: Int = 15
    ) async throws -> [[Float]] {
        guard !embeddings.isEmpty else { return [] }

        let operation = "umap_gradient"

        // UMAP gradient requires GPU
        if await healthMonitor.shouldFallbackToCPU(operation: operation) {
            throw AccelerationError.gpuNotAvailable
        }

        guard embeddings.count == lowDimEmbeddings.count else {
            throw AccelerationError.dimensionMismatch(
                expected: embeddings.count,
                got: lowDimEmbeddings.count
            )
        }

        let n = embeddings.count
        let d = lowDimEmbeddings[0].count

        do {
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

            await healthMonitor.recordSuccess(operation: operation)
            return gradients
        } catch {
            await healthMonitor.recordFailure(operation: operation, error: error)
            throw error
        }
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

    // MARK: - GPU Decision Helpers

    /// Map SupportedDistanceMetric to GPUOperation for decision engine.
    private nonisolated func gpuOperation(for metric: SupportedDistanceMetric) -> GPUOperation {
        switch metric {
        case .cosine:
            return .cosineSimilarity
        case .euclidean:
            return .l2Distance
        case .dotProduct:
            return .dotProduct
        case .manhattan:
            return .manhattanDistance
        case .chebyshev:
            return .l2Distance // Chebyshev uses same routing as L2
        }
    }

    /// Estimate CPU time for distance computation based on workload.
    ///
    /// Uses simple heuristic: time ≈ dimension × candidateCount × constant
    /// The constant is calibrated to ~50ns per float operation.
    private nonisolated func estimateCPUTime(
        candidateCount: Int,
        dimension: Int
    ) -> TimeInterval {
        let operationsPerCandidate = dimension * 3 // multiply, subtract, accumulate
        let totalOperations = candidateCount * operationsPerCandidate
        let nsPerOperation: Double = 50 // ~50ns per SIMD operation
        return Double(totalOperations) * nsPerOperation / 1_000_000_000
    }

    /// Determine whether GPU should be used based on decision profile and engine.
    ///
    /// Decision flow:
    /// 1. If `.alwaysCPU` profile, return false
    /// 2. If `.alwaysGPU` profile, return true
    /// 3. If adaptive profile, consult decision engine
    private func shouldUseGPU(
        operation: GPUOperation,
        vectorCount: Int = 1,
        candidateCount: Int,
        k: Int = 1,
        dimension: Int
    ) async -> Bool {
        // Short-circuit for forced profiles
        if decisionProfile.alwaysUsesCPU {
            return false
        }
        if decisionProfile.alwaysUsesGPU {
            return true
        }

        // Consult decision engine for adaptive profiles
        guard let engine = decisionEngine else {
            return true // Fallback to GPU if engine not available
        }

        return await engine.shouldUseGPU(
            operation: operation,
            vectorCount: vectorCount,
            candidateCount: candidateCount,
            k: k,
            dimension: dimension
        )
    }

    /// Record performance for adaptive learning.
    private func recordPerformance(
        operation: GPUOperation,
        cpuTime: TimeInterval,
        gpuTime: TimeInterval
    ) async {
        guard let engine = decisionEngine else { return }
        await engine.recordPerformance(
            operation: operation,
            cpuTime: cpuTime,
            gpuTime: gpuTime
        )
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

    // MARK: - GPU Health Monitoring

    /// Get current GPU health status.
    ///
    /// Returns detailed information about GPU health including failure counts,
    /// degradation levels, and disabled operations.
    ///
    /// - Returns: Complete health status snapshot
    public func gpuHealthStatus() async -> GPUHealthStatus {
        await healthMonitor.getHealthStatus()
    }

    /// Check if GPU is currently healthy.
    ///
    /// Returns `true` if no operations have degraded health status.
    /// Use this for quick health checks before critical operations.
    ///
    /// - Returns: `true` if GPU is healthy, `false` if any degradation detected
    public func isGPUHealthy() async -> Bool {
        await healthMonitor.isHealthy()
    }

    /// Reset health tracking state.
    ///
    /// Clears all failure counts, re-enables disabled operations, and
    /// resets degradation levels. Use after hardware changes or when
    /// you want to retry GPU operations after fixing issues.
    public func resetHealthTracking() async {
        await healthMonitor.reset()
    }

    /// Check if fallback to CPU is recommended for a specific operation.
    ///
    /// This is useful for callers who want to check health before attempting
    /// GPU operations, rather than relying on automatic fallback.
    ///
    /// - Parameter operation: The operation identifier to check
    /// - Returns: `true` if CPU fallback is recommended
    public func isFallbackRecommended(for operation: String) async -> Bool {
        await healthMonitor.isFallbackRecommended(operation: operation)
    }

    /// Get degradation level for a specific operation.
    ///
    /// - Parameter operation: The operation identifier
    /// - Returns: Current degradation level for the operation
    public func degradationLevel(for operation: String) async -> GPUDegradationLevel {
        await healthMonitor.getDegradationLevel(for: operation)
    }

    /// Re-export GPUHealthStatus from VectorAccelerate for convenience.
    public typealias HealthStatus = GPUHealthStatus

    /// Re-export GPUDegradationLevel from VectorAccelerate for convenience.
    public typealias DegradationLevel = GPUDegradationLevel

    // MARK: - GPU Decision Engine

    /// Get current GPU/CPU routing decision profile.
    ///
    /// - Returns: The current decision profile
    public func currentDecisionProfile() -> GPUDecisionProfile {
        decisionProfile
    }

    /// Update the GPU/CPU routing decision profile.
    ///
    /// Changing the profile affects how future operations are routed.
    /// Note: This does not re-initialize the decision engine. For full
    /// reconfiguration, create a new AccelerationManager instance.
    ///
    /// - Parameter profile: New decision profile to use
    public func updateDecisionProfile(_ profile: GPUDecisionProfile) async {
        self.decisionProfile = profile

        // Update decision engine thresholds if using an adaptive profile
        if let thresholds = profile.toThresholds(), let engine = decisionEngine {
            await engine.updateThresholds(thresholds)
        }
    }

    /// Get GPU performance statistics from the decision engine.
    ///
    /// Returns aggregated performance data for each operation type,
    /// including speedup ratios and average times. This data is used
    /// for adaptive GPU/CPU routing decisions.
    ///
    /// - Returns: Performance statistics, or nil if using a non-adaptive profile
    public func gpuPerformanceStats() async -> GPUPerformanceStats? {
        guard let engine = decisionEngine else { return nil }
        return await engine.getPerformanceStats()
    }

    /// Get current GPU activation thresholds.
    ///
    /// These thresholds determine when operations are routed to GPU vs CPU.
    ///
    /// - Returns: Current thresholds, or nil if using a non-adaptive profile
    public func currentActivationThresholds() async -> GPUActivationThresholds? {
        guard let engine = decisionEngine else { return nil }
        return await engine.getThresholds()
    }

    /// Reset the decision engine's performance history.
    ///
    /// Clears all recorded performance data and resets adaptive ratios.
    /// Use this after hardware changes or when performance characteristics change.
    public func resetDecisionEngineHistory() async {
        await decisionEngine?.reset()
    }

    /// Re-export GPUPerformanceStats from VectorAccelerate for convenience.
    public typealias PerformanceStats = GPUPerformanceStats

    /// Re-export GPUActivationThresholds from VectorAccelerate for convenience.
    public typealias ActivationThresholds = GPUActivationThresholds

    /// Re-export GPUOperation from VectorAccelerate for convenience.
    public typealias Operation = GPUOperation
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
