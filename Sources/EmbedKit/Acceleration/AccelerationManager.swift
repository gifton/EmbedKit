// EmbedKit - Acceleration Manager
// Manages GPU/CPU compute path selection and execution

import Foundation
import VectorCore
import VectorAccelerate
import VectorIndex

// MARK: - Acceleration Manager

/// Actor that manages GPU acceleration with automatic CPU fallback.
///
/// The AccelerationManager provides transparent GPU acceleration for vector operations.
/// It automatically selects the optimal compute path based on operation characteristics
/// and falls back to CPU when GPU is unavailable or when operations are too small
/// to benefit from GPU parallelism.
///
/// ## Example Usage
/// ```swift
/// let manager = try await AccelerationManager.create()
///
/// // Compute batch distances - automatically uses GPU if beneficial
/// let distances = try await manager.batchDistance(
///     from: queryVector,
///     to: candidateVectors,
///     metric: .cosine
/// )
/// ```
public actor AccelerationManager {

    // MARK: - Shared Instance

    /// Shared instance for convenience.
    /// Lazily initialized on first access.
    public static func shared() async -> AccelerationManager {
        await SharedAccelerationManager.instance
    }

    // MARK: - Properties

    /// Current compute preference.
    public private(set) var preference: ComputePreference

    /// Thresholds for GPU usage decisions.
    public private(set) var thresholds: AccelerationThresholds

    /// Whether GPU is available on this device.
    public let isGPUAvailable: Bool

    /// The Metal context for GPU operations (nil if GPU unavailable).
    private var metalContext: MetalContext?

    /// Compute engine for GPU operations.
    private var computeEngine: ComputeEngine?

    /// Statistics tracking.
    private var stats: MutableStatistics

    /// Whether GPU is currently disabled due to errors.
    private var gpuTemporarilyDisabled: Bool = false

    /// Count of consecutive GPU errors.
    private var consecutiveGPUErrors: Int = 0

    /// Maximum consecutive errors before temporarily disabling GPU.
    private let maxConsecutiveErrors = 3

    // MARK: - Initialization

    /// Create an acceleration manager with specified preferences.
    /// - Parameters:
    ///   - preference: Compute preference (default: .auto)
    ///   - thresholds: Thresholds for GPU usage decisions
    public init(
        preference: ComputePreference = .auto,
        thresholds: AccelerationThresholds = .default
    ) async {
        self.preference = preference
        self.thresholds = thresholds
        self.stats = MutableStatistics()

        // Check GPU availability
        self.isGPUAvailable = MetalContext.isAvailable

        // Initialize GPU resources if available and not CPU-only
        if isGPUAvailable && preference != .cpuOnly {
            do {
                let context = try await MetalContext()
                self.metalContext = context
                self.computeEngine = try await ComputeEngine(context: context)
            } catch {
                // GPU initialization failed - continue with CPU only
                self.metalContext = nil
                self.computeEngine = nil
            }
        }
    }

    /// Factory method for creating an acceleration manager.
    public static func create(
        preference: ComputePreference = .auto,
        thresholds: AccelerationThresholds = .default
    ) async -> AccelerationManager {
        await AccelerationManager(preference: preference, thresholds: thresholds)
    }

    // MARK: - Configuration

    /// Update the compute preference.
    public func setPreference(_ preference: ComputePreference) async {
        self.preference = preference

        // Re-initialize GPU if switching from cpuOnly to auto/gpuOnly
        if preference != .cpuOnly && computeEngine == nil && isGPUAvailable {
            do {
                let context = try await MetalContext()
                self.metalContext = context
                self.computeEngine = try await ComputeEngine(context: context)
                self.gpuTemporarilyDisabled = false
                self.consecutiveGPUErrors = 0
            } catch {
                // Failed to initialize GPU
            }
        }
    }

    /// Update acceleration thresholds.
    public func setThresholds(_ thresholds: AccelerationThresholds) {
        self.thresholds = thresholds
    }

    // MARK: - Distance Operations

    /// Compute batch distances from a query to multiple candidates.
    ///
    /// Automatically selects GPU or CPU based on:
    /// - Compute preference setting
    /// - Number of candidates
    /// - Vector dimension
    /// - GPU availability
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
                    actual: candidate.count
                )
            }
        }

        // Decide compute path
        let useGPU = shouldUseGPU(
            candidateCount: candidates.count,
            dimension: dimension
        )

        if useGPU {
            return try await gpuBatchDistance(query: query, candidates: candidates, metric: metric)
        } else {
            return try await cpuBatchDistance(query: query, candidates: candidates, metric: metric)
        }
    }

    /// Compute distance between two vectors.
    /// Note: Single distance computation almost always uses CPU (GPU overhead not worth it).
    public func distance(
        from a: [Float],
        to b: [Float],
        metric: SupportedDistanceMetric
    ) async throws -> Float {
        guard a.count == b.count else {
            throw AccelerationError.dimensionMismatch(expected: a.count, actual: b.count)
        }

        // Single distance always uses CPU (GPU overhead not worth it)
        let start = CFAbsoluteTimeGetCurrent()
        let result = cpuDistance(a, b, metric: metric)
        stats.cpuOperations += 1
        stats.cpuTimeTotal += CFAbsoluteTimeGetCurrent() - start

        return result
    }

    // MARK: - Vector Operations

    /// Normalize a batch of vectors.
    public func normalizeBatch(_ vectors: [[Float]]) async throws -> [[Float]] {
        guard !vectors.isEmpty else { return [] }

        let useGPU = preference != .cpuOnly &&
                     !gpuTemporarilyDisabled &&
                     vectors.count >= thresholds.minBatchForNormalization &&
                     computeEngine != nil

        if useGPU {
            do {
                let start = CFAbsoluteTimeGetCurrent()
                var results: [[Float]] = []
                results.reserveCapacity(vectors.count)

                for vector in vectors {
                    let normalized = try await computeEngine!.normalize(vector)
                    results.append(normalized)
                }

                stats.gpuOperations += 1
                stats.gpuTimeTotal += CFAbsoluteTimeGetCurrent() - start
                consecutiveGPUErrors = 0

                return results
            } catch {
                handleGPUError()
                // Fall through to CPU
            }
        }

        // CPU fallback
        let start = CFAbsoluteTimeGetCurrent()
        let results = vectors.map { normalize($0) }
        stats.cpuOperations += 1
        stats.cpuTimeTotal += CFAbsoluteTimeGetCurrent() - start

        return results
    }

    /// Normalize a single vector (always CPU - too small for GPU benefit).
    public func normalize(_ vector: [Float]) -> [Float] {
        let magnitude = sqrt(vector.reduce(0) { $0 + $1 * $1 })
        guard magnitude > 0 else { return vector }
        return vector.map { $0 / magnitude }
    }

    // MARK: - Statistics

    /// Get current acceleration statistics.
    public func statistics() -> AccelerationStatistics {
        AccelerationStatistics(
            gpuOperations: stats.gpuOperations,
            cpuOperations: stats.cpuOperations,
            gpuTimeTotal: stats.gpuTimeTotal,
            cpuTimeTotal: stats.cpuTimeTotal,
            gpuFallbacks: stats.gpuFallbacks
        )
    }

    /// Reset statistics.
    public func resetStatistics() {
        stats = MutableStatistics()
    }

    // MARK: - Private Helpers

    /// Determine if GPU should be used for an operation.
    private func shouldUseGPU(candidateCount: Int, dimension: Int) -> Bool {
        switch preference {
        case .cpuOnly:
            return false
        case .gpuOnly:
            return computeEngine != nil && !gpuTemporarilyDisabled
        case .auto:
            guard computeEngine != nil && !gpuTemporarilyDisabled else {
                return false
            }
            return candidateCount >= thresholds.minCandidatesForGPU &&
                   dimension >= thresholds.minDimensionForGPU
        }
    }

    /// GPU batch distance computation.
    private func gpuBatchDistance(
        query: [Float],
        candidates: [[Float]],
        metric: SupportedDistanceMetric
    ) async throws -> [Float] {
        guard let engine = computeEngine else {
            throw AccelerationError.gpuNotAvailable
        }

        do {
            let start = CFAbsoluteTimeGetCurrent()

            // Use VectorAccelerate's ComputeEngine batch distance
            let distances = try await engine.batchDistance(
                query: query,
                candidates: candidates,
                metric: metric
            )

            stats.gpuOperations += 1
            stats.gpuTimeTotal += CFAbsoluteTimeGetCurrent() - start
            consecutiveGPUErrors = 0

            return distances
        } catch {
            handleGPUError()

            // Fallback to CPU
            if preference != .gpuOnly {
                return try await cpuBatchDistance(query: query, candidates: candidates, metric: metric)
            } else {
                throw AccelerationError.gpuOperationFailed(error.localizedDescription)
            }
        }
    }

    /// CPU batch distance computation with automatic 384-dimension optimization.
    ///
    /// For 384-dimensional vectors (MiniLM), uses VectorCore's `Vector384Optimized`
    /// for 2-3x faster batch computation. Falls back to generic vDSP for other dimensions.
    private func cpuBatchDistance(
        query: [Float],
        candidates: [[Float]],
        metric: SupportedDistanceMetric
    ) async throws -> [Float] {
        let start = CFAbsoluteTimeGetCurrent()

        // AccelerateBLAS methods auto-detect 384-dim and use Vector384Optimized
        let distances: [Float]
        switch metric {
        case .cosine:
            distances = AccelerateBLAS.batchCosineDistance(query: query, candidates: candidates)
        case .euclidean:
            distances = AccelerateBLAS.batchEuclideanDistance(query: query, candidates: candidates)
        case .dotProduct:
            distances = candidates.map { -AccelerateBLAS.dotProduct(query, $0) }
        case .manhattan:
            distances = AccelerateBLAS.batchManhattanDistance(query: query, candidates: candidates)
        case .chebyshev:
            distances = AccelerateBLAS.batchChebyshevDistance(query: query, candidates: candidates)
        }

        stats.cpuOperations += 1
        stats.cpuTimeTotal += CFAbsoluteTimeGetCurrent() - start

        return distances
    }

    /// CPU single distance computation using Accelerate BLAS.
    ///
    /// For 384-dimensional vectors (MiniLM), AccelerateBLAS auto-uses Vector384Optimized.
    private func cpuDistance(_ a: [Float], _ b: [Float], metric: SupportedDistanceMetric) -> Float {
        switch metric {
        case .cosine:
            return AccelerateBLAS.cosineDistance(a, b)

        case .euclidean:
            return AccelerateBLAS.euclideanDistance(a, b)

        case .dotProduct:
            // For dot product, higher is more similar, so negate for distance-like ordering
            return -AccelerateBLAS.dotProduct(a, b)

        case .manhattan:
            return AccelerateBLAS.manhattanDistance(a, b)

        case .chebyshev:
            return AccelerateBLAS.chebyshevDistance(a, b)
        }
    }

    /// Handle GPU error - track and potentially disable.
    private func handleGPUError() {
        consecutiveGPUErrors += 1
        stats.gpuFallbacks += 1

        if consecutiveGPUErrors >= maxConsecutiveErrors {
            gpuTemporarilyDisabled = true
        }
    }

    /// Re-enable GPU after temporary disable.
    public func reEnableGPU() {
        gpuTemporarilyDisabled = false
        consecutiveGPUErrors = 0
    }

    // MARK: - AccelerableIndex Integration

    /// Perform accelerated search using an AccelerableIndex.
    ///
    /// This method provides hybrid CPU+GPU search by:
    /// 1. Using the index's `shouldAccelerate()` to decide the compute path
    /// 2. Getting candidates via `getCandidates()` (index handles traversal)
    /// 3. Computing distances on GPU (if beneficial) or CPU
    /// 4. Finalizing results via `finalizeResults()` (index handles ID mapping)
    ///
    /// - Parameters:
    ///   - index: An AccelerableIndex (FlatIndex, HNSWIndex, IVFIndex all conform)
    ///   - query: Query vector
    ///   - k: Number of results to return
    ///   - filter: Optional metadata filter
    /// - Returns: Search results with IDs and distances
    public func acceleratedSearch(
        index: any AccelerableIndex,
        query: [Float],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil
    ) async throws -> [VectorIndex.SearchResult] {
        // Check if acceleration is worthwhile
        let candidateEstimate = await index.count
        let shouldAccelerate = await index.shouldAccelerate(
            queryCount: 1,
            candidateCount: candidateEstimate,
            k: k
        )

        // If acceleration not beneficial, use index's native search
        guard shouldAccelerate && preference != .cpuOnly else {
            return try await index.search(query: query, k: k, filter: filter)
        }

        // Get candidates from index (handles HNSW traversal, IVF cluster selection, etc.)
        let candidates = try await index.getCandidates(query: query, k: k, filter: filter)

        // Compute distances - GPU or CPU based on thresholds
        let distances = try await computeDistancesForCandidates(
            query: query,
            candidates: candidates,
            metric: await index.metric
        )

        // Select top-k from computed distances
        let topK = selectTopK(k: k, from: distances)

        // Let index finalize results (handles ID mapping, metadata, etc.)
        let acceleratedResults = AcceleratedResults(
            indices: topK.indices,
            distances: topK.distances
        )

        return await index.finalizeResults(
            candidates: candidates,
            results: acceleratedResults,
            filter: filter
        )
    }

    /// Perform accelerated batch search using an AccelerableIndex.
    ///
    /// Processes multiple queries efficiently using the index's batch candidate retrieval.
    ///
    /// - Parameters:
    ///   - index: An AccelerableIndex
    ///   - queries: Array of query vectors
    ///   - k: Number of results per query
    ///   - filter: Optional metadata filter
    /// - Returns: Array of search results, one per query
    public func acceleratedBatchSearch(
        index: any AccelerableIndex,
        queries: [[Float]],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil
    ) async throws -> [[VectorIndex.SearchResult]] {
        guard !queries.isEmpty else { return [] }

        // Check if acceleration is worthwhile
        let candidateEstimate = await index.count
        let shouldAccelerate = await index.shouldAccelerate(
            queryCount: queries.count,
            candidateCount: candidateEstimate,
            k: k
        )

        // If acceleration not beneficial, use index's native batch search
        guard shouldAccelerate && preference != .cpuOnly else {
            return try await index.batchSearch(queries: queries, k: k, filter: filter)
        }

        // Get batch candidates from index
        let batchCandidates = try await index.getBatchCandidates(
            queries: queries,
            k: k,
            filter: filter
        )

        let metric = await index.metric

        // Process each query's candidates
        var batchResults: [AcceleratedResults] = []
        batchResults.reserveCapacity(queries.count)

        for (query, candidates) in zip(queries, batchCandidates) {
            let distances = try await computeDistancesForCandidates(
                query: query,
                candidates: candidates,
                metric: metric
            )
            let topK = selectTopK(k: k, from: distances)
            batchResults.append(AcceleratedResults(
                indices: topK.indices,
                distances: topK.distances
            ))
        }

        // Finalize all results
        return await index.finalizeBatchResults(
            batchCandidates: batchCandidates,
            batchResults: batchResults,
            filter: filter
        )
    }

    // MARK: - Private Helpers for AccelerableIndex

    /// Compute distances for AccelerationCandidates.
    ///
    /// Extracts vectors from contiguous storage and computes distances.
    /// Uses GPU for large candidate sets, CPU (with 384-dim optimization) otherwise.
    private func computeDistancesForCandidates(
        query: [Float],
        candidates: AccelerationCandidates,
        metric: SupportedDistanceMetric
    ) async throws -> [Float] {
        guard candidates.vectorCount > 0 else { return [] }

        let dimension = candidates.dimension
        let candidateCount = candidates.vectorCount

        // Decide compute path
        let useGPU = shouldUseGPU(candidateCount: candidateCount, dimension: dimension)

        if useGPU, let engine = computeEngine {
            // Extract vectors for GPU processing
            let candidateVectors = extractVectors(from: candidates)
            return try await gpuBatchDistanceFromVectors(
                query: query,
                candidates: candidateVectors,
                metric: metric,
                engine: engine
            )
        } else {
            // CPU path - extract and compute with 384-dim optimization
            let candidateVectors = extractVectors(from: candidates)
            return try await cpuBatchDistance(
                query: query,
                candidates: candidateVectors,
                metric: metric
            )
        }
    }

    /// Extract vectors from AccelerationCandidates contiguous storage.
    private func extractVectors(from candidates: AccelerationCandidates) -> [[Float]] {
        var vectors: [[Float]] = []
        vectors.reserveCapacity(candidates.vectorCount)

        for i in 0..<candidates.vectorCount {
            let slice = candidates.vector(at: i)
            vectors.append(Array(slice))
        }

        return vectors
    }

    /// GPU batch distance using ComputeEngine directly.
    private func gpuBatchDistanceFromVectors(
        query: [Float],
        candidates: [[Float]],
        metric: SupportedDistanceMetric,
        engine: ComputeEngine
    ) async throws -> [Float] {
        let start = CFAbsoluteTimeGetCurrent()

        let distances = try await engine.batchDistance(
            query: query,
            candidates: candidates,
            metric: metric
        )

        stats.gpuOperations += 1
        stats.gpuTimeTotal += CFAbsoluteTimeGetCurrent() - start
        consecutiveGPUErrors = 0

        return distances
    }

    /// Select top-k smallest distances using VectorCore's TopKSelection.
    private func selectTopK(k: Int, from distances: [Float]) -> TopKResult {
        TopKSelection.select(k: k, from: distances)
    }
}

// MARK: - Mutable Statistics

/// Internal mutable statistics for tracking.
private struct MutableStatistics {
    var gpuOperations: Int = 0
    var cpuOperations: Int = 0
    var gpuTimeTotal: TimeInterval = 0
    var cpuTimeTotal: TimeInterval = 0
    var gpuFallbacks: Int = 0
}

// MARK: - Shared Instance Helper

/// Actor for thread-safe singleton management.
private actor SharedAccelerationManager {
    static let holder = SharedAccelerationManager()
    private var manager: AccelerationManager?

    static var instance: AccelerationManager {
        get async {
            if let existing = await holder.manager {
                return existing
            }
            let new = await AccelerationManager()
            await holder.setManager(new)
            return new
        }
    }

    private func setManager(_ manager: AccelerationManager) {
        self.manager = manager
    }
}

