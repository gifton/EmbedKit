// EmbedKit - Acceleration Manager
// Manages GPU/CPU compute path selection and execution

import Foundation
import VectorCore
import VectorAccelerate

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

    /// CPU batch distance computation.
    private func cpuBatchDistance(
        query: [Float],
        candidates: [[Float]],
        metric: SupportedDistanceMetric
    ) async throws -> [Float] {
        let start = CFAbsoluteTimeGetCurrent()

        let distances = candidates.map { candidate in
            cpuDistance(query, candidate, metric: metric)
        }

        stats.cpuOperations += 1
        stats.cpuTimeTotal += CFAbsoluteTimeGetCurrent() - start

        return distances
    }

    /// CPU single distance computation.
    private func cpuDistance(_ a: [Float], _ b: [Float], metric: SupportedDistanceMetric) -> Float {
        switch metric {
        case .cosine:
            var dotProduct: Float = 0
            var normA: Float = 0
            var normB: Float = 0

            for i in 0..<a.count {
                dotProduct += a[i] * b[i]
                normA += a[i] * a[i]
                normB += b[i] * b[i]
            }

            let magnitude = sqrt(normA) * sqrt(normB)
            guard magnitude > 0 else { return 1.0 }
            return 1.0 - (dotProduct / magnitude) // Cosine distance = 1 - similarity

        case .euclidean:
            var sum: Float = 0
            for i in 0..<a.count {
                let diff = a[i] - b[i]
                sum += diff * diff
            }
            return sqrt(sum)

        case .dotProduct:
            var sum: Float = 0
            for i in 0..<a.count {
                sum += a[i] * b[i]
            }
            // For dot product, higher is more similar, so negate for distance-like ordering
            return -sum

        case .manhattan:
            var sum: Float = 0
            for i in 0..<a.count {
                sum += abs(a[i] - b[i])
            }
            return sum

        case .chebyshev:
            var maxDiff: Float = 0
            for i in 0..<a.count {
                maxDiff = max(maxDiff, abs(a[i] - b[i]))
            }
            return maxDiff
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

