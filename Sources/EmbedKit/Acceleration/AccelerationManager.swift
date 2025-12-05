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

    /// Statistics tracking.
    private var stats: MutableStatistics

    // MARK: - Initialization

    /// Create an acceleration manager with a new Metal4Context.
    public init() async throws {
        self.context = try await Metal4ContextManager.shared()
        self.distanceProvider = try await UniversalKernelDistanceProvider(context: context)
        self.normalizationKernel = try await L2NormalizationKernel(context: context)
        self.stats = MutableStatistics()
    }

    /// Create an acceleration manager with a shared Metal4Context.
    public init(context: Metal4Context) async throws {
        self.context = context
        self.distanceProvider = try await UniversalKernelDistanceProvider(context: context)
        self.normalizationKernel = try await L2NormalizationKernel(context: context)
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
