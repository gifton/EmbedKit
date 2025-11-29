// EmbedKit - Compute Preference Types
// Configuration for GPU/CPU compute path selection

import Foundation

// MARK: - Compute Preference

/// Preference for GPU acceleration of search and distance operations.
///
/// Controls whether GPU acceleration is used for vector search, distance computation,
/// and batch operations in `AccelerationManager` and `EmbeddingStore`.
///
/// This is different from `ComputeDevice` (used for model inference target):
/// - `ComputeDevice`: Controls CoreML compute units for **model inference**
/// - `ComputePreference`: Controls GPU/CPU path for **search and distance operations**
///
/// ## Usage
/// ```swift
/// // For search acceleration
/// let config = IndexConfiguration(dimension: 384, computePreference: .auto)
///
/// // For AccelerationManager
/// let manager = await AccelerationManager(preference: .cpuOnly)
/// ```
public enum ComputePreference: String, Codable, Sendable, CaseIterable {
    /// Automatically select the best compute path based on operation characteristics.
    /// Uses GPU when beneficial (large batches, high dimensions), CPU otherwise.
    /// This is the recommended default.
    case auto

    /// Force CPU-only computation.
    /// Use for deterministic results in tests or when GPU is unavailable.
    case cpuOnly

    /// Prefer GPU when available, with no fallback to CPU.
    /// Use for debugging GPU-specific issues. Will throw if GPU unavailable.
    case gpuOnly
}

// MARK: - Acceleration Thresholds

/// Thresholds for determining when GPU acceleration is beneficial.
///
/// Based on VectorAccelerate benchmarks:
/// - GPU overhead makes small operations slower than CPU
/// - GPU excels at large batch operations with high-dimensional vectors
public struct AccelerationThresholds: Sendable {
    /// Minimum number of candidate vectors for GPU batch distance computation.
    /// Below this threshold, CPU is faster due to GPU dispatch overhead.
    public let minCandidatesForGPU: Int

    /// Minimum vector dimension for GPU operations.
    /// Low-dimensional vectors don't benefit from GPU parallelism.
    public let minDimensionForGPU: Int

    /// Minimum batch size for GPU normalization.
    public let minBatchForNormalization: Int

    /// Default thresholds based on VectorAccelerate benchmarks.
    public static let `default` = AccelerationThresholds(
        minCandidatesForGPU: 1000,
        minDimensionForGPU: 64,
        minBatchForNormalization: 100
    )

    /// Aggressive thresholds - use GPU more often (for testing).
    public static let aggressive = AccelerationThresholds(
        minCandidatesForGPU: 100,
        minDimensionForGPU: 32,
        minBatchForNormalization: 10
    )

    /// Conservative thresholds - use GPU only for very large operations.
    public static let conservative = AccelerationThresholds(
        minCandidatesForGPU: 10_000,
        minDimensionForGPU: 128,
        minBatchForNormalization: 500
    )

    public init(
        minCandidatesForGPU: Int = 1000,
        minDimensionForGPU: Int = 64,
        minBatchForNormalization: Int = 100
    ) {
        self.minCandidatesForGPU = minCandidatesForGPU
        self.minDimensionForGPU = minDimensionForGPU
        self.minBatchForNormalization = minBatchForNormalization
    }
}

// MARK: - Acceleration Statistics

/// Statistics about GPU/CPU usage in acceleration operations.
public struct AccelerationStatistics: Sendable {
    /// Number of operations routed to GPU.
    public let gpuOperations: Int

    /// Number of operations routed to CPU.
    public let cpuOperations: Int

    /// Total time spent on GPU operations.
    public let gpuTimeTotal: TimeInterval

    /// Total time spent on CPU operations.
    public let cpuTimeTotal: TimeInterval

    /// Number of GPU fallbacks due to errors.
    public let gpuFallbacks: Int

    /// Computed average speedup (GPU time / CPU time for comparable operations).
    public var averageSpeedup: Double {
        guard cpuTimeTotal > 0, gpuTimeTotal > 0 else { return 1.0 }
        return cpuTimeTotal / gpuTimeTotal
    }

    /// Percentage of operations using GPU.
    public var gpuUtilization: Double {
        let total = gpuOperations + cpuOperations
        guard total > 0 else { return 0.0 }
        return Double(gpuOperations) / Double(total)
    }

    public init(
        gpuOperations: Int = 0,
        cpuOperations: Int = 0,
        gpuTimeTotal: TimeInterval = 0,
        cpuTimeTotal: TimeInterval = 0,
        gpuFallbacks: Int = 0
    ) {
        self.gpuOperations = gpuOperations
        self.cpuOperations = cpuOperations
        self.gpuTimeTotal = gpuTimeTotal
        self.cpuTimeTotal = cpuTimeTotal
        self.gpuFallbacks = gpuFallbacks
    }

    /// Combine two statistics.
    public func merged(with other: AccelerationStatistics) -> AccelerationStatistics {
        AccelerationStatistics(
            gpuOperations: gpuOperations + other.gpuOperations,
            cpuOperations: cpuOperations + other.cpuOperations,
            gpuTimeTotal: gpuTimeTotal + other.gpuTimeTotal,
            cpuTimeTotal: cpuTimeTotal + other.cpuTimeTotal,
            gpuFallbacks: gpuFallbacks + other.gpuFallbacks
        )
    }
}

// MARK: - Acceleration Error

/// Errors from acceleration operations.
public enum AccelerationError: Error, LocalizedError {
    case gpuNotAvailable
    case gpuOperationFailed(String)
    /// Dimension mismatch error. Parameter naming matches `EmbedKitError.dimensionMismatch`.
    case dimensionMismatch(expected: Int, got: Int)
    case invalidInput(String)

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
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .gpuNotAvailable:
            return "Use `.cpuOnly` compute preference for CPU-based processing. GPU acceleration requires a Metal-compatible device (most Macs since 2012, all iOS devices since iPhone 5s)."
        case .gpuOperationFailed:
            return "Try reducing batch size or input dimensions. If the error persists, fall back to CPU processing with `.cpuOnly` preference."
        case .dimensionMismatch:
            return "Ensure all vectors in the operation have the same dimensions. Check that embeddings come from the same model."
        case .invalidInput:
            return "Verify input data is not empty, contains valid values (no NaN/Inf), and meets the operation's requirements."
        }
    }
}
