import Foundation

/// Numeric configuration for acceleration paths (CPU/GPU)
///
/// Controls stability/performance trade-offs and epsilon thresholds used
/// for operations like normalization. This provides a single surface that
/// higher-level configurations can pass through to acceleration backends.
public struct AccelerationNumerics: Sendable {
    /// Enable more numerically stable algorithms (e.g., two-pass normalization)
    /// at the cost of performance. Intended for debugging/validation builds
    /// or ill-conditioned inputs.
    public var stableNormalizationEnabled: Bool

    /// Small threshold used to prevent division by zero or to treat
    /// near-zero magnitudes as zero. This should mirror VectorCore defaults
    /// so that CPU and GPU make consistent decisions.
    public var epsilon: Float

    public init(
        stableNormalizationEnabled: Bool = true,
        epsilon: Float = 1e-8
    ) {
        self.stableNormalizationEnabled = stableNormalizationEnabled
        self.epsilon = epsilon
    }
}
