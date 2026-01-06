// EmbedKit - UMAP Configuration
// Configuration for GPU-accelerated UMAP dimensionality reduction

import Foundation
import VectorAccelerate

// MARK: - UMAP Configuration

/// Configuration for UMAP (Uniform Manifold Approximation and Projection) dimensionality reduction.
///
/// UMAP projects high-dimensional embeddings into lower dimensions (2D or 3D) while
/// preserving local neighborhood structure. This enables visualization and exploration
/// of embedding spaces.
///
/// EmbedKit's UMAP implementation uses VectorAccelerate's GPU-accelerated gradient kernel,
/// achieving ~10x speedup over CPU implementations for large datasets.
///
/// ## Example Usage
///
/// ```swift
/// // Default 2D projection
/// let config = UMAPConfiguration.visualization2D()
///
/// // Custom configuration for dense embeddings
/// let config = UMAPConfiguration(
///     targetDimension: 3,
///     neighbors: 30,
///     minDistance: 0.05,
///     iterations: 300
/// )
/// ```
///
/// ## Parameters
///
/// - **targetDimension**: Output dimensionality (2 for scatter plots, 3 for 3D visualization)
/// - **neighbors**: Number of nearest neighbors to consider. Higher values preserve more
///   global structure; lower values emphasize local clusters. Typical range: 5-50.
/// - **minDistance**: Minimum distance between points in the output space. Lower values
///   create tighter clusters; higher values spread points more evenly. Range: 0.0-1.0.
/// - **iterations**: Number of optimization epochs. More iterations improve quality but
///   increase computation time. Typical range: 100-500.
///
/// ## Performance Characteristics
///
/// - Time complexity: O(n × k × d × iterations) where n = points, k = neighbors, d = dimension
/// - Memory: O(n × k) for the k-NN graph
/// - GPU acceleration provides ~10x speedup for n > 1000
public struct UMAPConfiguration: Sendable, Equatable, Codable {

    // MARK: - Core Parameters

    /// Target output dimensionality.
    ///
    /// - 2: For 2D scatter plots (most common)
    /// - 3: For 3D visualizations
    ///
    /// Higher dimensions are supported but rarely useful for visualization.
    public var targetDimension: Int

    /// Number of nearest neighbors to consider for local structure.
    ///
    /// Controls the balance between local and global structure preservation:
    /// - Lower values (5-10): Emphasize local clusters, may fragment global structure
    /// - Higher values (30-50): Preserve more global relationships, smoother manifolds
    ///
    /// Typical default: 15
    public var neighbors: Int

    /// Minimum distance between points in the output space.
    ///
    /// Controls cluster density in the projection:
    /// - Lower values (0.0-0.1): Tight, dense clusters with clear separation
    /// - Higher values (0.25-1.0): More uniform spread, less pronounced clustering
    ///
    /// Typical default: 0.1
    public var minDistance: Float

    /// Number of optimization epochs.
    ///
    /// More iterations generally improve projection quality at the cost of time:
    /// - 100-200: Quick visualization, may have some distortion
    /// - 200-500: Good quality for most use cases
    /// - 500+: High quality, diminishing returns
    ///
    /// GPU acceleration makes higher iteration counts practical.
    public var iterations: Int

    // MARK: - Advanced Parameters

    /// Initial learning rate for gradient descent.
    ///
    /// Controls step size during optimization. The rate decays linearly over epochs.
    /// Default of 1.0 works well for most cases. Increase for faster convergence
    /// on large datasets, decrease if optimization is unstable.
    public var learningRate: Float

    /// Number of negative samples per positive edge.
    ///
    /// Negative sampling provides repulsive force to spread non-neighbors apart.
    /// Higher values improve separation but increase computation.
    ///
    /// Typical default: 5
    public var negativeSampleRate: Int

    /// Spread parameter for the UMAP curve.
    ///
    /// Together with `minDistance`, controls the effective distance scaling.
    /// Most users should leave this at 1.0 and adjust `minDistance` instead.
    public var spread: Float

    // MARK: - Initialization

    /// Creates a UMAP configuration with specified parameters.
    ///
    /// - Parameters:
    ///   - targetDimension: Output dimensionality (default: 2)
    ///   - neighbors: Number of nearest neighbors (default: 15)
    ///   - minDistance: Minimum distance in output space (default: 0.1)
    ///   - iterations: Number of optimization epochs (default: 200)
    ///   - learningRate: Initial learning rate (default: 1.0)
    ///   - negativeSampleRate: Negative samples per edge (default: 5)
    ///   - spread: Spread parameter (default: 1.0)
    public init(
        targetDimension: Int = 2,
        neighbors: Int = 15,
        minDistance: Float = 0.1,
        iterations: Int = 200,
        learningRate: Float = 1.0,
        negativeSampleRate: Int = 5,
        spread: Float = 1.0
    ) {
        self.targetDimension = targetDimension
        self.neighbors = neighbors
        self.minDistance = minDistance
        self.iterations = iterations
        self.learningRate = learningRate
        self.negativeSampleRate = negativeSampleRate
        self.spread = spread
    }

    // MARK: - Factory Methods

    /// Default configuration for general use.
    ///
    /// Projects to 2D with balanced parameters suitable for most embedding types.
    public static let `default` = UMAPConfiguration()

    /// Configuration optimized for 2D scatter plot visualization.
    ///
    /// Uses standard parameters that work well for embedding visualization.
    /// Suitable for plotting with Swift Charts or other 2D graphing libraries.
    ///
    /// - Returns: Configuration for 2D projection
    ///
    /// ## Example
    /// ```swift
    /// let config = UMAPConfiguration.visualization2D()
    /// let points = try await manager.projectEmbeddings(embeddings, config: config)
    /// // points[i] = [x, y] coordinates
    /// ```
    public static func visualization2D() -> UMAPConfiguration {
        UMAPConfiguration(
            targetDimension: 2,
            neighbors: 15,
            minDistance: 0.1,
            iterations: 200
        )
    }

    /// Configuration optimized for 3D visualization.
    ///
    /// Projects to 3D space for use with SceneKit, RealityKit, or other 3D engines.
    /// Uses slightly more iterations than 2D to account for additional dimension.
    ///
    /// - Returns: Configuration for 3D projection
    ///
    /// ## Example
    /// ```swift
    /// let config = UMAPConfiguration.visualization3D()
    /// let points = try await manager.projectEmbeddings(embeddings, config: config)
    /// // points[i] = [x, y, z] coordinates
    /// ```
    public static func visualization3D() -> UMAPConfiguration {
        UMAPConfiguration(
            targetDimension: 3,
            neighbors: 15,
            minDistance: 0.1,
            iterations: 250
        )
    }

    /// Configuration for quick preview visualization.
    ///
    /// Reduced iterations for faster results at the cost of projection quality.
    /// Useful for interactive exploration or initial dataset overview.
    ///
    /// - Parameter dimension: Target dimension (default: 2)
    /// - Returns: Fast preview configuration
    public static func quickPreview(dimension: Int = 2) -> UMAPConfiguration {
        UMAPConfiguration(
            targetDimension: dimension,
            neighbors: 10,
            minDistance: 0.1,
            iterations: 100
        )
    }

    /// Configuration for high-quality visualization.
    ///
    /// More neighbors and iterations for better structure preservation.
    /// Use for publication-quality visualizations or detailed analysis.
    ///
    /// - Parameter dimension: Target dimension (default: 2)
    /// - Returns: High-quality configuration
    public static func highQuality(dimension: Int = 2) -> UMAPConfiguration {
        UMAPConfiguration(
            targetDimension: dimension,
            neighbors: 30,
            minDistance: 0.1,
            iterations: 500
        )
    }

    /// Configuration emphasizing cluster separation.
    ///
    /// Lower neighbors and minDistance create tighter, more separated clusters.
    /// Useful when you expect distinct groups in your embeddings.
    ///
    /// - Parameter dimension: Target dimension (default: 2)
    /// - Returns: Configuration for cluster emphasis
    public static func clusterEmphasis(dimension: Int = 2) -> UMAPConfiguration {
        UMAPConfiguration(
            targetDimension: dimension,
            neighbors: 10,
            minDistance: 0.05,
            iterations: 250
        )
    }

    /// Configuration preserving global structure.
    ///
    /// Higher neighbors preserve more global relationships between clusters.
    /// Use when understanding the overall topology is more important than local detail.
    ///
    /// - Parameter dimension: Target dimension (default: 2)
    /// - Returns: Configuration for global structure
    public static func globalStructure(dimension: Int = 2) -> UMAPConfiguration {
        UMAPConfiguration(
            targetDimension: dimension,
            neighbors: 50,
            minDistance: 0.25,
            iterations: 300
        )
    }

    // MARK: - VectorAccelerate Integration

    /// Converts to VectorAccelerate's UMAPParameters for kernel execution.
    ///
    /// The `a` and `b` curve parameters are computed from `minDistance` and `spread`
    /// using UMAP's standard formula.
    ///
    /// - Parameter epoch: Current epoch (for learning rate decay)
    /// - Parameter totalEpochs: Total number of epochs
    /// - Returns: GPU kernel parameters
    @available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
    internal func toKernelParameters(epoch: Int = 0, totalEpochs: Int? = nil) -> UMAPParameters {
        // Compute decayed learning rate
        let effectiveTotalEpochs = totalEpochs ?? iterations
        let decayFactor = 1.0 - Float(epoch) / Float(max(effectiveTotalEpochs, 1))
        let effectiveLR = learningRate * max(decayFactor, 0.01)

        // Use VectorAccelerate's parameter computation from minDistance
        var params = UMAPParameters.from(minDist: minDistance, spread: spread)
        params.learningRate = effectiveLR
        params.negativeSampleRate = negativeSampleRate

        return params
    }

    // MARK: - Validation

    /// Validates configuration parameters.
    ///
    /// - Throws: `UMAPConfigurationError` if any parameters are invalid
    public func validate() throws {
        guard targetDimension >= 1 else {
            throw UMAPConfigurationError.invalidTargetDimension(targetDimension)
        }
        guard neighbors >= 2 else {
            throw UMAPConfigurationError.invalidNeighbors(neighbors)
        }
        guard minDistance >= 0 && minDistance <= 1 else {
            throw UMAPConfigurationError.invalidMinDistance(minDistance)
        }
        guard iterations >= 1 else {
            throw UMAPConfigurationError.invalidIterations(iterations)
        }
        guard learningRate > 0 else {
            throw UMAPConfigurationError.invalidLearningRate(learningRate)
        }
        guard negativeSampleRate >= 0 else {
            throw UMAPConfigurationError.invalidNegativeSampleRate(negativeSampleRate)
        }
        guard spread > 0 else {
            throw UMAPConfigurationError.invalidSpread(spread)
        }
    }
}

// MARK: - Errors

/// Errors from UMAP configuration validation.
public enum UMAPConfigurationError: Error, LocalizedError, Sendable {
    case invalidTargetDimension(Int)
    case invalidNeighbors(Int)
    case invalidMinDistance(Float)
    case invalidIterations(Int)
    case invalidLearningRate(Float)
    case invalidNegativeSampleRate(Int)
    case invalidSpread(Float)
    case insufficientPoints(required: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidTargetDimension(let d):
            return "Invalid target dimension: \(d). Must be >= 1."
        case .invalidNeighbors(let n):
            return "Invalid neighbors: \(n). Must be >= 2."
        case .invalidMinDistance(let d):
            return "Invalid minDistance: \(d). Must be in range [0, 1]."
        case .invalidIterations(let i):
            return "Invalid iterations: \(i). Must be >= 1."
        case .invalidLearningRate(let lr):
            return "Invalid learningRate: \(lr). Must be > 0."
        case .invalidNegativeSampleRate(let r):
            return "Invalid negativeSampleRate: \(r). Must be >= 0."
        case .invalidSpread(let s):
            return "Invalid spread: \(s). Must be > 0."
        case .insufficientPoints(let required, let actual):
            return "Insufficient points: need at least \(required) for \(required) neighbors, got \(actual)."
        }
    }
}

// MARK: - CustomStringConvertible

extension UMAPConfiguration: CustomStringConvertible {
    public var description: String {
        "UMAPConfiguration(dim=\(targetDimension), k=\(neighbors), minDist=\(minDistance), iters=\(iterations))"
    }
}
