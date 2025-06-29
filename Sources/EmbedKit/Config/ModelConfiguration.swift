import Foundation

// MARK: - Model Configuration

/// Configuration specific to model loading and inference
public struct ModelConfiguration: Sendable {
    /// Model identifier
    public let identifier: ModelIdentifier
    
    /// Maximum sequence length for input
    public let maxSequenceLength: Int
    
    /// Whether to normalize embeddings
    public let normalizeEmbeddings: Bool
    
    /// Pooling strategy for embeddings
    public let poolingStrategy: PoolingStrategy
    
    /// Model loading options
    public let loadingOptions: LoadingOptions
    
    /// Compute units to use
    public let computeUnits: ComputeUnits
    
    public init(
        identifier: ModelIdentifier,
        maxSequenceLength: Int,
        normalizeEmbeddings: Bool = true,
        poolingStrategy: PoolingStrategy = .mean,
        loadingOptions: LoadingOptions = LoadingOptions(),
        computeUnits: ComputeUnits = .auto
    ) {
        self.identifier = identifier
        self.maxSequenceLength = maxSequenceLength
        self.normalizeEmbeddings = normalizeEmbeddings
        self.poolingStrategy = poolingStrategy
        self.loadingOptions = loadingOptions
        self.computeUnits = computeUnits
    }
    
    // MARK: - Model-Specific Factory Methods
    
    /// Create configuration for MiniLM-L6-v2 model
    public static func miniLM_L6_v2(
        normalizeEmbeddings: Bool = true,
        poolingStrategy: PoolingStrategy = .mean,
        loadingOptions: LoadingOptions = LoadingOptions(),
        computeUnits: ComputeUnits = .auto
    ) -> ModelConfiguration {
        ModelConfiguration(
            identifier: .miniLM_L6_v2,
            maxSequenceLength: 512,
            normalizeEmbeddings: normalizeEmbeddings,
            poolingStrategy: poolingStrategy,
            loadingOptions: loadingOptions,
            computeUnits: computeUnits
        )
    }
    
    /// Create configuration for a custom model with explicit dimensions
    public static func custom(
        identifier: ModelIdentifier,
        maxSequenceLength: Int,
        normalizeEmbeddings: Bool = true,
        poolingStrategy: PoolingStrategy = .mean,
        loadingOptions: LoadingOptions = LoadingOptions(),
        computeUnits: ComputeUnits = .auto
    ) -> ModelConfiguration {
        ModelConfiguration(
            identifier: identifier,
            maxSequenceLength: maxSequenceLength,
            normalizeEmbeddings: normalizeEmbeddings,
            poolingStrategy: poolingStrategy,
            loadingOptions: loadingOptions,
            computeUnits: computeUnits
        )
    }
    
    /// Create high-performance configuration with explicit dimensions
    public static func highPerformance(
        identifier: ModelIdentifier,
        maxSequenceLength: Int
    ) -> ModelConfiguration {
        ModelConfiguration(
            identifier: identifier,
            maxSequenceLength: maxSequenceLength,
            computeUnits: .cpuAndGPU
        )
    }
    
    /// Create memory-optimized configuration with explicit dimensions
    public static func memoryOptimized(
        identifier: ModelIdentifier,
        maxSequenceLength: Int
    ) -> ModelConfiguration {
        ModelConfiguration(
            identifier: identifier,
            maxSequenceLength: maxSequenceLength,
            loadingOptions: LoadingOptions(preloadWeights: false),
            computeUnits: .cpuOnly
        )
    }
}

/// Model loading options
public struct LoadingOptions: Sendable {
    /// Whether to preload model weights
    public let preloadWeights: Bool
    
    /// Whether to enable CoreML optimizations
    public let enableOptimizations: Bool
    
    /// Whether to verify model integrity
    public let verifyIntegrity: Bool
    
    /// Custom model URL (if not using bundled model)
    public let customModelURL: URL?
    
    public init(
        preloadWeights: Bool = true,
        enableOptimizations: Bool = true,
        verifyIntegrity: Bool = false,
        customModelURL: URL? = nil
    ) {
        self.preloadWeights = preloadWeights
        self.enableOptimizations = enableOptimizations
        self.verifyIntegrity = verifyIntegrity
        self.customModelURL = customModelURL
    }
}

/// Compute units for model execution
public enum ComputeUnits: String, Sendable {
    case cpuOnly = "cpu_only"
    case cpuAndGPU = "cpu_and_gpu"
    case cpuAndNeuralEngine = "cpu_and_neural_engine"
    case all = "all"
    case auto = "auto"
}