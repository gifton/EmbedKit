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
        identifier: ModelIdentifier = .default,
        maxSequenceLength: Int = 512,
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
    
    // MARK: - Presets
    
    public static let `default` = ModelConfiguration()
    
    public static let highPerformance = ModelConfiguration(
        maxSequenceLength: 256,
        computeUnits: .cpuAndGPU
    )
    
    public static let memoryOptimized = ModelConfiguration(
        maxSequenceLength: 128,
        loadingOptions: LoadingOptions(preloadWeights: false),
        computeUnits: .cpuOnly
    )
    
    public static let production = ModelConfiguration(
        identifier: .miniLM_L6_v2,
        maxSequenceLength: 512,
        normalizeEmbeddings: true,
        poolingStrategy: .mean,
        loadingOptions: LoadingOptions(
            preloadWeights: true,
            enableOptimizations: true,
            verifyIntegrity: true
        ),
        computeUnits: .auto
    )
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