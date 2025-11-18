// ============================================================================
// MARK: - Core Protocol Definitions
// ============================================================================

/// Core protocol that all embedding models must conform to
public protocol EmbeddingModelProtocol: Sendable {
    associatedtype Configuration: ModelConfigurationProtocol
    associatedtype Tokenizer: TokenizerProtocol
    associatedtype Backend: ModelBackendProtocol

    var modelIdentifier: ModelIdentifier { get }
    var configuration: Configuration { get }
    var tokenizer: Tokenizer { get }
    var backend: Backend { get }
    var dimensions: Int { get }

    /// Embed a single text
    func embed(_ text: String) async throws -> Embedding

    /// Batch embedding for efficiency
    func embedBatch(_ texts: [String]) async throws -> [Embedding]

    /// Model capabilities
    var capabilities: ModelCapabilities { get }
}

/// Model identification and metadata
public struct ModelIdentifier: Hashable, Sendable {
    public let provider: ModelProvider
    public let name: String
    public let version: String
    public let variant: String? // e.g., "base", "large", "xl"

    var fullIdentifier: String {
        "\(provider.rawValue)/\(name)-\(variant ?? "base")-v\(version)"
    }
}

public enum ModelProvider: String, CaseIterable, Sendable {
    case openai = "openai"
    case anthropic = "anthropic"
    case apple = "apple"           // Apple's on-device models
    case huggingface = "huggingface"
    case google = "google"
    case local = "local"           // Custom local models
}

/// Capabilities that models can advertise
public struct ModelCapabilities: OptionSet, Sendable {
    public let rawValue: Int

    public static let textEmbedding = ModelCapabilities(rawValue: 1 << 0)
    public static let multiModal = ModelCapabilities(rawValue: 1 << 1)
    public static let streaming = ModelCapabilities(rawValue: 1 << 2)
    public static let fineTunable = ModelCapabilities(rawValue: 1 << 3)
    public static let onDevice = ModelCapabilities(rawValue: 1 << 4)
    public static let cloudBased = ModelCapabilities(rawValue: 1 << 5)
    public static let batchProcessing = ModelCapabilities(rawValue: 1 << 6)
    public static let semanticSearch = ModelCapabilities(rawValue: 1 << 7)
}

// ============================================================================
// MARK: - Model Configuration
// ============================================================================

public protocol ModelConfigurationProtocol: Sendable {
    var maxSequenceLength: Int { get }
    var dimensions: Int { get }
    var vocabularySize: Int { get }
    var poolingStrategy: PoolingStrategy { get }
    var normalizeEmbeddings: Bool { get }

    /// Model-specific parameters
    var customParameters: [String: Any] { get }
}

public enum PoolingStrategy: String, CaseIterable, Sendable {
    case mean
    case max
    case cls  // Use CLS token (BERT-style)
    case lastToken  // Use last token (GPT-style)
    case attentionWeighted
}

// ============================================================================
// MARK: - Tokenizer Protocol Hierarchy
// ============================================================================

public protocol TokenizerProtocol: Sendable {
    var strategy: TokenizationStrategy { get }
    var vocabulary: VocabularyProtocol { get }
    var specialTokens: SpecialTokenConfiguration { get }

    func tokenize(_ text: String) async throws -> TokenizedInput
    func detokenize(_ tokens: [Int]) async throws -> String
}

public protocol VocabularyProtocol: Sendable {
    var size: Int { get }
    func tokenToId(_ token: String) -> Int?
    func idToToken(_ id: Int) -> String?
    func encode(_ text: String) -> [String]  // Split into tokens
}

public enum TokenizationStrategy: String, CaseIterable, Sendable {
    case wordPiece      // BERT, RoBERTa
    case bpe            // GPT, GPT-2
    case sentencePiece  // T5, LLaMA
    case unigram        // Albert
    case character      // CharCNN
    case word           // Simple word-level
    case custom         // For proprietary tokenizers
}

/// Extended special tokens for different model types
public struct SpecialTokenConfiguration: Sendable {
    // Common tokens
    public let pad: TokenInfo?
    public let unk: TokenInfo?
    public let cls: TokenInfo?
    public let sep: TokenInfo?
    public let mask: TokenInfo?

    // GPT-style tokens
    public let bos: TokenInfo?  // Beginning of sequence
    public let eos: TokenInfo?  // End of sequence

    // Additional tokens
    public let additional: [String: TokenInfo]

    public struct TokenInfo: Sendable {
        public let token: String
        public let id: Int
    }
}

// ============================================================================
// MARK: - Backend Abstraction
// ============================================================================

public protocol ModelBackendProtocol: Sendable {
    associatedtype Input
    associatedtype Output

    var backendType: BackendType { get }
    var device: ComputeDevice { get }

    func process(_ input: Input) async throws -> Output
    func warmup() async throws  // Pre-load model
    func unload() async throws  // Free resources
}

public enum BackendType: String, CaseIterable, Sendable {
    case coreML
    case metal
    case onnx
    case tensorflow
    case pytorch
    case custom
}

public enum ComputeDevice: String, CaseIterable, Sendable {
    case cpu
    case gpu
    case neuralEngine
    case auto  // Let system decide
}

// ============================================================================
// MARK: - Model Registry (Discovery & Management)
// ============================================================================

/// Central registry for all available models
public actor ModelRegistry {
    private var registeredModels: [ModelIdentifier: any ModelFactoryProtocol] = [:]
    private var activeModels: [ModelIdentifier: any EmbeddingModelProtocol] = [:]
    private let cache: ModelCache

    public static let shared = ModelRegistry()

    private init() {
        self.cache = ModelCache()
        registerBuiltInModels()
    }

    /// Register a new model factory
    public func register<F: ModelFactoryProtocol>(_ factory: F) {
        let identifier = factory.modelIdentifier
        registeredModels[identifier] = factory
    }

    /// Get list of available models
    public func availableModels(
        provider: ModelProvider? = nil,
        capabilities: ModelCapabilities? = nil
    ) -> [ModelInfo] {
        registeredModels.compactMap { identifier, factory in
            // Filter by provider if specified
            if let provider = provider, identifier.provider != provider {
                return nil
            }

            // Filter by capabilities if specified
            if let capabilities = capabilities,
               !factory.capabilities.contains(capabilities) {
                return nil
            }

            return ModelInfo(
                identifier: identifier,
                capabilities: factory.capabilities,
                metadata: factory.metadata
            )
        }
    }

    /// Load a specific model
    public func loadModel(
        _ identifier: ModelIdentifier,
        configuration: ModelConfiguration? = nil
    ) async throws -> any EmbeddingModelProtocol {
        // Check if already loaded
        if let existingModel = activeModels[identifier] {
            return existingModel
        }

        // Find factory
        guard let factory = registeredModels[identifier] else {
            throw ModelError.modelNotFound(identifier)
        }

        // Create model
        let model = try await factory.createModel(configuration: configuration)

        // Cache it
        activeModels[identifier] = model

        return model
    }

    /// Unload a model to free resources
    public func unloadModel(_ identifier: ModelIdentifier) async throws {
        guard let model = activeModels[identifier] else { return }

        // Unload backend resources
        try await model.backend.unload()

        // Remove from active models
        activeModels.removeValue(forKey: identifier)
    }

    private func registerBuiltInModels() {
        // Register Apple's on-device models
        register(AppleEmbeddingModelFactory())

        // Register BERT models
        register(BERTModelFactory(variant: .base))
        register(BERTModelFactory(variant: .large))

        // Register OpenAI models
        register(OpenAIEmbeddingModelFactory(model: .textEmbeddingAda002))
        register(OpenAIEmbeddingModelFactory(model: .textEmbedding3Small))
        register(OpenAIEmbeddingModelFactory(model: .textEmbedding3Large))

        // Register local models
        register(LocalModelFactory())
    }
}

// ============================================================================
// MARK: - Factory Protocol
// ============================================================================

public protocol ModelFactoryProtocol: Sendable {
    associatedtype Model: EmbeddingModelProtocol

    var modelIdentifier: ModelIdentifier { get }
    var capabilities: ModelCapabilities { get }
    var metadata: ModelMetadata { get }

    func createModel(configuration: ModelConfiguration?) async throws -> Model
}

public struct ModelMetadata: Sendable {
    public let displayName: String
    public let description: String
    public let dimensionality: Int
    public let maxTokens: Int
    public let modelSize: Int64  // in bytes
    public let requiredMemory: Int64  // in bytes
    public let supportedLanguages: [String]
    public let license: String?
    public let downloadURL: URL?
}

// ============================================================================
// MARK: - Concrete Implementation Examples
// ============================================================================

/// Apple's on-device embedding model
public struct AppleEmbeddingModel: EmbeddingModelProtocol {
    public typealias Configuration = AppleModelConfiguration
    public typealias Tokenizer = AppleTokenizer
    public typealias Backend = CoreMLBackend

    public let modelIdentifier: ModelIdentifier
    public let configuration: Configuration
    public let tokenizer: Tokenizer
    public let backend: Backend
    public let dimensions: Int

    public var capabilities: ModelCapabilities {
        [.textEmbedding, .onDevice, .batchProcessing, .semanticSearch]
    }

    public func embed(_ text: String) async throws -> Embedding {
        // 1. Tokenize
        let tokens = try await tokenizer.tokenize(text)

        // 2. Process through CoreML
        let output = try await backend.process(tokens)

        // 3. Apply pooling
        let pooled = applyPooling(output, strategy: configuration.poolingStrategy)

        // 4. Normalize if needed
        let final = configuration.normalizeEmbeddings ? normalize(pooled) : pooled

        return Embedding(vector: final, metadata: [:])
    }

    public func embedBatch(_ texts: [String]) async throws -> [Embedding] {
        // Optimized batch processing
        try await withThrowingTaskGroup(of: Embedding.self) { group in
            for text in texts {
                group.addTask {
                    try await self.embed(text)
                }
            }

            var embeddings: [Embedding] = []
            for try await embedding in group {
                embeddings.append(embedding)
            }
            return embeddings
        }
    }

    private func applyPooling(_ output: MLMultiArray, strategy: PoolingStrategy) -> [Float] {
        // Implementation details...
        []
    }

    private func normalize(_ vector: [Float]) -> [Float] {
        // L2 normalization
        let magnitude = sqrt(vector.map { $0 * $0 }.reduce(0, +))
        return vector.map { $0 / magnitude }
    }
}

/// Factory for Apple's models
public struct AppleEmbeddingModelFactory: ModelFactoryProtocol {
    public typealias Model = AppleEmbeddingModel

    public var modelIdentifier: ModelIdentifier {
        ModelIdentifier(
            provider: .apple,
            name: "text-embedding",
            version: "1.0",
            variant: "base"
        )
    }

    public var capabilities: ModelCapabilities {
        [.textEmbedding, .onDevice, .batchProcessing, .semanticSearch]
    }

    public var metadata: ModelMetadata {
        ModelMetadata(
            displayName: "Apple Text Embedding",
            description: "Apple's on-device text embedding model optimized for iOS/macOS",
            dimensionality: 768,
            maxTokens: 512,
            modelSize: 500_000_000,  // ~500MB
            requiredMemory: 1_000_000_000,  // ~1GB
            supportedLanguages: ["en", "es", "fr", "de", "zh", "ja"],
            license: "Apple",
            downloadURL: nil  // Bundled with OS
        )
    }

    public func createModel(configuration: ModelConfiguration?) async throws -> AppleEmbeddingModel {
        let config = configuration as? AppleModelConfiguration ?? AppleModelConfiguration.default

        // Load CoreML model
        let modelURL = try await downloadOrLocateModel()
        let backend = try CoreMLBackend(modelURL: modelURL, device: .neuralEngine)

        // Create tokenizer
        let tokenizer = try await AppleTokenizer(vocabularyPath: config.vocabularyPath)

        // Warm up model
        try await backend.warmup()

        return AppleEmbeddingModel(
            modelIdentifier: modelIdentifier,
            configuration: config,
            tokenizer: tokenizer,
            backend: backend,
            dimensions: config.dimensions
        )
    }

    private func downloadOrLocateModel() async throws -> URL {
        // Logic to find or download model
        URL(fileURLWithPath: "/System/Library/Models/TextEmbedding.mlmodelc")
    }
}

// ============================================================================
// MARK: - Usage Example
// ============================================================================

public struct EmbedKitAPI {
    private let registry = ModelRegistry.shared

    /// List available models
    public func listModels(onDevice: Bool = false) async -> [ModelInfo] {
        let capabilities: ModelCapabilities? = onDevice ? .onDevice : nil
        return await registry.availableModels(capabilities: capabilities)
    }

    /// Easy-to-use embedding API
    public func embed(
        text: String,
        using modelId: String = "apple/text-embedding-base-v1.0"
    ) async throws -> Embedding {
        let identifier = try parseModelIdentifier(modelId)
        let model = try await registry.loadModel(identifier)
        return try await model.embed(text)
    }

    /// Batch embedding with specific model
    public func embedBatch(
        texts: [String],
        using model: ModelIdentifier,
        configuration: ModelConfiguration? = nil
    ) async throws -> [Embedding] {
        let model = try await registry.loadModel(model, configuration: configuration)
        return try await model.embedBatch(texts)
    }

    private func parseModelIdentifier(_ string: String) throws -> ModelIdentifier {
        // Parse "provider/name-variant-version" format
        // Implementation...
        ModelIdentifier(provider: .apple, name: "text-embedding", version: "1.0", variant: "base")
    }
}

// ============================================================================
// MARK: - Configuration Builder (Fluent API)
// ============================================================================

public class ModelConfigurationBuilder {
    private var maxSequenceLength: Int = 512
    private var dimensions: Int = 768
    private var poolingStrategy: PoolingStrategy = .mean
    private var normalizeEmbeddings: Bool = true
    private var customParameters: [String: Any] = [:]

    public func withMaxSequenceLength(_ length: Int) -> Self {
        self.maxSequenceLength = length
        return self
    }

    public func withDimensions(_ dim: Int) -> Self {
        self.dimensions = dim
        return self
    }

    public func withPoolingStrategy(_ strategy: PoolingStrategy) -> Self {
        self.poolingStrategy = strategy
        return self
    }

    public func withNormalization(_ enabled: Bool) -> Self {
        self.normalizeEmbeddings = enabled
        return self
    }

    public func withCustomParameter(_ key: String, value: Any) -> Self {
        self.customParameters[key] = value
        return self
    }

    public func build() -> ModelConfiguration {
        ModelConfiguration(
            maxSequenceLength: maxSequenceLength,
            dimensions: dimensions,
            vocabularySize: 30000,  // Default
            poolingStrategy: poolingStrategy,
            normalizeEmbeddings: normalizeEmbeddings,
            customParameters: customParameters
        )
    }
}

// ============================================================================
// MARK: - Error Handling
// ============================================================================

public enum ModelError: LocalizedError {
    case modelNotFound(ModelIdentifier)
    case modelLoadFailed(String)
    case tokenizationFailed(String)
    case embeddingFailed(String)
    case unsupportedOperation(String)
    case configurationInvalid(String)
    case resourceNotAvailable(String)
    case dimensionMismatch(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let id):
            return "Model not found: \(id.fullIdentifier)"
        case .modelLoadFailed(let reason):
            return "Failed to load model: \(reason)"
        case .tokenizationFailed(let reason):
            return "Tokenization failed: \(reason)"
        case .embeddingFailed(let reason):
            return "Embedding generation failed: \(reason)"
        case .unsupportedOperation(let op):
            return "Unsupported operation: \(op)"
        case .configurationInvalid(let reason):
            return "Invalid configuration: \(reason)"
        case .resourceNotAvailable(let resource):
            return "Resource not available: \(resource)"
        case .dimensionMismatch(let expected, let actual):
            return "Dimension mismatch: expected \(expected), got \(actual)"
        }
    }
}