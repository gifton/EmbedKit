// Model Manager - Clean, Powerful, Customizable

import Foundation
import CoreML
import Metal

// ============================================================================
// MARK: - Model Manager (The Core API)
// ============================================================================

/// The main entry point for working with embedding models
public actor ModelManager {

    // MARK: - Properties

    private var loadedModels: [ModelID: any EmbeddingModel] = [:]
    private let resourceManager: ResourceManager
    private let metricsCollector: MetricsCollector

    // MARK: - Initialization

    public init() {
        self.resourceManager = ResourceManager()
        self.metricsCollector = MetricsCollector()
    }

    // MARK: - Model Loading

    /// Load a model with full customization
    public func loadModel(
        _ spec: ModelSpecification,
        configuration: EmbeddingConfiguration = EmbeddingConfiguration()
    ) async throws -> any EmbeddingModel {

        // Check if already loaded
        if let existing = loadedModels[spec.id] {
            return existing
        }

        // Create model based on spec
        let model = try await createModel(spec: spec, configuration: configuration)

        // Warmup if requested
        if spec.preload {
            try await model.warmup()
        }

        // Store reference
        loadedModels[spec.id] = model

        return model
    }

    /// Load Apple's on-device model
    public func loadAppleModel(
        variant: AppleModelVariant = .base,
        configuration: EmbeddingConfiguration = EmbeddingConfiguration()
    ) async throws -> AppleEmbeddingModel {
        let spec = ModelSpecification(
            id: ModelID(
                provider: "apple",
                name: "text-embedding",
                version: "1.0.0",
                variant: variant.rawValue
            ),
            source: .system,
            format: .coreml,
            preload: true
        )

        let model = try await loadModel(spec, configuration: configuration)
        guard let appleModel = model as? AppleEmbeddingModel else {
            throw EmbedKitError.modelLoadFailed("Failed to cast to AppleEmbeddingModel")
        }
        return appleModel
    }

    /// Load a local CoreML model
    public func loadLocalModel(
        at url: URL,
        tokenizer: any Tokenizer,
        configuration: EmbeddingConfiguration = EmbeddingConfiguration()
    ) async throws -> LocalEmbeddingModel {
        let spec = ModelSpecification(
            id: ModelID(
                provider: "local",
                name: url.lastPathComponent,
                version: "1.0.0"
            ),
            source: .local(url),
            format: .coreml,
            preload: false
        )

        return try await LocalEmbeddingModel(
            modelURL: url,
            tokenizer: tokenizer,
            configuration: configuration
        )
    }

    // MARK: - Model Management

    /// Unload a model to free resources
    public func unloadModel(_ id: ModelID) async throws {
        guard let model = loadedModels[id] else { return }
        try await model.release()
        loadedModels.removeValue(forKey: id)
    }

    /// Get all loaded models
    public var loadedModelIDs: [ModelID] {
        Array(loadedModels.keys)
    }

    /// Get metrics for a specific model
    public func metrics(for id: ModelID) async throws -> ModelMetrics {
        guard let model = loadedModels[id] else {
            throw EmbedKitError.modelNotFound(id)
        }
        return await model.metrics
    }

    // MARK: - Direct Embedding Methods

    /// Generate embedding with specific model
    public func embed(
        _ text: String,
        using modelID: ModelID,
        configuration: EmbeddingConfiguration? = nil
    ) async throws -> Embedding {
        guard let model = loadedModels[modelID] else {
            throw EmbedKitError.modelNotFound(modelID)
        }

        return try await model.embed(text)
    }

    /// Batch embedding with optimal processing
    public func embedBatch(
        _ texts: [String],
        using modelID: ModelID,
        options: BatchOptions = BatchOptions()
    ) async throws -> [Embedding] {
        guard let model = loadedModels[modelID] else {
            throw EmbedKitError.modelNotFound(modelID)
        }

        return try await model.embedBatch(texts, options: options)
    }

    // MARK: - Private Helpers

    private func createModel(
        spec: ModelSpecification,
        configuration: EmbeddingConfiguration
    ) async throws -> any EmbeddingModel {
        switch spec.format {
        case .coreml:
            return try await createCoreMLModel(spec: spec, configuration: configuration)
        case .onnx:
            return try await createONNXModel(spec: spec, configuration: configuration)
        case .pytorch:
            throw EmbedKitError.modelLoadFailed("PyTorch models not yet supported")
        }
    }

    private func createCoreMLModel(
        spec: ModelSpecification,
        configuration: EmbeddingConfiguration
    ) async throws -> any EmbeddingModel {
        switch spec.source {
        case .system:
            // Apple's system model
            return try await AppleEmbeddingModel(configuration: configuration)

        case .local(let url):
            // Local CoreML model
            let tokenizer = try await detectTokenizer(for: spec)
            return try await LocalEmbeddingModel(
                modelURL: url,
                tokenizer: tokenizer,
                configuration: configuration
            )

        case .remote(let url):
            // Download and cache
            let localURL = try await resourceManager.downloadModel(from: url, id: spec.id)
            let tokenizer = try await detectTokenizer(for: spec)
            return try await LocalEmbeddingModel(
                modelURL: localURL,
                tokenizer: tokenizer,
                configuration: configuration
            )
        }
    }

    private func createONNXModel(
        spec: ModelSpecification,
        configuration: EmbeddingConfiguration
    ) async throws -> any EmbeddingModel {
        // TODO: Implement ONNX support
        throw EmbedKitError.modelLoadFailed("ONNX support coming soon")
    }

    private func detectTokenizer(for spec: ModelSpecification) async throws -> any Tokenizer {
        // Auto-detect tokenizer based on model metadata
        // For now, return a default
        return BertTokenizer()
    }
}

// ============================================================================
// MARK: - Model Specifications
// ============================================================================

public struct ModelSpecification: Sendable {
    public let id: ModelID
    public let source: ModelSource
    public let format: ModelFormat
    public let preload: Bool

    public init(
        id: ModelID,
        source: ModelSource,
        format: ModelFormat,
        preload: Bool = false
    ) {
        self.id = id
        self.source = source
        self.format = format
        self.preload = preload
    }
}

public enum ModelSource: Sendable {
    case system         // OS-provided model
    case local(URL)     // Local file
    case remote(URL)    // Download from URL
}

public enum ModelFormat: String, CaseIterable, Sendable {
    case coreml
    case onnx
    case pytorch
}

public enum AppleModelVariant: String, CaseIterable, Sendable {
    case base
    case large
    case multilingual
}

// ============================================================================
// MARK: - Concrete Model Implementations
// ============================================================================

/// Apple's on-device embedding model
public actor AppleEmbeddingModel: EmbeddingModel {

    public let id: ModelID
    public let dimensions: Int = 768
    public let device: ComputeDevice

    private let configuration: EmbeddingConfiguration
    private let backend: CoreMLBackend
    private let tokenizer: AppleTokenizer
    private var metricsData: MetricsData

    init(configuration: EmbeddingConfiguration) async throws {
        self.id = ModelID(
            provider: "apple",
            name: "text-embedding",
            version: "1.0.0",
            variant: "base"
        )
        self.configuration = configuration
        self.device = configuration.preferredDevice

        // Initialize components
        self.backend = try await CoreMLBackend(
            modelName: "AppleTextEmbedding",
            device: device
        )
        self.tokenizer = AppleTokenizer()
        self.metricsData = MetricsData()
    }

    public func embed(_ text: String) async throws -> Embedding {
        let startTime = Date()

        // Tokenize
        let tokenConfig = TokenizerConfig(
            maxLength: configuration.maxTokens,
            truncation: configuration.truncationStrategy,
            padding: configuration.paddingStrategy,
            addSpecialTokens: configuration.includeSpecialTokens
        )
        let tokenized = try await tokenizer.encode(text, config: tokenConfig)

        // Process through CoreML
        let output = try await backend.process(
            tokenized,
            options: ProcessingOptions(device: device)
        )

        // Pool and normalize
        let pooled = applyPooling(
            output,
            strategy: configuration.poolingStrategy,
            mask: tokenized.attentionMask
        )

        let vector = configuration.normalizeOutput ?
            normalize(pooled) : pooled

        // Update metrics
        metricsData.recordRequest(
            tokens: tokenized.length,
            latency: Date().timeIntervalSince(startTime)
        )

        return Embedding(
            vector: vector,
            metadata: EmbeddingMetadata(
                modelID: id,
                tokenCount: tokenized.length,
                processingTime: Date().timeIntervalSince(startTime),
                normalized: configuration.normalizeOutput,
                poolingStrategy: configuration.poolingStrategy,
                truncated: tokenized.length >= configuration.maxTokens
            )
        )
    }

    public func embedBatch(
        _ texts: [String],
        options: BatchOptions
    ) async throws -> [Embedding] {
        // Sort by length if requested for optimal padding
        let sortedTexts = options.sortByLength ?
            texts.sorted { $0.count < $1.count } : texts

        // Process in batches
        var allEmbeddings: [Embedding] = []
        for chunk in sortedTexts.chunked(into: options.maxBatchSize) {
            let batchEmbeddings = try await processBatch(chunk)
            allEmbeddings.append(contentsOf: batchEmbeddings)
        }

        // Restore original order if sorted
        if options.sortByLength {
            // TODO: Track and restore original indices
        }

        return allEmbeddings
    }

    public func warmup() async throws {
        try await backend.load()
        _ = try await embed("warmup")
    }

    public func release() async throws {
        try await backend.unload()
    }

    public var metrics: ModelMetrics {
        metricsData.summary
    }

    // MARK: - Private Helpers

    private func processBatch(_ texts: [String]) async throws -> [Embedding] {
        // Parallel tokenization
        let tokenized = try await withThrowingTaskGroup(of: TokenizedText.self) { group in
            for text in texts {
                group.addTask {
                    try await self.tokenizer.encode(
                        text,
                        config: TokenizerConfig(
                            maxLength: self.configuration.maxTokens
                        )
                    )
                }
            }

            var results: [TokenizedText] = []
            for try await token in group {
                results.append(token)
            }
            return results
        }

        // Batch processing through backend
        let outputs = try await backend.processBatch(
            tokenized,
            options: ProcessingOptions(device: device)
        )

        // Convert to embeddings
        return zip(tokenized, outputs).map { tokenized, output in
            let pooled = applyPooling(
                output,
                strategy: configuration.poolingStrategy,
                mask: tokenized.attentionMask
            )

            let vector = configuration.normalizeOutput ?
                normalize(pooled) : pooled

            return Embedding(
                vector: vector,
                metadata: EmbeddingMetadata(
                    modelID: id,
                    tokenCount: tokenized.length,
                    processingTime: 0,
                    normalized: configuration.normalizeOutput,
                    poolingStrategy: configuration.poolingStrategy,
                    truncated: tokenized.length >= configuration.maxTokens
                )
            )
        }
    }

    private func applyPooling(
        _ output: [Float],
        strategy: PoolingStrategy,
        mask: [Int]
    ) -> [Float] {
        // Implementation depends on strategy
        switch strategy {
        case .mean:
            return meanPool(output, mask: mask)
        case .cls:
            return clsPool(output)
        case .max:
            return maxPool(output, mask: mask)
        default:
            return output
        }
    }

    private func meanPool(_ output: [Float], mask: [Int]) -> [Float] {
        // Implement mean pooling with attention mask
        output
    }

    private func clsPool(_ output: [Float]) -> [Float] {
        // Return first token embedding
        output
    }

    private func maxPool(_ output: [Float], mask: [Int]) -> [Float] {
        // Implement max pooling
        output
    }

    private func normalize(_ vector: [Float]) -> [Float] {
        let magnitude = sqrt(vector.reduce(0) { $0 + $1 * $1 })
        guard magnitude > 0 else { return vector }
        return vector.map { $0 / magnitude }
    }
}

/// Local model implementation
public actor LocalEmbeddingModel: EmbeddingModel {
    public let id: ModelID
    public let dimensions: Int
    public let device: ComputeDevice

    private let modelURL: URL
    private let tokenizer: any Tokenizer
    private let configuration: EmbeddingConfiguration
    private let backend: CoreMLBackend
    private var metricsData: MetricsData

    public init(
        modelURL: URL,
        tokenizer: any Tokenizer,
        configuration: EmbeddingConfiguration
    ) async throws {
        self.id = ModelID(
            provider: "local",
            name: modelURL.lastPathComponent,
            version: "1.0.0"
        )
        self.modelURL = modelURL
        self.tokenizer = tokenizer
        self.configuration = configuration
        self.device = configuration.preferredDevice

        // Load model
        self.backend = try await CoreMLBackend(
            modelURL: modelURL,
            device: device
        )

        // Detect dimensions from model
        self.dimensions = try await backend.outputDimensions
        self.metricsData = MetricsData()
    }

    public func embed(_ text: String) async throws -> Embedding {
        // Similar implementation to AppleEmbeddingModel
        fatalError("Implementation needed")
    }

    public func embedBatch(
        _ texts: [String],
        options: BatchOptions
    ) async throws -> [Embedding] {
        fatalError("Implementation needed")
    }

    public func warmup() async throws {
        try await backend.load()
    }

    public func release() async throws {
        try await backend.unload()
    }

    public var metrics: ModelMetrics {
        metricsData.summary
    }
}

// ============================================================================
// MARK: - Supporting Types
// ============================================================================

struct MetricsData {
    private var requests: Int = 0
    private var totalTokens: Int = 0
    private var latencies: [TimeInterval] = []

    mutating func recordRequest(tokens: Int, latency: TimeInterval) {
        requests += 1
        totalTokens += tokens
        latencies.append(latency)
    }

    var summary: ModelMetrics {
        ModelMetrics(
            totalRequests: requests,
            totalTokensProcessed: totalTokens,
            averageLatency: latencies.isEmpty ? 0 : latencies.reduce(0, +) / Double(latencies.count),
            p95Latency: latencies.sorted().last ?? 0,
            throughput: 0,
            cacheHitRate: 0,
            memoryUsage: 0,
            lastUsed: Date()
        )
    }
}

// Placeholder implementations
struct CoreMLBackend: ModelBackend {
    typealias Input = TokenizedText
    typealias Output = [Float]

    let modelName: String?
    let modelURL: URL?
    let device: ComputeDevice
    var isLoaded: Bool = false
    var memoryUsage: Int64 = 0
    var outputDimensions: Int = 768

    init(modelName: String, device: ComputeDevice) async throws {
        self.modelName = modelName
        self.modelURL = nil
        self.device = device
    }

    init(modelURL: URL, device: ComputeDevice) async throws {
        self.modelURL = modelURL
        self.modelName = nil
        self.device = device
    }

    func process(_ input: TokenizedText, options: ProcessingOptions) async throws -> [Float] {
        // CoreML processing
        Array(repeating: 0.0, count: outputDimensions)
    }

    func processBatch(_ inputs: [TokenizedText], options: ProcessingOptions) async throws -> [[Float]] {
        try await inputs.asyncMap { try await process($0, options: options) }
    }

    func load() async throws {}
    func unload() async throws {}
}

struct ResourceManager {
    func downloadModel(from url: URL, id: ModelID) async throws -> URL {
        // Download and cache implementation
        url
    }
}

struct MetricsCollector {}

// Tokenizer placeholders
struct AppleTokenizer: Tokenizer {
    var vocabularySize: Int = 30000
    var specialTokens: SpecialTokens = SpecialTokens(
        cls: nil, sep: nil, pad: nil, unk: nil,
        mask: nil, bos: nil, eos: nil
    )

    func encode(_ text: String, config: TokenizerConfig) async throws -> TokenizedText {
        TokenizedText(
            ids: [101, 102],
            tokens: ["[CLS]", "[SEP]"],
            attentionMask: [1, 1],
            typeIds: nil,
            specialTokenMask: [true, true],
            offsets: nil
        )
    }

    func decode(_ ids: [Int]) async throws -> String { "" }
}

struct BertTokenizer: Tokenizer {
    var vocabularySize: Int = 30000
    var specialTokens: SpecialTokens = SpecialTokens(
        cls: nil, sep: nil, pad: nil, unk: nil,
        mask: nil, bos: nil, eos: nil
    )

    func encode(_ text: String, config: TokenizerConfig) async throws -> TokenizedText {
        TokenizedText(
            ids: [], tokens: [], attentionMask: [],
            typeIds: nil, specialTokenMask: [], offsets: nil
        )
    }

    func decode(_ ids: [Int]) async throws -> String { "" }
}

// Helper extensions
extension Array {
    func chunked(into size: Int) -> [[Element]] {
        stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }

    func asyncMap<T>(_ transform: (Element) async throws -> T) async rethrows -> [T] {
        var results: [T] = []
        for element in self {
            results.append(try await transform(element))
        }
        return results
    }
}