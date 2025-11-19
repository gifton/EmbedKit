// EmbedKitV2 - Apple Embedding Model (Scaffold)

import Foundation

public actor AppleEmbeddingModelV2: EmbeddingModel {
    // MARK: - Identity
    public nonisolated let id: ModelID
    public nonisolated let dimensions: Int
    public nonisolated let device: ComputeDevice

    // MARK: - Dependencies
    private let backend: CoreMLBackendV2?
    private let tokenizer: any Tokenizer
    private let configuration: EmbeddingConfiguration

    // MARK: - Metrics
    private var metricsData = MetricsData()

    // MARK: - Init
    public init(
        backend: CoreMLBackendV2?,
        tokenizer: any Tokenizer,
        configuration: EmbeddingConfiguration = EmbeddingConfiguration(),
        id: ModelID? = nil,
        dimensions: Int = 384,
        device: ComputeDevice? = nil
    ) {
        self.backend = backend
        self.tokenizer = tokenizer
        self.configuration = configuration
        self.dimensions = dimensions
        self.device = device ?? configuration.preferredDevice
        self.id = id ?? ModelID(provider: "apple", name: "text-embedding", version: "1.0.0", variant: "base")
    }

    // MARK: - EmbeddingModel
    public func embed(_ text: String) async throws -> Embedding {
        throw EmbedKitError.modelLoadFailed("AppleEmbeddingModelV2.embed not implemented (scaffold)")
    }

    public func embedBatch(_ texts: [String], options: BatchOptions) async throws -> [Embedding] {
        throw EmbedKitError.modelLoadFailed("AppleEmbeddingModelV2.embedBatch not implemented (scaffold)")
    }

    public func warmup() async throws {
        try await backend?.load()
    }

    public func release() async throws {
        try await backend?.unload()
    }

    public var metrics: ModelMetrics { metricsData.snapshot(memoryUsage: currentMemoryUsage()) }

    public func resetMetrics() async throws { metricsData = MetricsData() }

    // MARK: - Input Builder (Scaffold)
    /// Build CoreML input from text using the configured tokenizer.
    /// Note: This is an internal helper for scaffolding; pooling and inference are not wired yet.
    func buildInput(for text: String) async throws -> CoreMLInputV2 {
        var tk = TokenizerConfig()
        tk.maxLength = configuration.maxTokens
        tk.truncation = configuration.truncationStrategy
        tk.padding = configuration.paddingStrategy
        tk.addSpecialTokens = configuration.includeSpecialTokens

        let tokenized = try await tokenizer.encode(text, config: tk)
        return CoreMLInputV2(tokenIDs: tokenized.ids, attentionMask: tokenized.attentionMask)
    }
}
