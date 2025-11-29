// EmbedKit - Local CoreML Model Adapter

import Foundation

/// Convenience wrapper to load a local CoreML model with a provided tokenizer,
/// delegating all embedding logic to AppleEmbeddingModel.
public actor LocalCoreMLModel: EmbeddingModel {
    public nonisolated let id: ModelID
    public nonisolated let dimensions: Int
    public nonisolated let device: ComputeDevice

    private let inner: AppleEmbeddingModel

    public init(
        modelURL: URL,
        tokenizer: any Tokenizer,
        configuration: EmbeddingConfiguration = EmbeddingConfiguration(),
        id: ModelID? = nil,
        dimensions: Int = 384,
        device: ComputeDevice? = nil,
        profiler: Profiler? = nil
    ) {
        let backend = CoreMLBackend(modelURL: modelURL, device: configuration.inferenceDevice)
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: configuration,
            id: id,
            dimensions: dimensions,
            device: device ?? configuration.inferenceDevice,
            profiler: profiler
        )
        self.inner = model
        self.id = model.id
        self.dimensions = dimensions
        self.device = device ?? configuration.inferenceDevice
    }

    public func embed(_ text: String) async throws -> Embedding { try await inner.embed(text) }
    public func embedBatch(_ texts: [String], options: BatchOptions) async throws -> [Embedding] { try await inner.embedBatch(texts, options: options) }
    public func warmup() async throws { try await inner.warmup() }
    public func release() async throws { try await inner.release() }
    public var metrics: ModelMetrics { get async { await inner.metrics } }
    public func resetMetrics() async throws { try await inner.resetMetrics() }
    public var stageMetricsSnapshot: StageMetrics { get async {await inner.stageMetricsSnapshot } }

    // MARK: - CoreML I/O override convenience
    public func setCoreMLInputKeyOverrides(token: String? = nil, mask: String? = nil, type: String? = nil, pos: String? = nil) async {
        await inner.setCoreMLInputKeyOverrides(token: token, mask: mask, type: type, pos: pos)
    }
    public func setCoreMLOutputKeyOverride(_ key: String?) async {
        await inner.setCoreMLOutputKeyOverride(key)
    }
}
