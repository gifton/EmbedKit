// EmbedKitV2 - Model Manager (Week 1)

import Foundation
import Logging

public actor ModelManager {
    private let logger = Logger(label: "EmbedKitV2.ModelManager")
    private var loadedModels: [ModelID: any EmbeddingModel] = [:]

    public init() {}

    // Week 1: Provide a stable API that returns a mock model
    public func loadAppleModel(
        configuration: EmbeddingConfiguration = EmbeddingConfiguration()
    ) async throws -> any EmbeddingModel {
        logger.debug("loadAppleModel() -> mock (Week 1)")
        return try await loadMockModel(configuration: configuration)
    }

    public func loadMockModel(
        configuration: EmbeddingConfiguration = EmbeddingConfiguration()
    ) async throws -> any EmbeddingModel {
        let model = MockEmbeddingModel(configuration: configuration)
        loadedModels[model.id] = model
        logger.info("Loaded model: \(model.id)")
        return model
    }

    // Week 2: Toggle real vs mock Apple model (scaffold for real)
    public func loadAppleModel(real useReal: Bool,
                               configuration: EmbeddingConfiguration = EmbeddingConfiguration(),
                               dimensions: Int = 384) async throws -> any EmbeddingModel {
        if useReal {
            // Scaffold: wire tokenizer + backend without implementing embed yet
            let tokenizer = SimpleTokenizer()
            let backend = CoreMLBackendV2(modelURL: nil, device: configuration.preferredDevice)
            let model = AppleEmbeddingModelV2(
                backend: backend,
                tokenizer: tokenizer,
                configuration: configuration,
                dimensions: dimensions,
                device: configuration.preferredDevice
            )
            loadedModels[model.id] = model
            logger.info("Loaded AppleEmbeddingModelV2 (scaffold): \(model.id)")
            return model
        } else {
            return try await loadAppleModel(configuration: configuration)
        }
    }

    public func unloadModel(_ id: ModelID) async {
        loadedModels.removeValue(forKey: id)
        logger.info("Unloaded model: \(id)")
    }

    // MARK: - Direct Embedding (for EmbedBench)

    public func embed(
        _ text: String,
        using modelID: ModelID
    ) async throws -> (embedding: Embedding, metrics: ModelMetrics) {
        guard let model = loadedModels[modelID] else {
            throw EmbedKitError.modelNotFound(modelID)
        }
        let embedding = try await model.embed(text)
        let metrics = await model.metrics
        return (embedding, metrics)
    }

    public struct BatchResult: Sendable {
        public let embeddings: [Embedding]
        public let totalTime: TimeInterval
        public let perItemTimes: [TimeInterval]
        public let tokenCounts: [Int]
    }

    public func embedBatch(
        _ texts: [String],
        using modelID: ModelID,
        options: BatchOptions = BatchOptions()
    ) async throws -> BatchResult {
        guard let model = loadedModels[modelID] else {
            throw EmbedKitError.modelNotFound(modelID)
        }

        let indices: [Int]
        if options.sortByLength {
            indices = texts.indices.sorted { texts[$0].count < texts[$1].count }
        } else {
            indices = Array(texts.indices)
        }

        var embeddings: [Embedding?] = Array(repeating: nil, count: texts.count)
        var perItemTimes: [TimeInterval] = Array(repeating: 0, count: texts.count)
        var tokenCounts: [Int] = Array(repeating: 0, count: texts.count)

        let start = CFAbsoluteTimeGetCurrent()
        for idx in indices {
            let t0 = CFAbsoluteTimeGetCurrent()
            let text = texts[idx]
            let e = try await model.embed(text)
            let t1 = CFAbsoluteTimeGetCurrent()
            embeddings[idx] = e
            perItemTimes[idx] = t1 - t0
            tokenCounts[idx] = text.count
        }
        let total = CFAbsoluteTimeGetCurrent() - start

        return BatchResult(
            embeddings: embeddings.compactMap { $0 },
            totalTime: total,
            perItemTimes: perItemTimes,
            tokenCounts: tokenCounts
        )
    }

    // MARK: - Spec Scaffolding (future weeks)

    public struct ModelSpecification: Sendable {
        public let id: ModelID
        public let source: ModelSource
        public let format: ModelFormat
        public let preload: Bool

        public init(id: ModelID, source: ModelSource, format: ModelFormat, preload: Bool = false) {
            self.id = id; self.source = source; self.format = format; self.preload = preload
        }
    }

    public enum ModelSource: Sendable { case system, local(URL), remote(URL) }
    public enum ModelFormat: String, Sendable { case coreml, onnx, pytorch }

    // MARK: - Ergonomics

    public var loadedModelIDs: [ModelID] {
        Array(loadedModels.keys)
    }

    public func unloadAll() async {
        loadedModels.removeAll()
        logger.info("Unloaded all models")
    }

    public func resetMetrics(for id: ModelID) async throws {
        guard let model = loadedModels[id] else {
            throw EmbedKitError.modelNotFound(id)
        }
        try await model.resetMetrics()
        logger.debug("Reset metrics for \(id)")
    }

    public func metrics(for id: ModelID) async throws -> ModelMetrics {
        guard let model = loadedModels[id] else {
            throw EmbedKitError.modelNotFound(id)
        }
        return await model.metrics
    }
}
