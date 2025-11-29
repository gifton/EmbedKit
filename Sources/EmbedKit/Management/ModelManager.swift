// EmbedKit - Model Manager

import Foundation
import Logging

public actor ModelManager {
    private let logger = Logger(label: "EmbedKit.ModelManager")
    private var profiler: Profiler? = nil
    private var loadedModels: [ModelID: any EmbeddingModel] = [:]
#if canImport(Darwin)
    private var memSource: DispatchSourceMemoryPressure?
#endif
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
            let backend = CoreMLBackend(modelURL: nil, device: configuration.inferenceDevice)
            let model = AppleEmbeddingModel(
                backend: backend,
                tokenizer: tokenizer,
                configuration: configuration,
                dimensions: dimensions,
                device: configuration.inferenceDevice
            )
            loadedModels[model.id] = model
            logger.info("Loaded AppleEmbeddingModel (scaffold): \(model.id)")
            return model
        } else {
            return try await loadAppleModel(configuration: configuration)
        }
    }

    // MARK: - Week 2: Real Apple model convenience loaders

    /// Load AppleEmbeddingModel with explicit CoreML model URL and a provided tokenizer.
    /// No hidden fallbacks: throws if backend cannot load at runtime.
    public func loadAppleModel(
        modelURL: URL,
        tokenizer: any Tokenizer,
        id: ModelID? = nil,
        configuration: EmbeddingConfiguration = EmbeddingConfiguration(),
        dimensions: Int = 384,
        preload: Bool = false
    ) async throws -> any EmbeddingModel {
        let backend = CoreMLBackend(modelURL: modelURL, device: configuration.inferenceDevice)
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: configuration,
            id: id,
            dimensions: dimensions,
            device: configuration.inferenceDevice,
            profiler: profiler
        )
        loadedModels[model.id] = model
        logger.info("Loaded AppleEmbeddingModel: \(model.id)")
        if preload { try await model.warmup() }
        return model
    }

    /// Load AppleEmbeddingModel with WordPieceTokenizer using a vocabulary file.
    /// The vocabulary is mapped 1:1 by line order. Lowercasing and unk token are configurable.
    public func loadAppleModel(
        modelURL: URL,
        vocabularyURL: URL,
        lowercase: Bool = true,
        unkToken: String = "[UNK]",
        id: ModelID? = nil,
        configuration: EmbeddingConfiguration = EmbeddingConfiguration(),
        dimensions: Int = 384,
        preload: Bool = false
    ) async throws -> any EmbeddingModel {
        let vocab = try Vocabulary.load(from: vocabularyURL)
        let tokenizer = WordPieceTokenizer(vocabulary: vocab, unkToken: unkToken, lowercase: lowercase)
        return try await loadAppleModel(
            modelURL: modelURL,
            tokenizer: tokenizer,
            id: id ?? deriveID(from: modelURL),
            configuration: configuration,
            dimensions: dimensions,
            preload: preload
        )
    }

    // MARK: - Helpers

    private func deriveID(from url: URL) -> ModelID {
        let name = url.deletingPathExtension().lastPathComponent
        return ModelID(provider: "apple", name: name, version: "1.0.0")
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
        let t0 = CFAbsoluteTimeGetCurrent()
        let embs = try await model.embedBatch(texts, options: options)
        let total = CFAbsoluteTimeGetCurrent() - t0
        let n = max(1, embs.count)
        let perItemTimes = Array(repeating: total / Double(n), count: n)
        let tokenCounts = embs.map { $0.metadata.tokenCount }
        return BatchResult(embeddings: embs, totalTime: total, perItemTimes: perItemTimes, tokenCounts: tokenCounts)
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

    // MARK: - CoreML I/O override convenience
    public func setCoreMLInputKeyOverrides(
        for id: ModelID,
        token: String? = nil,
        mask: String? = nil,
        type: String? = nil,
        pos: String? = nil
    ) async throws {
        guard let model = loadedModels[id] else { throw EmbedKitError.modelNotFound(id) }
        if let v2 = model as? AppleEmbeddingModel {
            await v2.setCoreMLInputKeyOverrides(token: token, mask: mask, type: type, pos: pos)
        }
    }

    public func setCoreMLOutputKeyOverride(for id: ModelID, key: String?) async throws {
        guard let model = loadedModels[id] else { throw EmbedKitError.modelNotFound(id) }
        if let v2 = model as? AppleEmbeddingModel {
            await v2.setCoreMLOutputKeyOverride(key)
        }
    }

    // MARK: - NLContextual convenience loader
    public func loadNLContextualEmbedding(
        language: String = "en",
        id: ModelID? = nil,
        configuration: EmbeddingConfiguration = EmbeddingConfiguration(),
        preload: Bool = false
    ) async throws -> any EmbeddingModel {
        let model = try AppleNLContextualModel(
            language: language,
            configuration: configuration,
            id: id ?? ModelID(provider: "apple", name: "nl-contextual", version: "system"),
            profiler: profiler
        )
        loadedModels[model.id] = model
        logger.info("Loaded AppleNLContextualModel: \(model.id)")
        if preload { try await model.warmup() }
        return model
    }


    // MARK: - Utilities
    /// Register an externally constructed model instance.
    public func register(_ model: any EmbeddingModel) async {
        loadedModels[model.id] = model
        logger.info("Registered model: \(model.id)")
    }

    /// Trim memory for a loaded model (clears caches; optionally unloads backend).
    public func trimMemory(for id: ModelID, aggressive: Bool = false) async throws {
        guard let model = loadedModels[id] else { throw EmbedKitError.modelNotFound(id) }
        if let v2 = model as? AppleEmbeddingModel {
            await v2.trimMemory(aggressive: aggressive)
        }
    }

    // MARK: - Memory pressure handling
    public enum MemoryPressureLevel { case normal, warning, critical }

    public func handleMemoryPressure(_ level: MemoryPressureLevel) async {
        logger.warning("Memory pressure: \(level)")
        for id in loadedModels.keys {
            try? await trimMemory(for: id, aggressive: level == .critical)
        }
    }

    // MARK: - Profiler control
    public func setProfiler(_ profiler: Profiler?) {
        self.profiler = profiler
    }

    #if canImport(Darwin)
    public func startMemoryPressureMonitoring() {
        guard memSource == nil else { return }
        let source = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical], queue: .global())
        source.setEventHandler { [weak self] in
            let mask = source.mask
            Task { [weak self] in
                if mask.contains(.critical) { await self?.handleMemoryPressure(.critical) }
                else { await self?.handleMemoryPressure(.warning) }
            }
        }
        source.resume()
        memSource = source
        logger.info("Memory pressure monitoring started")
    }

    public func stopMemoryPressureMonitoring() {
        memSource?.cancel()
        memSource = nil
        logger.info("Memory pressure monitoring stopped")
    }
    #endif

    /// For tests: simulate memory pressure event.
    public func simulateMemoryPressure(_ level: MemoryPressureLevel) async {
        await handleMemoryPressure(level)
    }

    // MARK: - EmbeddingGenerator Factory

    /// Creates an EmbeddingGenerator for a loaded model.
    ///
    /// The generator conforms to `VectorProducer` for seamless integration
    /// with VectorIndex and other VSK components.
    ///
    /// - Parameters:
    ///   - modelID: ID of a previously loaded model
    ///   - config: Generator configuration (default: `.default`)
    /// - Returns: An `EmbeddingGenerator` wrapping the model
    /// - Throws: `EmbedKitError.modelNotFound` if model is not loaded
    ///
    /// ## Example
    /// ```swift
    /// let model = try await manager.loadAppleModel(modelURL: url, vocabularyURL: vocabURL)
    /// let generator = try await manager.createGenerator(for: model.id)
    ///
    /// // Use as VectorProducer
    /// let vectors = try await generator.produce(["Hello", "World"])
    /// ```
    public func createGenerator(
        for modelID: ModelID,
        config: GeneratorConfiguration = .default
    ) async throws -> EmbeddingGenerator {
        guard let model = loadedModels[modelID] else {
            throw EmbedKitError.modelNotFound(modelID)
        }

        logger.info("Creating generator for model: \(modelID)")
        return EmbeddingGenerator(
            model: model,
            configuration: config.embedding,
            batchOptions: config.batch
        )
    }

    /// Creates an EmbeddingGenerator with a newly loaded Apple model.
    ///
    /// Convenience method that loads a model and creates a generator in one call.
    ///
    /// - Parameters:
    ///   - modelURL: URL to the CoreML model
    ///   - vocabularyURL: URL to the vocabulary file
    ///   - config: Generator configuration (default: `.default`)
    ///   - preload: Whether to warmup the model (default: false)
    /// - Returns: An `EmbeddingGenerator` ready for use
    ///
    /// ## Example
    /// ```swift
    /// let generator = try await manager.createGenerator(
    ///     modelURL: Bundle.main.url(forResource: "model", withExtension: "mlpackage")!,
    ///     vocabularyURL: Bundle.main.url(forResource: "vocab", withExtension: "txt")!,
    ///     config: .forSemanticSearch()
    /// )
    ///
    /// // Generate embeddings with progress
    /// for try await (vector, progress) in generator.generateWithProgress(texts) {
    ///     print("Progress: \(progress.percentage)%")
    /// }
    /// ```
    public func createGenerator(
        modelURL: URL,
        vocabularyURL: URL,
        config: GeneratorConfiguration = .default,
        preload: Bool = false
    ) async throws -> EmbeddingGenerator {
        let model = try await loadAppleModel(
            modelURL: modelURL,
            vocabularyURL: vocabularyURL,
            configuration: config.embedding,
            preload: preload
        )

        logger.info("Created generator with new model: \(model.id)")
        return EmbeddingGenerator(
            model: model,
            configuration: config.embedding,
            batchOptions: config.batch
        )
    }

    /// Creates an EmbeddingGenerator using the NLContextualEmbedding system model.
    ///
    /// Uses Apple's built-in contextual embedding model (no external files needed).
    ///
    /// - Parameters:
    ///   - language: Language code (default: "en")
    ///   - config: Generator configuration (default: `.default`)
    ///   - preload: Whether to warmup the model (default: false)
    /// - Returns: An `EmbeddingGenerator` using the system model
    ///
    /// ## Example
    /// ```swift
    /// let generator = try await manager.createSystemGenerator()
    /// let vectors = try await generator.produce(sentences)
    /// ```
    public func createSystemGenerator(
        language: String = "en",
        config: GeneratorConfiguration = .default,
        preload: Bool = false
    ) async throws -> EmbeddingGenerator {
        let model = try await loadNLContextualEmbedding(
            language: language,
            configuration: config.embedding,
            preload: preload
        )

        logger.info("Created system generator: \(model.id)")
        return EmbeddingGenerator(
            model: model,
            configuration: config.embedding,
            batchOptions: config.batch
        )
    }
}
