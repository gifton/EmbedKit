// EmbedKit - Embedding Store
// GPU-accelerated storage using VectorAccelerate's AcceleratedVectorIndex

import Foundation
import VectorCore
import VectorAccelerate

// MARK: - Embedding Store

/// Actor for GPU-accelerated embedding storage and search.
///
/// `EmbeddingStore` provides a high-level interface for semantic search operations,
/// using VectorAccelerate's `AcceleratedVectorIndex` for GPU-first vector search
/// with sub-millisecond latency.
///
/// ## Example Usage
/// ```swift
/// // Create a store with flat GPU index
/// let store = try await EmbeddingStore(
///     config: .flat(dimension: 384),
///     model: myEmbeddingModel
/// )
///
/// // Store text directly (embedding computed automatically)
/// let stored = try await store.store(text: "Hello world")
///
/// // Search with text query
/// let results = try await store.search(text: "greeting", k: 5)
///
/// // For IVF index, call train() after inserting vectors
/// try await store.train()
/// ```
public actor EmbeddingStore: EmbeddingStorable {

    // MARK: - Properties

    /// The GPU-accelerated vector index.
    private let index: AcceleratedVectorIndex

    /// Configuration used to create this store.
    public nonisolated let config: IndexConfiguration

    /// Optional embedding model for automatic text embedding.
    private let model: (any EmbeddingModel)?

    /// VectorHandle to String ID mapping.
    private var handleToID: [VectorHandle: String] = [:]

    /// String ID to VectorHandle mapping.
    private var idToHandle: [String: VectorHandle] = [:]

    /// Stored text lookup (id -> text).
    private var textStore: [String: String] = [:]

    /// Stored embeddings lookup (id -> embedding) for reranking.
    private var embeddingStore: [String: Embedding] = [:]

    /// Stored metadata lookup (id -> metadata).
    private var metadataStore: [String: [String: String]] = [:]

    /// Number of stored embeddings.
    public var count: Int {
        get async {
            await index.count
        }
    }

    // MARK: - Initialization

    /// Create an embedding store with the specified configuration.
    /// - Parameters:
    ///   - config: Index configuration
    ///   - model: Optional embedding model for automatic text embedding
    public init(
        config: IndexConfiguration,
        model: (any EmbeddingModel)? = nil
    ) async throws {
        self.config = config
        self.model = model

        // Validate configuration
        try config.validate()

        // Create GPU-accelerated index
        let vaConfig = config.toVectorAccelerate()
        self.index = try await AcceleratedVectorIndex(configuration: vaConfig)
    }

    /// Create an embedding store with a custom Metal4Context.
    /// - Parameters:
    ///   - config: Index configuration
    ///   - context: Shared Metal4Context for GPU operations
    ///   - model: Optional embedding model
    public init(
        config: IndexConfiguration,
        context: Metal4Context,
        model: (any EmbeddingModel)? = nil
    ) async throws {
        self.config = config
        self.model = model

        try config.validate()

        let vaConfig = config.toVectorAccelerate()
        self.index = try await AcceleratedVectorIndex(configuration: vaConfig, context: context)
    }

    // MARK: - Store Operations

    /// Store an embedding with optional metadata.
    public func store(
        _ embedding: Embedding,
        id: String? = nil,
        text: String? = nil,
        metadata: [String: String]? = nil
    ) async throws -> StoredEmbedding {
        let vectorId = id ?? UUID().uuidString

        // Validate dimension
        guard embedding.vector.count == config.dimension else {
            throw EmbeddingStoreError.dimensionMismatch(
                expected: config.dimension,
                actual: embedding.vector.count
            )
        }

        // Remove existing entry if ID already exists (overwrite semantics)
        if let existingHandle = idToHandle[vectorId] {
            try await index.remove(existingHandle)
            handleToID.removeValue(forKey: existingHandle)
            idToHandle.removeValue(forKey: vectorId)
            textStore.removeValue(forKey: vectorId)
            embeddingStore.removeValue(forKey: vectorId)
            metadataStore.removeValue(forKey: vectorId)
        }

        // Convert metadata to VectorAccelerate format
        let vaMetadata: VectorMetadata? = metadata

        // Insert into GPU index
        let handle = try await index.insert(embedding.vector, metadata: vaMetadata)

        // Store mappings
        handleToID[handle] = vectorId
        idToHandle[vectorId] = handle

        // Store text if configured
        if config.storeText, let text = text {
            textStore[vectorId] = text
        }

        // Store embedding for potential reranking
        embeddingStore[vectorId] = embedding

        // Store metadata
        if let metadata = metadata {
            metadataStore[vectorId] = metadata
        }

        return StoredEmbedding(
            id: vectorId,
            embedding: embedding,
            text: text,
            metadata: metadata
        )
    }

    /// Store text by computing its embedding automatically.
    /// Requires a model to be set.
    public func store(
        text: String,
        id: String? = nil,
        metadata: [String: String]? = nil
    ) async throws -> StoredEmbedding {
        guard let model = model else {
            throw EmbeddingStoreError.noModelConfigured
        }

        let embedding = try await model.embed(text)
        return try await store(embedding, id: id, text: text, metadata: metadata)
    }

    // MARK: - Search Operations

    /// Search for similar embeddings.
    public func search(
        _ query: Embedding,
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil
    ) async throws -> [EmbeddingSearchResult] {
        guard k > 0 else { return [] }

        // Validate dimension
        guard query.vector.count == config.dimension else {
            throw EmbeddingStoreError.dimensionMismatch(
                expected: config.dimension,
                actual: query.vector.count
            )
        }

        let results: [IndexSearchResult]
        if let filter = filter {
            // Use filtered search with predicate
            results = try await index.search(query: query.vector, k: k) { [metadataStore, handleToID] handle, vaMetadata in
                // Get our string ID for this handle
                guard let id = handleToID[handle] else { return true }
                // Get metadata from our store (more complete than VA metadata)
                let metadata = metadataStore[id] ?? vaMetadata
                return filter(metadata)
            }
        } else {
            results = try await index.search(query: query.vector, k: k)
        }

        return results.compactMap { result in
            guard let id = handleToID[result.handle] else { return nil }
            return EmbeddingSearchResult(
                id: id,
                distance: result.distance,
                text: textStore[id],
                metadata: metadataStore[id],
                embedding: embeddingStore[id],
                metric: config.metric
            )
        }
    }

    /// Search with text query (requires model).
    public func search(
        text: String,
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil
    ) async throws -> [EmbeddingSearchResult] {
        guard let model = model else {
            throw EmbeddingStoreError.noModelConfigured
        }

        let queryEmbedding = try await model.embed(text)
        return try await search(queryEmbedding, k: k, filter: filter)
    }

    /// Search with reranking.
    public func search(
        _ query: Embedding,
        k: Int,
        rerank: some RerankingStrategy,
        options: RerankOptions = .default,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil
    ) async throws -> [EmbeddingSearchResult] {
        // Fetch more candidates for reranking
        let candidateK = k * options.candidateMultiplier
        let candidates = try await search(query, k: candidateK, filter: filter)

        // Apply reranking
        var results = try await rerank.rerank(query: query, candidates: candidates, k: k)

        // Apply minimum similarity filter if specified
        if let minSim = options.minSimilarity {
            results = results.filter { $0.similarity >= minSim }
        }

        return results
    }

    /// Search with text query and reranking.
    public func search(
        text: String,
        k: Int,
        rerank: some RerankingStrategy,
        options: RerankOptions = .default,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil
    ) async throws -> [EmbeddingSearchResult] {
        guard let model = model else {
            throw EmbeddingStoreError.noModelConfigured
        }

        let queryEmbedding = try await model.embed(text)
        return try await search(queryEmbedding, k: k, rerank: rerank, options: options, filter: filter)
    }

    // MARK: - Batch Operations

    /// Store multiple texts in batch.
    public func storeBatch(
        texts: [String],
        ids: [String]? = nil,
        metadata: [[String: String]?]? = nil
    ) async throws -> [StoredEmbedding] {
        guard let model = model else {
            throw EmbeddingStoreError.noModelConfigured
        }

        // Compute embeddings in batch
        let embeddings = try await model.embedBatch(texts, options: BatchOptions())

        var results: [StoredEmbedding] = []
        results.reserveCapacity(texts.count)

        for (index, embedding) in embeddings.enumerated() {
            let id = ids?[safe: index]
            let text = texts[index]
            let meta = metadata?[safe: index] ?? nil
            let stored = try await store(embedding, id: id, text: text, metadata: meta)
            results.append(stored)
        }

        return results
    }

    /// Batch search with multiple queries.
    public func batchSearch(
        texts: [String],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil
    ) async throws -> [[EmbeddingSearchResult]] {
        guard let model = model else {
            throw EmbeddingStoreError.noModelConfigured
        }
        guard k > 0 else { return texts.map { _ in [] } }

        let queryEmbeddings = try await model.embedBatch(texts, options: BatchOptions())

        // Search each query
        var results: [[EmbeddingSearchResult]] = []
        results.reserveCapacity(queryEmbeddings.count)

        for queryEmbedding in queryEmbeddings {
            let queryResults = try await search(queryEmbedding, k: k, filter: filter)
            results.append(queryResults)
        }

        return results
    }

    // MARK: - Management Operations

    /// Remove an embedding by ID.
    public func remove(id: String) async throws {
        guard let handle = idToHandle[id] else {
            return // Already removed or never existed
        }

        try await index.remove(handle)

        // Clean up mappings
        handleToID.removeValue(forKey: handle)
        idToHandle.removeValue(forKey: id)
        textStore.removeValue(forKey: id)
        embeddingStore.removeValue(forKey: id)
        metadataStore.removeValue(forKey: id)
    }

    /// Check if an embedding exists.
    public func contains(id: String) async -> Bool {
        idToHandle[id] != nil
    }

    /// Clear all stored embeddings.
    public func clear() async throws {
        // Remove all handles from index
        for handle in handleToID.keys {
            try await index.remove(handle)
        }

        // Clear all mappings
        handleToID.removeAll()
        idToHandle.removeAll()
        textStore.removeAll()
        embeddingStore.removeAll()
        metadataStore.removeAll()
    }

    // MARK: - IVF Training

    /// Train the IVF index.
    ///
    /// For IVF indexes, this builds the K-means cluster centroids.
    /// Call this after inserting a representative sample of vectors.
    ///
    /// - Note: Has no effect on flat indexes.
    public func train() async throws {
        guard config.indexType == .ivf else { return }
        try await index.train()
    }

    /// Whether the IVF index is trained.
    public var isTrained: Bool {
        get async {
            let stats = await index.statistics()
            return stats.ivfStats?.isTrained ?? true
        }
    }

    // MARK: - Compaction

    /// Compact the index to reclaim space from deleted vectors.
    ///
    /// With VectorAccelerate 0.3.2+ stable handles, `VectorHandle` values remain
    /// valid across compaction. No handle remapping is needed - the underlying
    /// index maintains an internal indirection table.
    public func compact() async throws {
        try await index.compact()
    }

    // MARK: - Statistics

    /// Get GPU index statistics.
    public func statistics() async -> GPUIndexStats {
        await index.statistics()
    }

    /// Get memory usage in bytes.
    public var memoryUsage: Int {
        get async {
            let stats = await index.statistics()
            return stats.gpuMemoryBytes
        }
    }

    // MARK: - Persistence

    /// Save the store's mappings and text data to a directory.
    ///
    /// **Note:** The GPU index vectors are NOT saved. On reload, you must
    /// re-insert all vectors. This saves only:
    /// - Configuration
    /// - ID mappings
    /// - Text store
    /// - Metadata store
    /// - Embeddings (so they can be re-inserted)
    public func save(to directory: URL) async throws {
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )

        // Save config
        let configURL = directory.appendingPathComponent("config.json")
        let storedConfig = StoredConfig(from: config)
        let configData = try JSONEncoder().encode(storedConfig)
        try configData.write(to: configURL)

        // Save text store
        if !textStore.isEmpty {
            let textURL = directory.appendingPathComponent("texts.json")
            let textData = try JSONEncoder().encode(textStore)
            try textData.write(to: textURL)
        }

        // Save metadata store
        if !metadataStore.isEmpty {
            let metaURL = directory.appendingPathComponent("metadata.json")
            let metaData = try JSONEncoder().encode(metadataStore)
            try metaData.write(to: metaURL)
        }

        // Save embeddings (required for re-insertion)
        if !embeddingStore.isEmpty {
            let embURL = directory.appendingPathComponent("embeddings.json")
            let embeddingsDict = embeddingStore.mapValues { $0.vector }
            let embData = try JSONEncoder().encode(embeddingsDict)
            try embData.write(to: embURL)
        }
    }

    /// Load a store from a directory.
    ///
    /// This creates a new GPU index and re-inserts all saved embeddings.
    public static func load(
        from directory: URL,
        model: (any EmbeddingModel)? = nil
    ) async throws -> EmbeddingStore {
        // Load config
        let configURL = directory.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let storedConfig = try JSONDecoder().decode(StoredConfig.self, from: configData)
        let config = storedConfig.toConfig()

        // Create new store
        let store = try await EmbeddingStore(config: config, model: model)

        // Load text store
        let textURL = directory.appendingPathComponent("texts.json")
        if FileManager.default.fileExists(atPath: textURL.path) {
            let textData = try Data(contentsOf: textURL)
            let texts = try JSONDecoder().decode([String: String].self, from: textData)
            await store.setTextStore(texts)
        }

        // Load metadata store
        let metaURL = directory.appendingPathComponent("metadata.json")
        if FileManager.default.fileExists(atPath: metaURL.path) {
            let metaData = try Data(contentsOf: metaURL)
            let metadata = try JSONDecoder().decode([String: [String: String]].self, from: metaData)
            await store.setMetadataStore(metadata)
        }

        // Load and re-insert embeddings
        let embURL = directory.appendingPathComponent("embeddings.json")
        if FileManager.default.fileExists(atPath: embURL.path) {
            let embData = try Data(contentsOf: embURL)
            let embeddingsDict = try JSONDecoder().decode([String: [Float]].self, from: embData)

            // Placeholder ModelID for loaded embeddings
            let loadedModelID = ModelID(provider: "embedkit", name: "loaded", version: "1.0")

            for (id, vector) in embeddingsDict {
                let embMetadata = EmbeddingMetadata(
                    modelID: loadedModelID,
                    tokenCount: 0,
                    processingTime: 0
                )
                let embedding = Embedding(vector: vector, metadata: embMetadata)
                let text = await store.getText(for: id)
                let metadata = await store.getMetadata(for: id)
                _ = try await store.store(embedding, id: id, text: text, metadata: metadata)
            }
        }

        // Train if IVF
        if config.indexType == .ivf {
            try await store.train()
        }

        return store
    }

    // MARK: - Private Helpers

    private func setTextStore(_ texts: [String: String]) {
        self.textStore = texts
    }

    private func setMetadataStore(_ metadata: [String: [String: String]]) {
        self.metadataStore = metadata
    }

    private func getText(for id: String) -> String? {
        textStore[id]
    }

    private func getMetadata(for id: String) -> [String: String]? {
        metadataStore[id]
    }

    // MARK: - Acceleration (Convenience)

    /// Whether GPU acceleration is available.
    /// Always true for EmbeddingStore since it requires GPU.
    public nonisolated var isAccelerationAvailable: Bool {
        true
    }

    /// Compute batch distances using GPU.
    public func computeDistances(
        from query: Embedding,
        to candidates: [Embedding]
    ) async throws -> [Float] {
        // Use AccelerationManager for standalone distance computation
        let manager = try await AccelerationManager.create()
        return try await manager.batchDistance(
            from: query.vector,
            to: candidates.map { $0.vector },
            metric: config.metric
        )
    }
}

// MARK: - Stored Config (for persistence)

private struct StoredConfig: Codable {
    let indexType: String
    let dimension: Int
    let metric: String
    let capacity: Int
    let storeText: Bool
    let nlist: Int?
    let nprobe: Int?

    init(from config: IndexConfiguration) {
        self.indexType = config.indexType.rawValue
        self.dimension = config.dimension
        self.metric = config.metric.rawValue
        self.capacity = config.capacity
        self.storeText = config.storeText
        self.nlist = config.nlist
        self.nprobe = config.nprobe
    }

    func toConfig() -> IndexConfiguration {
        IndexConfiguration(
            indexType: IndexType(rawValue: indexType) ?? .flat,
            dimension: dimension,
            metric: SupportedDistanceMetric(rawValue: metric) ?? .cosine,
            capacity: capacity,
            storeText: storeText,
            nlist: nlist,
            nprobe: nprobe
        )
    }
}

// MARK: - Errors

/// Errors from embedding store operations.
public enum EmbeddingStoreError: Error, LocalizedError, Sendable {
    case noModelConfigured
    case dimensionMismatch(expected: Int, actual: Int)
    case indexNotFound
    case persistenceError(String)
    case gpuInitializationFailed(Error)

    public var errorDescription: String? {
        switch self {
        case .noModelConfigured:
            return "No embedding model configured. Provide a model or use pre-computed embeddings."
        case .dimensionMismatch(let expected, let actual):
            return "Dimension mismatch: expected \(expected), got \(actual)"
        case .indexNotFound:
            return "Index not found at specified path"
        case .persistenceError(let msg):
            return "Persistence error: \(msg)"
        case .gpuInitializationFailed(let error):
            return "GPU initialization failed: \(error.localizedDescription)"
        }
    }
}

// MARK: - Safe Array Access

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
