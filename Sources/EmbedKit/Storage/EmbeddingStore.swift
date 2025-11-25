// EmbedKit - Embedding Store
// Main storage actor wrapping VectorIndex

import Foundation
import VectorCore
import VectorIndex

// MARK: - Embedding Store

/// Actor for storing and searching embeddings using VectorIndex.
///
/// `EmbeddingStore` provides a high-level interface for semantic search operations,
/// wrapping VectorIndex's low-level index implementations with embedding-aware
/// functionality.
///
/// ## Example Usage
/// ```swift
/// // Create a store with HNSW index
/// let store = try await EmbeddingStore(
///     config: .default(dimension: 384),
///     model: myEmbeddingModel
/// )
///
/// // Store text directly (embedding computed automatically)
/// let stored = try await store.store(text: "Hello world")
///
/// // Search with text query
/// let results = try await store.search(text: "greeting", k: 5)
///
/// // Or search with pre-computed embedding
/// let embedding = try await model.embed("query")
/// let results = try await store.search(embedding, k: 5)
/// ```
public actor EmbeddingStore: EmbeddingStorable {

    // MARK: - Properties

    /// The underlying vector index.
    private let index: any VectorIndexProtocol

    /// Configuration used to create this store.
    public nonisolated let config: IndexConfiguration

    /// Optional embedding model for automatic text embedding.
    private let model: (any EmbeddingModel)?

    /// Stored text lookup (id -> text).
    private var textStore: [String: String] = [:]

    /// Stored embeddings lookup (id -> embedding) for reranking.
    private var embeddingStore: [String: Embedding] = [:]

    /// Acceleration manager for GPU-accelerated operations.
    private var accelerator: AccelerationManager?

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

        // Create the appropriate index type
        switch config.indexType {
        case .flat:
            self.index = FlatIndex(dimension: config.dimension, metric: config.metric)
        case .hnsw:
            let hnswConfig = config.hnswConfig ?? .default
            self.index = HNSWIndex(
                dimension: config.dimension,
                metric: config.metric,
                config: hnswConfig.toVectorIndex
            )
        case .ivf:
            let ivfConfig = config.ivfConfig ?? .default
            self.index = IVFIndex(
                dimension: config.dimension,
                metric: config.metric,
                config: ivfConfig.toVectorIndex
            )
        }

        // Initialize accelerator if not CPU-only
        if config.computePreference != .cpuOnly {
            self.accelerator = await AccelerationManager(preference: config.computePreference)
        }
    }

    /// Create an embedding store from an existing VectorIndex.
    /// - Parameters:
    ///   - index: Existing vector index
    ///   - config: Optional configuration (uses defaults if not provided)
    ///   - model: Optional embedding model
    public init<I: VectorIndexProtocol>(
        index: I,
        config: IndexConfiguration? = nil,
        model: (any EmbeddingModel)? = nil
    ) async {
        self.index = index
        self.model = model

        // Compute default config if not provided
        if let providedConfig = config {
            self.config = providedConfig
        } else {
            self.config = IndexConfiguration(
                indexType: .flat, // We don't know the actual type
                dimension: await index.dimension,
                metric: await index.metric
            )
        }

        // Initialize accelerator if not CPU-only
        if self.config.computePreference != .cpuOnly {
            self.accelerator = await AccelerationManager(preference: self.config.computePreference)
        }
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

        // Store in index
        try await index.insert(
            id: vectorId,
            vector: embedding.vector,
            metadata: metadata
        )

        // Store text if configured
        if config.storeText, let text = text {
            textStore[vectorId] = text
        }

        // Store embedding for potential reranking
        embeddingStore[vectorId] = embedding

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

        let results = try await index.search(
            query: query.vector,
            k: k,
            filter: filter
        )

        return results.map { result in
            EmbeddingSearchResult(
                from: result,
                text: textStore[result.id],
                metadata: nil, // VectorIndex stores this internally
                embedding: embeddingStore[result.id],
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

        let queryEmbeddings = try await model.embedBatch(texts, options: BatchOptions())

        var results: [[EmbeddingSearchResult]] = []
        results.reserveCapacity(texts.count)

        for embedding in queryEmbeddings {
            let searchResults = try await search(embedding, k: k, filter: filter)
            results.append(searchResults)
        }

        return results
    }

    // MARK: - Management Operations

    /// Remove an embedding by ID.
    public func remove(id: String) async throws {
        try await index.remove(id: id)
        textStore.removeValue(forKey: id)
        embeddingStore.removeValue(forKey: id)
    }

    /// Check if an embedding exists.
    public func contains(id: String) async -> Bool {
        await index.contains(id: id)
    }

    /// Clear all stored embeddings.
    public func clear() async throws {
        await index.clear()
        textStore.removeAll()
        embeddingStore.removeAll()
    }

    /// Optimize the index for better search performance.
    public func optimize() async throws {
        try await index.optimize()
    }

    /// Get index statistics.
    public func statistics() async -> IndexStats {
        await index.statistics()
    }

    // MARK: - Persistence

    /// Save the store to a directory.
    public func save(to directory: URL) async throws {
        // Create directory if needed
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )

        // Save index
        let indexURL = directory.appendingPathComponent("index.json")
        try await index.save(to: indexURL)

        // Save text store
        if !textStore.isEmpty {
            let textURL = directory.appendingPathComponent("texts.json")
            let textData = try JSONEncoder().encode(textStore)
            try textData.write(to: textURL)
        }

        // Save config
        let configURL = directory.appendingPathComponent("config.json")
        let configData = try JSONEncoder().encode(StoredConfig(from: config))
        try configData.write(to: configURL)
    }

    /// Load a store from a directory.
    public static func load(
        from directory: URL,
        model: (any EmbeddingModel)? = nil
    ) async throws -> EmbeddingStore {
        // Load config
        let configURL = directory.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let storedConfig = try JSONDecoder().decode(StoredConfig.self, from: configData)
        let config = storedConfig.toConfig()

        // Load index
        let indexURL = directory.appendingPathComponent("index.json")

        let store: EmbeddingStore
        switch config.indexType {
        case .flat:
            let index = try await FlatIndex.load(from: indexURL)
            store = await EmbeddingStore(index: index, config: config, model: model)
        case .hnsw:
            let index = try await HNSWIndex.load(from: indexURL)
            store = await EmbeddingStore(index: index, config: config, model: model)
        case .ivf:
            let index = try await IVFIndex.load(from: indexURL)
            store = await EmbeddingStore(index: index, config: config, model: model)
        }

        // Load text store if exists
        let textURL = directory.appendingPathComponent("texts.json")
        if FileManager.default.fileExists(atPath: textURL.path) {
            let textData = try Data(contentsOf: textURL)
            let loadedTexts = try JSONDecoder().decode([String: String].self, from: textData)
            await store.setTextStore(loadedTexts)
        }

        return store
    }

    /// Internal method to set text store (for loading).
    private func setTextStore(_ texts: [String: String]) {
        self.textStore = texts
    }

    // MARK: - Acceleration

    /// Whether GPU acceleration is available for this store.
    public var isAccelerationAvailable: Bool {
        get async {
            guard let accelerator = accelerator else { return false }
            return await accelerator.isGPUAvailable
        }
    }

    /// Get acceleration statistics.
    public func accelerationStatistics() async -> AccelerationStatistics? {
        await accelerator?.statistics()
    }

    /// Compute batch distances with acceleration (for custom operations).
    public func computeDistances(
        from query: Embedding,
        to candidates: [Embedding]
    ) async throws -> [Float] {
        guard let accelerator = accelerator else {
            // CPU fallback - compute cosine distances
            return candidates.map { candidate in
                1.0 - query.similarity(to: candidate)
            }
        }

        return try await accelerator.batchDistance(
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
    let storeText: Bool
    let computePreference: String?

    init(from config: IndexConfiguration) {
        self.indexType = config.indexType.rawValue
        self.dimension = config.dimension
        self.metric = config.metric.rawValue
        self.storeText = config.storeText
        self.computePreference = config.computePreference.rawValue
    }

    func toConfig() -> IndexConfiguration {
        IndexConfiguration(
            indexType: IndexType(rawValue: indexType) ?? .flat,
            dimension: dimension,
            metric: SupportedDistanceMetric(rawValue: metric) ?? .cosine,
            storeText: storeText,
            computePreference: computePreference.flatMap { ComputePreference(rawValue: $0) } ?? .auto
        )
    }
}

// MARK: - Errors

/// Errors from embedding store operations.
public enum EmbeddingStoreError: Error, LocalizedError {
    case noModelConfigured
    case dimensionMismatch(expected: Int, actual: Int)
    case indexNotFound
    case persistenceError(String)

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
        }
    }
}

// MARK: - Safe Array Access

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
