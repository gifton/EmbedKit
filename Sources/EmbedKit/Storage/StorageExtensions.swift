// EmbedKit - Storage Extensions
// Convenience extensions for seamless storage integration

import Foundation
import VectorCore
import VectorIndex

// MARK: - Embedding + Storage

public extension Embedding {
    /// Store this embedding in an embedding store.
    /// - Parameters:
    ///   - store: The store to save to
    ///   - id: Optional identifier (auto-generated if nil)
    ///   - text: Original text that produced this embedding
    ///   - metadata: Additional metadata
    /// - Returns: The stored embedding record
    @discardableResult
    func store(
        in store: EmbeddingStore,
        id: String? = nil,
        text: String? = nil,
        metadata: [String: String]? = nil
    ) async throws -> StoredEmbedding {
        try await store.store(self, id: id, text: text, metadata: metadata)
    }

    /// Find similar embeddings in a store.
    /// - Parameters:
    ///   - store: The store to search
    ///   - k: Number of results to return
    /// - Returns: Search results sorted by similarity
    func findSimilar(
        in store: EmbeddingStore,
        k: Int = 10
    ) async throws -> [EmbeddingSearchResult] {
        try await store.search(self, k: k)
    }
}

// MARK: - EmbeddingModel + Storage

public extension EmbeddingModel {
    /// Create an embedding store backed by this model.
    /// - Parameter config: Index configuration (defaults to HNSW with model's dimensions)
    /// - Returns: Configured embedding store
    func createStore(
        config: IndexConfiguration? = nil
    ) async throws -> EmbeddingStore {
        let storeConfig = config ?? .default(dimension: dimensions)
        return try await EmbeddingStore(config: storeConfig, model: self)
    }

    /// Create a flat (exact search) embedding store.
    func createFlatStore() async throws -> EmbeddingStore {
        try await EmbeddingStore(config: .exact(dimension: dimensions), model: self)
    }

    /// Create an HNSW (fast approximate) embedding store.
    func createHNSWStore(config: HNSWConfiguration = .default) async throws -> EmbeddingStore {
        let storeConfig = IndexConfiguration(
            indexType: .hnsw,
            dimension: dimensions,
            metric: .cosine,
            storeText: true,
            hnswConfig: config
        )
        return try await EmbeddingStore(config: storeConfig, model: self)
    }
}

// MARK: - Array + Batch Storage

public extension Array where Element == String {
    /// Store all texts in an embedding store.
    /// - Parameters:
    ///   - store: The store to save to
    ///   - metadata: Optional metadata for each text
    /// - Returns: Stored embedding records
    @discardableResult
    func store(
        in store: EmbeddingStore,
        metadata: [[String: String]?]? = nil
    ) async throws -> [StoredEmbedding] {
        try await store.storeBatch(texts: self, metadata: metadata)
    }
}

public extension Array where Element == Embedding {
    /// Store all embeddings in an embedding store.
    /// - Parameters:
    ///   - store: The store to save to
    ///   - texts: Original texts (optional)
    ///   - metadata: Metadata for each embedding
    /// - Returns: Stored embedding records
    @discardableResult
    func store(
        in store: EmbeddingStore,
        texts: [String]? = nil,
        metadata: [[String: String]?]? = nil
    ) async throws -> [StoredEmbedding] {
        try await store.storeBatch(self, texts: texts, metadata: metadata)
    }
}

// MARK: - Semantic Search Convenience

public extension EmbeddingStore {
    /// Find the most similar item to a text query.
    /// - Parameter text: Query text
    /// - Returns: Best matching result, or nil if store is empty
    func findMostSimilar(to text: String) async throws -> EmbeddingSearchResult? {
        let results = try await search(text: text, k: 1)
        return results.first
    }

    /// Find the most similar item to an embedding.
    /// - Parameter embedding: Query embedding
    /// - Returns: Best matching result, or nil if store is empty
    func findMostSimilar(to embedding: Embedding) async throws -> EmbeddingSearchResult? {
        let results = try await search(embedding, k: 1)
        return results.first
    }

    /// Check if a similar item exists above a threshold.
    /// - Parameters:
    ///   - text: Text to check
    ///   - threshold: Minimum similarity (0-1)
    /// - Returns: True if a similar item exists
    func containsSimilar(
        to text: String,
        threshold: Float = 0.9
    ) async throws -> Bool {
        guard let result = try await findMostSimilar(to: text) else {
            return false
        }
        return result.similarity >= threshold
    }

    /// Get all texts similar to a query, sorted by similarity.
    /// - Parameters:
    ///   - text: Query text
    ///   - minSimilarity: Minimum similarity threshold
    ///   - limit: Maximum results to return
    /// - Returns: Similar texts with their similarity scores
    func findSimilarTexts(
        to text: String,
        minSimilarity: Float = 0.5,
        limit: Int = 100
    ) async throws -> [(text: String, similarity: Float)] {
        let results = try await search(text: text, k: limit)
        return results
            .filter { $0.similarity >= minSimilarity && $0.text != nil }
            .map { (text: $0.text!, similarity: $0.similarity) }
    }
}

// MARK: - Quick Store Factory

/// Quick factory for creating embedding stores.
public enum EmbeddingStores {
    /// Create an in-memory store with flat index.
    public static func inMemory(
        dimension: Int,
        model: (any EmbeddingModel)? = nil
    ) async throws -> EmbeddingStore {
        try await EmbeddingStore(
            config: .exact(dimension: dimension),
            model: model
        )
    }

    /// Create a fast HNSW-backed store.
    public static func fast(
        dimension: Int,
        model: (any EmbeddingModel)? = nil
    ) async throws -> EmbeddingStore {
        try await EmbeddingStore(
            config: .fast(dimension: dimension),
            model: model
        )
    }

    /// Create a scalable IVF-backed store for large datasets.
    public static func scalable(
        dimension: Int,
        expectedSize: Int = 100_000,
        model: (any EmbeddingModel)? = nil
    ) async throws -> EmbeddingStore {
        try await EmbeddingStore(
            config: .scalable(dimension: dimension, expectedSize: expectedSize),
            model: model
        )
    }
}
