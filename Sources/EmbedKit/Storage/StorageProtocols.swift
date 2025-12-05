// EmbedKit - Storage Protocols
// Protocols for embedding storage and retrieval

import Foundation
import VectorCore

// MARK: - Embedding Storable Protocol

/// Protocol for types that can store and retrieve embeddings.
///
/// Implement this protocol to create custom storage backends
/// (e.g., cloud vector databases like Pinecone, Weaviate, etc.)
public protocol EmbeddingStorable: Actor {
    /// Store an embedding with optional metadata.
    /// - Parameters:
    ///   - embedding: The embedding to store
    ///   - id: Optional identifier (auto-generated if nil)
    ///   - text: Original text that was embedded
    ///   - metadata: Additional metadata to store
    /// - Returns: The stored embedding record
    func store(
        _ embedding: Embedding,
        id: String?,
        text: String?,
        metadata: [String: String]?
    ) async throws -> StoredEmbedding

    /// Search for similar embeddings.
    /// - Parameters:
    ///   - query: Query embedding
    ///   - k: Number of results to return
    ///   - filter: Optional metadata filter
    /// - Returns: Search results sorted by relevance
    func search(
        _ query: Embedding,
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)?
    ) async throws -> [EmbeddingSearchResult]

    /// Remove an embedding by ID.
    func remove(id: String) async throws

    /// Check if an embedding exists.
    func contains(id: String) async -> Bool

    /// Get the total count of stored embeddings.
    var count: Int { get async }

    /// Clear all stored embeddings.
    func clear() async throws
}

// MARK: - Batch Operations Extension

public extension EmbeddingStorable {
    /// Store multiple embeddings in batch.
    func storeBatch(
        _ embeddings: [Embedding],
        ids: [String]? = nil,
        texts: [String]? = nil,
        metadata: [[String: String]?]? = nil
    ) async throws -> [StoredEmbedding] {
        var results: [StoredEmbedding] = []
        results.reserveCapacity(embeddings.count)

        for (index, embedding) in embeddings.enumerated() {
            let id = ids?[safe: index]
            let text = texts?[safe: index]
            let meta = metadata?[safe: index] ?? nil
            let stored = try await store(embedding, id: id, text: text, metadata: meta)
            results.append(stored)
        }

        return results
    }

    /// Search with multiple queries.
    func batchSearch(
        _ queries: [Embedding],
        k: Int,
        filter: (@Sendable ([String: String]?) -> Bool)? = nil
    ) async throws -> [[EmbeddingSearchResult]] {
        var results: [[EmbeddingSearchResult]] = []
        results.reserveCapacity(queries.count)

        for query in queries {
            let searchResults = try await search(query, k: k, filter: filter)
            results.append(searchResults)
        }

        return results
    }

    /// Remove multiple embeddings.
    func removeBatch(_ ids: [String]) async throws {
        for id in ids {
            try await remove(id: id)
        }
    }
}

// MARK: - Stored Embedding

/// Represents an embedding that has been stored in an index.
public struct StoredEmbedding: Sendable, Identifiable, Equatable {
    /// Unique identifier for this stored embedding
    public let id: String

    /// The embedding vector
    public let embedding: Embedding

    /// Original text that was embedded (if stored)
    public let text: String?

    /// Additional metadata
    public let metadata: [String: String]?

    /// Timestamp when stored
    public let storedAt: Date

    public init(
        id: String,
        embedding: Embedding,
        text: String? = nil,
        metadata: [String: String]? = nil,
        storedAt: Date = Date()
    ) {
        self.id = id
        self.embedding = embedding
        self.text = text
        self.metadata = metadata
        self.storedAt = storedAt
    }

    public static func == (lhs: StoredEmbedding, rhs: StoredEmbedding) -> Bool {
        lhs.id == rhs.id
    }
}

// MARK: - Persistable Storage

/// Protocol for storage backends that support persistence.
public protocol PersistableStorage: EmbeddingStorable {
    /// Save the index to a file.
    func save(to url: URL) async throws

    /// Load an index from a file.
    static func load(from url: URL) async throws -> Self
}

// MARK: - Safe Array Access

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
