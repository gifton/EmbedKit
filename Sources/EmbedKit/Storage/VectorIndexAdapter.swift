//
//  VectorIndexAdapter.swift
//  EmbedKit
//
//  Integration adapter for connecting EmbedKit with VectorIndex for storage and retrieval
//  Refactored to remove backward compatibility - clean API only
//

import Foundation
import VectorCore

/// Configuration for VectorIndex integration
public struct VectorIndexConfiguration: Sendable {
    /// Distance metric to use for similarity search
    public enum DistanceMetric: String, Sendable {
        case cosine
        case euclidean
        case dotProduct
    }

    public let distanceMetric: DistanceMetric
    public let maxVectors: Int
    public let indexType: String  // e.g., "hnsw", "flat", "ivf"

    public init(
        distanceMetric: DistanceMetric = .cosine,
        maxVectors: Int = 100_000,
        indexType: String = "flat"
    ) {
        self.distanceMetric = distanceMetric
        self.maxVectors = maxVectors
        self.indexType = indexType
    }
}

/// Search result from vector index
public struct VectorSearchResult: Sendable {
    public let id: String  // Changed from UUID to String for flexibility
    public var score: Float  // Made mutable for reranking
    public let metadata: [String: String]
    public let embedding: DynamicEmbedding?

    public init(
        id: String,
        score: Float,
        metadata: [String: String],
        embedding: DynamicEmbedding? = nil
    ) {
        self.id = id
        self.score = score
        self.metadata = metadata
        self.embedding = embedding
    }
}

/// Metadata associated with stored vectors
public struct VectorMetadata: Sendable {
    public let text: String
    public let timestamp: Date
    public let additionalData: [String: String]

    public init(
        text: String,
        timestamp: Date = Date(),
        additionalData: [String: String] = [:]
    ) {
        self.text = text
        self.timestamp = timestamp
        self.additionalData = additionalData
    }
}

/// Protocol for vector storage backends
public protocol VectorStorageBackend: Actor {
    /// Add a vector to the index
    func add(
        vector: [Float],
        metadata: [String: String]
    ) async throws -> UUID

    /// Add multiple vectors in batch
    func addBatch(
        vectors: [[Float]],
        metadata: [[String: String]]
    ) async throws -> [UUID]

    /// Search for similar vectors
    func search(
        query: [Float],
        k: Int,
        threshold: Float?
    ) async throws -> [(id: UUID, score: Float, metadata: [String: String])]

    /// Remove a vector by ID
    func remove(id: UUID) async throws

    /// Get vector by ID
    func get(id: UUID) async throws -> (vector: [Float], metadata: [String: String])?

    /// Get total count of vectors
    func count() async -> Int

    /// Clear all vectors
    func clear() async throws
}

/// Adapter for integrating EmbedKit with VectorIndex or other storage backends
public actor VectorIndexAdapter {
    private let pipeline: EmbeddingPipeline
    private let storage: any VectorStorageBackend
    private let configuration: VectorIndexConfiguration

    /// Statistics
    private var totalVectorsAdded: Int = 0
    private var totalSearches: Int = 0

    // MARK: - Initialization

    /// Initialize with pipeline and storage backend
    public init(
        pipeline: EmbeddingPipeline,
        storage: any VectorStorageBackend,
        configuration: VectorIndexConfiguration = VectorIndexConfiguration()
    ) {
        self.pipeline = pipeline
        self.storage = storage
        self.configuration = configuration
    }

    // MARK: - Text Storage

    /// Add a text entry with automatic embedding generation
    public func addText(
        _ text: String,
        metadata: VectorMetadata? = nil
    ) async throws -> UUID {
        // Generate embedding
        let embedding = try await pipeline.embed(text)

        // Prepare metadata
        var meta: [String: String] = [
            "text": text,
            "timestamp": String(metadata?.timestamp.timeIntervalSince1970 ?? Date().timeIntervalSince1970),
            "dimension": String(embedding.dimensions)
        ]

        // Add additional metadata
        if let additional = metadata?.additionalData {
            for (key, value) in additional {
                meta[key] = value
            }
        }

        // Store in index
        let id = try await storage.add(
            vector: embedding.toFloatArray(),
            metadata: meta
        )

        totalVectorsAdded += 1
        return id
    }

    /// Add multiple text entries in batch
    public func addTexts(
        _ texts: [String],
        metadata: [VectorMetadata?] = []
    ) async throws -> [UUID] {
        // Generate embeddings in batch
        let embeddings = try await pipeline.embed(batch: texts)

        // Prepare vectors and metadata
        var vectors: [[Float]] = []
        var metadataArray: [[String: String]] = []

        for (index, (text, embedding)) in zip(texts, embeddings).enumerated() {
            vectors.append(embedding.toFloatArray())

            var meta: [String: String] = [
                "text": text,
                "timestamp": String(Date().timeIntervalSince1970),
                "dimension": String(embedding.dimensions)
            ]

            // Add custom metadata if provided
            if index < metadata.count, let customMeta = metadata[index] {
                meta["timestamp"] = String(customMeta.timestamp.timeIntervalSince1970)
                for (key, value) in customMeta.additionalData {
                    meta[key] = value
                }
            }

            metadataArray.append(meta)
        }

        // Store in batch
        let ids = try await storage.addBatch(
            vectors: vectors,
            metadata: metadataArray
        )

        totalVectorsAdded += ids.count
        return ids
    }

    // MARK: - Direct Embedding Storage

    /// Add a pre-computed embedding
    public func addEmbedding(
        _ embedding: DynamicEmbedding,
        metadata: [String: String]
    ) async throws -> UUID {
        var meta = metadata
        meta["dimension"] = String(embedding.dimensions)
        meta["timestamp"] = String(Date().timeIntervalSince1970)

        let id = try await storage.add(
            vector: embedding.toFloatArray(),
            metadata: meta
        )

        totalVectorsAdded += 1
        return id
    }

    /// Add multiple pre-computed embeddings
    public func addEmbeddings(
        _ embeddings: [DynamicEmbedding],
        metadata: [[String: String]] = []
    ) async throws -> [UUID] {
        var vectors: [[Float]] = []
        var metadataArray: [[String: String]] = []

        for (index, embedding) in embeddings.enumerated() {
            vectors.append(embedding.toFloatArray())

            var meta: [String: String] = index < metadata.count ? metadata[index] : [:]
            meta["dimension"] = String(embedding.dimensions)
            meta["timestamp"] = String(Date().timeIntervalSince1970)

            metadataArray.append(meta)
        }

        let ids = try await storage.addBatch(
            vectors: vectors,
            metadata: metadataArray
        )

        totalVectorsAdded += ids.count
        return ids
    }

    // MARK: - Search (New Clean API)

    /// Search with optional reranking strategy
    public func semanticSearch(
        query: String,
        k: Int = 10,
        rerankStrategy: (any RerankingStrategy)? = nil,
        rerankOptions: RerankOptions = .default,
        filter: [String: Any]? = nil
    ) async throws -> [VectorSearchResult] {
        // Embed query
        let queryEmbedding = try await pipeline.embed(query)

        // Determine search size based on reranking
        let searchK = rerankStrategy != nil
            ? k * rerankOptions.candidateMultiplier
            : k

        // Initial search
        var results = try await searchByVector(
            queryEmbedding,
            k: searchK,
            filter: filter
        )

        // Apply reranking if strategy provided
        if let reranker = rerankStrategy {
            results = try await reranker.rerank(
                query: queryEmbedding,
                candidates: results,
                k: k,
                options: rerankOptions
            )
        }

        totalSearches += 1
        return results
    }

    /// Batch search with optional reranking
    public func batchSearch(
        queries: [String],
        k: Int = 10,
        rerankStrategy: (any RerankingStrategy)? = nil,
        rerankOptions: RerankOptions = .default
    ) async throws -> [[VectorSearchResult]] {
        let embeddings = try await pipeline.embed(batch: queries)

        return try await withThrowingTaskGroup(of: (Int, [VectorSearchResult]).self) { group in
            for (index, embedding) in embeddings.enumerated() {
                group.addTask {
                    let searchK = rerankStrategy != nil
                        ? k * rerankOptions.candidateMultiplier
                        : k

                    var results = try await self.searchByVector(embedding, k: searchK)

                    if let reranker = rerankStrategy {
                        results = try await reranker.rerank(
                            query: embedding,
                            candidates: results,
                            k: k,
                            options: rerankOptions
                        )
                    }

                    return (index, results)
                }
            }

            var allResults = Array(repeating: [VectorSearchResult](), count: queries.count)
            for try await (index, results) in group {
                allResults[index] = results
            }
            return allResults
        }
    }

    /// Search by text query (basic, no reranking)
    public func searchByText(
        _ query: String,
        k: Int = 10,
        threshold: Float? = nil,
        includeEmbeddings: Bool = false
    ) async throws -> [VectorSearchResult] {
        let queryEmbedding = try await pipeline.embed(query)
        return try await searchByVector(
            queryEmbedding,
            k: k,
            threshold: threshold,
            filter: nil,
            includeEmbeddings: includeEmbeddings
        )
    }

    /// Search by embedding vector
    public func searchByVector(
        _ embedding: DynamicEmbedding,
        k: Int = 10,
        threshold: Float? = nil,
        filter: [String: Any]? = nil,
        includeEmbeddings: Bool = false
    ) async throws -> [VectorSearchResult] {
        // Perform search
        let results = try await storage.search(
            query: embedding.toFloatArray(),
            k: k,
            threshold: threshold
        )

        totalSearches += 1

        // Convert to VectorSearchResult
        var searchResults: [VectorSearchResult] = []
        for result in results {
            var resultEmbedding: DynamicEmbedding? = nil

            if includeEmbeddings {
                // Retrieve full vector if requested
                if let stored = try await storage.get(id: result.id) {
                    resultEmbedding = try? DynamicEmbedding(values: stored.vector)
                }
            }

            searchResults.append(VectorSearchResult(
                id: result.id.uuidString,  // Convert UUID to String
                score: result.score,
                metadata: result.metadata,
                embedding: resultEmbedding
            ))
        }

        // Apply filter if provided (simple implementation)
        if let filter = filter {
            searchResults = searchResults.filter { result in
                for (key, value) in filter {
                    guard let metaValue = result.metadata[key as! String],
                          metaValue == value as! String else {
                        return false
                    }
                }
                return true
            }
        }

        return searchResults
    }

    // MARK: - Management

    /// Remove a vector by ID
    public func remove(id: UUID) async throws {
        try await storage.remove(id: id)
    }

    /// Get a specific vector and its metadata
    public func get(id: UUID) async throws -> (embedding: DynamicEmbedding, metadata: [String: String])? {
        guard let stored = try await storage.get(id: id) else {
            return nil
        }

        let embedding = try DynamicEmbedding(values: stored.vector)
        return (embedding, stored.metadata)
    }

    /// Get total count of vectors
    public func count() async -> Int {
        return await storage.count()
    }

    /// Clear all vectors
    public func clear() async throws {
        try await storage.clear()
        totalVectorsAdded = 0
    }

    // MARK: - Statistics

    /// Get adapter statistics
    public func getStatistics() async -> (vectorsAdded: Int, searches: Int, currentCount: Int) {
        return (
            vectorsAdded: totalVectorsAdded,
            searches: totalSearches,
            currentCount: await count()
        )
    }

    // MARK: - Helpers

    /// Get the distance metric as SupportedDistanceMetric
    public func getDistanceMetric() -> SupportedDistanceMetric {
        switch configuration.distanceMetric {
        case .euclidean:
            return .euclidean
        case .cosine:
            return .cosine
        case .dotProduct:
            return .dotProduct
        }
    }
}

// MARK: - DynamicEmbedding Extension

extension DynamicEmbedding {
    /// Convert to float array for storage
    public func toFloatArray() -> [Float] {
        switch self {
        case .dim3(let embedding):
            return embedding.toArray()
        case .dim384(let embedding):
            return embedding.toArray()
        case .dim768(let embedding):
            return embedding.toArray()
        case .dim1536(let embedding):
            return embedding.toArray()
        }
    }
}

// MARK: - In-Memory Storage Backend (for testing)

/// Simple in-memory storage backend for testing
public actor InMemoryVectorStorage: VectorStorageBackend {
    private var vectors: [UUID: (vector: [Float], metadata: [String: String])] = [:]

    public init() {}

    public func add(vector: [Float], metadata: [String: String]) async throws -> UUID {
        let id = UUID()
        vectors[id] = (vector, metadata)
        return id
    }

    public func addBatch(vectors: [[Float]], metadata: [[String: String]]) async throws -> [UUID] {
        var ids: [UUID] = []
        for (vector, meta) in zip(vectors, metadata) {
            let id = try await add(vector: vector, metadata: meta)
            ids.append(id)
        }
        return ids
    }

    public func search(
        query: [Float],
        k: Int,
        threshold: Float?
    ) async throws -> [(id: UUID, score: Float, metadata: [String: String])] {
        var scores: [(id: UUID, score: Float, metadata: [String: String])] = []

        for (id, stored) in vectors {
            let score = cosineSimilarity(query, stored.vector)

            if let threshold = threshold, score < threshold {
                continue
            }

            scores.append((id: id, score: score, metadata: stored.metadata))
        }

        // Sort by score (descending) and take top k
        scores.sort { $0.score > $1.score }
        return Array(scores.prefix(k))
    }

    public func remove(id: UUID) async throws {
        vectors.removeValue(forKey: id)
    }

    public func get(id: UUID) async throws -> (vector: [Float], metadata: [String: String])? {
        return vectors[id]
    }

    public func count() async -> Int {
        return vectors.count
    }

    public func clear() async throws {
        vectors.removeAll()
    }

    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0 }

        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0

        for i in 0..<a.count {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }

        let denominator = sqrt(normA) * sqrt(normB)
        return denominator > 0 ? dot / denominator : 0
    }
}
