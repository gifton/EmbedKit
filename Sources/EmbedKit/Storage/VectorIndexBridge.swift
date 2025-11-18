//
//  VectorIndexBridge.swift
//  EmbedKit
//
//  Bridge between EmbedKit and the official VectorIndex package
//

import Foundation
import VectorCore
import VectorIndex

/// Bridge implementation connecting EmbedKit to VectorIndex
public actor VectorIndexBridge: VectorStorageBackend {
    private let index: any VectorIndexProtocol
    private var uuidToVectorId: [UUID: VectorID] = [:]
    private var vectorIdToUuid: [VectorID: UUID] = [:]
    private var metadataStore: [UUID: [String: String]] = [:]

    /// Initialize with FlatIndex (default)
    public init(
        dimensions: Int,
        distanceMetric: SupportedDistanceMetric = .cosine
    ) {
        self.index = FlatIndex(dimension: dimensions, metric: distanceMetric)
    }

    /// Initialize with HNSW Index for better performance
    public static func hnsw(
        dimensions: Int,
        distanceMetric: SupportedDistanceMetric = .cosine,
        maxElements: Int = 10000
    ) -> VectorIndexBridge {
        let index = HNSWIndex(dimension: dimensions, metric: distanceMetric)
        return VectorIndexBridge(index: index)
    }

    /// Initialize with IVF Index for large-scale search
    public static func ivf(
        dimensions: Int,
        distanceMetric: SupportedDistanceMetric = .cosine,
        nlist: Int = 100
    ) -> VectorIndexBridge {
        let index = IVFIndex(dimension: dimensions, metric: distanceMetric)
        return VectorIndexBridge(index: index)
    }

    /// Initialize with existing index
    private init(index: any VectorIndexProtocol) {
        self.index = index
    }

    /// Add a vector to the index
    public func add(
        vector: [Float],
        metadata: [String: String]
    ) async throws -> UUID {
        let uuid = UUID()
        let vectorId = uuid.uuidString

        // Add to VectorIndex
        try await index.insert(id: vectorId, vector: vector, metadata: metadata)

        // Store mapping and metadata
        uuidToVectorId[uuid] = vectorId
        vectorIdToUuid[vectorId] = uuid
        metadataStore[uuid] = metadata

        return uuid
    }

    /// Add multiple vectors in batch
    public func addBatch(
        vectors: [[Float]],
        metadata: [[String: String]]
    ) async throws -> [UUID] {
        var items: [(id: VectorID, vector: [Float], metadata: [String: String]?)] = []
        var uuids: [UUID] = []

        for (vector, meta) in zip(vectors, metadata) {
            let uuid = UUID()
            let vectorId = uuid.uuidString

            items.append((id: vectorId, vector: vector, metadata: meta))
            uuids.append(uuid)

            // Store mappings
            uuidToVectorId[uuid] = vectorId
            vectorIdToUuid[vectorId] = uuid
            metadataStore[uuid] = meta
        }

        // Batch insert
        try await index.batchInsert(items)

        return uuids
    }

    /// Search for similar vectors
    public func search(
        query: [Float],
        k: Int,
        threshold: Float? = nil
    ) async throws -> [(id: UUID, score: Float, metadata: [String: String])] {
        // Search in VectorIndex
        let searchResults = try await index.search(query: query, k: k, filter: nil)

        var results: [(id: UUID, score: Float, metadata: [String: String])] = []

        for result in searchResults {
            guard let uuid = vectorIdToUuid[result.id] else { continue }
            guard let metadata = metadataStore[uuid] else { continue }

            // Apply threshold if specified (score is similarity, not distance)
            if let threshold = threshold, result.score < threshold {
                continue
            }

            results.append((id: uuid, score: result.score, metadata: metadata))
        }

        return results
    }

    /// Remove a vector by ID
    public func remove(id: UUID) async throws {
        guard let vectorId = uuidToVectorId[id] else {
            return  // Silently ignore if not found
        }

        // Remove from VectorIndex
        try await index.remove(id: vectorId)

        // Remove from mappings
        uuidToVectorId.removeValue(forKey: id)
        vectorIdToUuid.removeValue(forKey: vectorId)
        metadataStore.removeValue(forKey: id)
    }

    /// Get a specific vector and its metadata
    public func get(id: UUID) async throws -> (vector: [Float], metadata: [String: String])? {
        guard let metadata = metadataStore[id] else {
            return nil
        }

        // VectorIndex doesn't provide direct vector retrieval in current API
        // Return empty vector as placeholder - this is a known limitation
        return ([], metadata)
    }

    /// Get total count of vectors
    public func count() async -> Int {
        return await index.count
    }

    /// Clear all vectors
    public func clear() async throws {
        // Clear the index
        await index.clear()

        // Clear local storage
        uuidToVectorId.removeAll()
        vectorIdToUuid.removeAll()
        metadataStore.removeAll()
    }
}

/// Errors specific to VectorIndex bridge
public enum VectorIndexError: LocalizedError {
    case vectorNotFound(UUID)
    case dimensionMismatch(expected: Int, actual: Int)
    case indexNotInitialized

    public var errorDescription: String? {
        switch self {
        case .vectorNotFound(let id):
            return "Vector not found: \(id)"
        case .dimensionMismatch(let expected, let actual):
            return "Dimension mismatch. Expected: \(expected), Actual: \(actual)"
        case .indexNotInitialized:
            return "Vector index not initialized"
        }
    }
}

// MARK: - Convenience Extensions

public extension VectorIndexAdapter {
    /// Create adapter with VectorIndex bridge (FlatIndex)
    static func withVectorIndex(
        pipeline: EmbeddingPipeline,
        dimensions: Int = 384,
        distanceMetric: SupportedDistanceMetric = .cosine
    ) -> VectorIndexAdapter {
        let bridge = VectorIndexBridge(
            dimensions: dimensions,
            distanceMetric: distanceMetric
        )

        return VectorIndexAdapter(
            pipeline: pipeline,
            storage: bridge,
            configuration: VectorIndexConfiguration(
                distanceMetric: distanceMetricToEmbedKit(distanceMetric),
                maxVectors: 100_000,
                indexType: "flat"
            )
        )
    }

    /// Create adapter with HNSW index for better performance
    static func withHNSW(
        pipeline: EmbeddingPipeline,
        dimensions: Int = 384,
        distanceMetric: SupportedDistanceMetric = .cosine,
        maxElements: Int = 10000
    ) -> VectorIndexAdapter {
        let bridge = VectorIndexBridge.hnsw(
            dimensions: dimensions,
            distanceMetric: distanceMetric,
            maxElements: maxElements
        )

        return VectorIndexAdapter(
            pipeline: pipeline,
            storage: bridge,
            configuration: VectorIndexConfiguration(
                distanceMetric: distanceMetricToEmbedKit(distanceMetric),
                maxVectors: maxElements,
                indexType: "hnsw"
            )
        )
    }

    /// Create adapter with IVF index for large-scale search
    static func withIVF(
        pipeline: EmbeddingPipeline,
        dimensions: Int = 384,
        distanceMetric: SupportedDistanceMetric = .cosine,
        nlist: Int = 100
    ) -> VectorIndexAdapter {
        let bridge = VectorIndexBridge.ivf(
            dimensions: dimensions,
            distanceMetric: distanceMetric,
            nlist: nlist
        )

        return VectorIndexAdapter(
            pipeline: pipeline,
            storage: bridge,
            configuration: VectorIndexConfiguration(
                distanceMetric: distanceMetricToEmbedKit(distanceMetric),
                maxVectors: 1_000_000,
                indexType: "ivf"
            )
        )
    }

    private static func distanceMetricToEmbedKit(_ metric: SupportedDistanceMetric) -> VectorIndexConfiguration.DistanceMetric {
        switch metric {
        case .cosine:
            return .cosine
        case .euclidean:
            return .euclidean
        case .dotProduct:
            return .dotProduct
        case .manhattan:
            return .euclidean  // Fallback to euclidean for unsupported metrics
        case .chebyshev:
            return .euclidean  // Fallback to euclidean for unsupported metrics
        }
    }
}