//
//  RerankingStrategy.swift
//  EmbedKit
//
//  Clean implementation without any backward compatibility
//

import Foundation
import VectorIndex
import VectorCore

// MARK: - Reranking Strategy Protocol

/// Protocol for implementing reranking strategies that actually rerank
public protocol RerankingStrategy: Sendable {
    /// Rerank search results by recomputing scores
    func rerank(
        query: DynamicEmbedding,
        candidates: [VectorSearchResult],
        k: Int,
        options: RerankOptions
    ) async throws -> [VectorSearchResult]

    /// Strategy name for logging/debugging
    var name: String { get }
}

// MARK: - Reranking Options

/// Configuration for reranking behavior
public struct RerankOptions: Sendable {
    /// How many candidates to fetch (multiplier of k)
    public var candidateMultiplier: Int = 3

    /// Enable parallel scoring for large candidate sets
    public var enableParallel: Bool = true

    /// Vectors processed per batch
    public var tileSize: Int = 128

    /// Skip missing vectors vs using sentinels
    public var skipMissing: Bool = true

    /// Max threads for parallel ops (0 = auto)
    public var maxConcurrency: Int = 0

    /// Common presets
    public static let `default` = RerankOptions()

    public static let fast = RerankOptions(
        candidateMultiplier: 2,
        enableParallel: false,
        tileSize: 64
    )

    public static let accurate = RerankOptions(
        candidateMultiplier: 5,
        enableParallel: true,
        tileSize: 256
    )
}

// MARK: - Exact Reranking Strategy

/// Reranking using exact distance computation on original vectors
public actor ExactRerankStrategy: RerankingStrategy {
    private let storage: any VectorStorageBackend
    private let metric: SupportedDistanceMetric
    private let dimension: Int

    public let name = "ExactRerank"

    public init(
        storage: any VectorStorageBackend,
        metric: SupportedDistanceMetric,
        dimension: Int
    ) {
        self.storage = storage
        self.metric = metric
        self.dimension = dimension
    }

    public func rerank(
        query: DynamicEmbedding,
        candidates: [VectorSearchResult],
        k: Int,
        options: RerankOptions
    ) async throws -> [VectorSearchResult] {
        guard !candidates.isEmpty else { return [] }

        let queryVector = query.toFloatArray()
        guard queryVector.count == dimension else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: queryVector.count)
        }

        // Fetch original vectors for candidates
        let vectors = try await fetchVectors(for: candidates)
        guard !vectors.isEmpty else {
            return options.skipMissing ? [] : Array(candidates.prefix(k))
        }

        // Build matrix for ExactRerank kernel
        var vectorMatrix: [Float] = []
        var validCandidates: [(Int64, VectorSearchResult)] = []

        for (index, candidate) in candidates.enumerated() {
            if let vector = vectors[candidate.id] {
                vectorMatrix.append(contentsOf: vector)
                validCandidates.append((Int64(index), candidate))
            } else if !options.skipMissing {
                validCandidates.append((Int64(index), candidate))
            }
        }

        // Use VectorIndex's ExactRerank kernel
        let rerankOpts = IndexOps.Rerank.RerankOpts(
            backend: .denseArray,
            gatherTile: options.tileSize,
            returnSorted: true,
            skipMissing: options.skipMissing,
            enableParallel: options.enableParallel,
            maxConcurrency: options.maxConcurrency
        )

        let (scores, ids) = IndexOps.Rerank.topKDense(
            q: queryVector,
            d: dimension,
            metric: metric,
            candIDs: validCandidates.map { $0.0 },
            xb: vectorMatrix,
            K: min(k, validCandidates.count),
            opts: rerankOpts
        )

        // Map back to results
        var rerankedResults: [VectorSearchResult] = []
        for (score, id) in zip(scores, ids) where id >= 0 {
            if let candidate = validCandidates.first(where: { $0.0 == id }) {
                var result = candidate.1
                result.score = score
                rerankedResults.append(result)
            }
        }

        return rerankedResults
    }

    private func fetchVectors(for candidates: [VectorSearchResult]) async throws -> [String: [Float]] {
        var vectors: [String: [Float]] = [:]

        for candidate in candidates {
            if let uuid = UUID(uuidString: candidate.id),
               let stored = try? await storage.get(id: uuid) {
                vectors[candidate.id] = stored.vector
            }
        }

        return vectors
    }
}

// MARK: - Future Reranking Strategies

/// Placeholder for cross-encoder neural reranking
public struct CrossEncoderRerankStrategy: RerankingStrategy {
    public let name = "CrossEncoder"

    public init() {}

    public func rerank(
        query: DynamicEmbedding,
        candidates: [VectorSearchResult],
        k: Int,
        options: RerankOptions
    ) async throws -> [VectorSearchResult] {
        // TODO: Implement when cross-encoder models are available
        throw RerankError.notImplemented("Cross-encoder reranking coming soon")
    }
}

// MARK: - Error Types

public enum RerankError: Error {
    case dimensionMismatch(expected: Int, actual: Int)
    case notImplemented(String)
    case invalidConfiguration(String)
}