// EmbedKit - Reranking Strategy
// Protocols and implementations for search result reranking

import Foundation
import VectorCore

// MARK: - Reranking Strategy Protocol

/// Protocol for reranking search results.
///
/// Reranking allows refining initial search results using more expensive
/// but more accurate scoring methods.
public protocol RerankingStrategy: Sendable {
    /// Rerank search results.
    /// - Parameters:
    ///   - query: The query embedding
    ///   - candidates: Initial search results to rerank
    ///   - k: Number of results to return after reranking
    /// - Returns: Reranked results
    func rerank(
        query: Embedding,
        candidates: [EmbeddingSearchResult],
        k: Int
    ) async throws -> [EmbeddingSearchResult]

    /// Strategy name for logging/debugging.
    var name: String { get }
}

// MARK: - Reranking Options

/// Options for controlling reranking behavior.
public struct RerankOptions: Sendable {
    /// How many candidates to fetch for reranking (multiplier of k).
    public var candidateMultiplier: Int

    /// Enable parallel scoring for large candidate sets.
    public var enableParallel: Bool

    /// Minimum similarity threshold for results.
    public var minSimilarity: Float?

    /// Default options.
    public static let `default` = RerankOptions(
        candidateMultiplier: 3,
        enableParallel: true,
        minSimilarity: nil
    )

    /// Fast reranking with fewer candidates.
    public static let fast = RerankOptions(
        candidateMultiplier: 2,
        enableParallel: false,
        minSimilarity: nil
    )

    /// Accurate reranking with more candidates.
    public static let accurate = RerankOptions(
        candidateMultiplier: 5,
        enableParallel: true,
        minSimilarity: nil
    )

    public init(
        candidateMultiplier: Int = 3,
        enableParallel: Bool = true,
        minSimilarity: Float? = nil
    ) {
        self.candidateMultiplier = candidateMultiplier
        self.enableParallel = enableParallel
        self.minSimilarity = minSimilarity
    }
}

// MARK: - Exact Cosine Reranking

/// Reranks results by computing exact cosine similarity.
///
/// Use this when initial search used approximate methods (HNSW, IVF)
/// and you want more accurate final ranking.
///
/// Uses VectorCore's `TopKSelection` for O(n log k) heap-based selection
/// instead of O(n log n) full sorting.
public struct ExactCosineRerank: RerankingStrategy {
    public let name = "ExactCosine"

    public init() {}

    public func rerank(
        query: Embedding,
        candidates: [EmbeddingSearchResult],
        k: Int
    ) async throws -> [EmbeddingSearchResult] {
        guard !candidates.isEmpty && k > 0 else { return [] }

        // Recompute similarity for candidates that have embeddings
        var reranked: [EmbeddingSearchResult] = []
        reranked.reserveCapacity(candidates.count)

        for candidate in candidates {
            if let embedding = candidate.embedding {
                let similarity = query.similarity(to: embedding)
                reranked.append(EmbeddingSearchResult(
                    id: candidate.id,
                    distance: 1 - similarity,
                    similarity: similarity,
                    text: candidate.text,
                    metadata: candidate.metadata,
                    embedding: embedding
                ))
            } else {
                // Keep original if no embedding available
                reranked.append(candidate)
            }
        }

        // Use TopKSelection for O(n log k) selection instead of O(n log n) sort
        // Since we want highest similarity (not lowest distance), use negative similarity
        let topK = TopKSelection.select(k: k, from: reranked) { -$0.similarity }
        return topK
    }
}

// MARK: - Diversity Reranking

/// Reranks to maximize diversity among results using MMR (Maximal Marginal Relevance).
///
/// Balances relevance with diversity to avoid redundant results.
public struct DiversityRerank: RerankingStrategy {
    public let name = "DiversityMMR"

    /// Lambda parameter: 0 = max diversity, 1 = max relevance.
    public let lambda: Float

    public init(lambda: Float = 0.5) {
        self.lambda = max(0, min(1, lambda))
    }

    public func rerank(
        query: Embedding,
        candidates: [EmbeddingSearchResult],
        k: Int
    ) async throws -> [EmbeddingSearchResult] {
        guard !candidates.isEmpty else { return [] }
        guard k > 0 else { return [] }

        var selected: [EmbeddingSearchResult] = []
        var remaining = candidates

        // Greedily select using MMR
        while selected.count < k && !remaining.isEmpty {
            var bestScore: Float = -.infinity
            var bestIndex = 0

            for (index, candidate) in remaining.enumerated() {
                // Relevance to query
                let relevance = candidate.similarity

                // Maximum similarity to already selected results
                var maxSimToSelected: Float = 0
                for selectedResult in selected {
                    if let candEmb = candidate.embedding,
                       let selEmb = selectedResult.embedding {
                        let sim = candEmb.similarity(to: selEmb)
                        maxSimToSelected = max(maxSimToSelected, sim)
                    }
                }

                // MMR score
                let mmrScore = lambda * relevance - (1 - lambda) * maxSimToSelected

                if mmrScore > bestScore {
                    bestScore = mmrScore
                    bestIndex = index
                }
            }

            selected.append(remaining.remove(at: bestIndex))
        }

        return selected
    }
}

// MARK: - Threshold Reranking

/// Filters results by a minimum similarity threshold.
public struct ThresholdRerank: RerankingStrategy {
    public let name = "Threshold"

    /// Minimum similarity score to include.
    public let minSimilarity: Float

    public init(minSimilarity: Float) {
        self.minSimilarity = minSimilarity
    }

    public func rerank(
        query: Embedding,
        candidates: [EmbeddingSearchResult],
        k: Int
    ) async throws -> [EmbeddingSearchResult] {
        candidates
            .filter { $0.similarity >= minSimilarity }
            .prefix(k)
            .map { $0 }
    }
}

// MARK: - Composite Reranking

/// Combines multiple reranking strategies in sequence.
public struct CompositeRerank: RerankingStrategy {
    public let name: String
    private let strategies: [any RerankingStrategy]

    public init(strategies: [any RerankingStrategy]) {
        self.strategies = strategies
        self.name = "Composite[\(strategies.map { $0.name }.joined(separator: "→"))]"
    }

    public func rerank(
        query: Embedding,
        candidates: [EmbeddingSearchResult],
        k: Int
    ) async throws -> [EmbeddingSearchResult] {
        var results = candidates

        for (index, strategy) in strategies.enumerated() {
            // Only apply k limit on final strategy
            let limit = (index == strategies.count - 1) ? k : results.count
            results = try await strategy.rerank(query: query, candidates: results, k: limit)
        }

        return results
    }
}

// MARK: - No-Op Reranking

/// Pass-through strategy that performs no reranking.
public struct NoRerank: RerankingStrategy {
    public let name = "None"

    public init() {}

    public func rerank(
        query: Embedding,
        candidates: [EmbeddingSearchResult],
        k: Int
    ) async throws -> [EmbeddingSearchResult] {
        Array(candidates.prefix(k))
    }
}
