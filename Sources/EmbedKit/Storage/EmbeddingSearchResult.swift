// EmbedKit - Embedding Search Result
// Result type for vector similarity searches

import Foundation
import VectorCore

// MARK: - Embedding Search Result

/// Result from a vector similarity search.
///
/// Contains the matched embedding along with relevance scores and metadata.
public struct EmbeddingSearchResult: Sendable, Identifiable {
    /// Unique identifier of the matched embedding
    public let id: String

    /// Distance/score from the query (interpretation depends on metric)
    /// - For euclidean: lower is more similar
    /// - For cosine: lower is more similar (cosine distance, not similarity)
    /// - For dot product: higher is more similar
    public let distance: Float

    /// Cosine similarity score (0-1, higher is more similar)
    /// Computed from distance when available
    public let similarity: Float

    /// Original text that was embedded (if stored)
    public let text: String?

    /// Additional metadata associated with the embedding
    public let metadata: [String: String]?

    /// The embedding vector (if retrieved)
    public let embedding: Embedding?

    public init(
        id: String,
        distance: Float,
        similarity: Float? = nil,
        text: String? = nil,
        metadata: [String: String]? = nil,
        embedding: Embedding? = nil
    ) {
        self.id = id
        self.distance = distance
        // Convert distance to similarity if not provided
        // Assumes cosine distance: similarity = 1 - distance
        self.similarity = similarity ?? max(0, 1 - distance)
        self.text = text
        self.metadata = metadata
        self.embedding = embedding
    }

    /// Create with distance and metric.
    ///
    /// Automatically computes similarity from distance based on the metric.
    internal init(
        id: String,
        distance: Float,
        text: String? = nil,
        metadata: [String: String]? = nil,
        embedding: Embedding? = nil,
        metric: SupportedDistanceMetric
    ) {
        self.id = id
        self.distance = distance
        self.text = text
        self.metadata = metadata
        self.embedding = embedding

        // Convert distance to similarity based on metric
        switch metric {
        case .cosine:
            // Cosine distance: 0 = identical, 2 = opposite
            self.similarity = max(0, 1 - distance)
        case .euclidean:
            // Euclidean: 0 = identical, larger = more different
            // GPU returns L2² (squared distance), take sqrt then apply decay
            let euclideanDist = sqrt(distance)
            self.similarity = exp(-euclideanDist)
        case .dotProduct:
            // Dot product: higher = more similar (already a similarity)
            // Normalize to 0-1 range (assuming normalized vectors)
            self.similarity = (distance + 1) / 2
        case .manhattan, .chebyshev:
            // Similar treatment to euclidean
            self.similarity = exp(-distance)
        }
    }
}

// MARK: - Comparable

extension EmbeddingSearchResult: Comparable {
    public static func < (lhs: EmbeddingSearchResult, rhs: EmbeddingSearchResult) -> Bool {
        // Lower similarity = less relevant = should be "less than"
        // This makes sorted() put higher similarity first when reversed, or use sorted(by: >)
        lhs.similarity < rhs.similarity
    }
}

// MARK: - Equatable

extension EmbeddingSearchResult: Equatable {
    public static func == (lhs: EmbeddingSearchResult, rhs: EmbeddingSearchResult) -> Bool {
        lhs.id == rhs.id && lhs.distance == rhs.distance
    }
}

// MARK: - Hashable

extension EmbeddingSearchResult: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
        hasher.combine(distance)
    }
}

// MARK: - Collection Extensions

public extension Array where Element == EmbeddingSearchResult {
    /// Filter results by minimum similarity threshold.
    func filtered(minSimilarity: Float) -> [EmbeddingSearchResult] {
        filter { $0.similarity >= minSimilarity }
    }

    /// Get only results with text.
    var withText: [EmbeddingSearchResult] {
        filter { $0.text != nil }
    }

    /// Get texts from results.
    var texts: [String] {
        compactMap { $0.text }
    }

    /// Get IDs from results.
    var ids: [String] {
        map { $0.id }
    }

    /// Best match (highest similarity).
    var best: EmbeddingSearchResult? {
        self.max(by: { $0.similarity < $1.similarity })
    }
}
