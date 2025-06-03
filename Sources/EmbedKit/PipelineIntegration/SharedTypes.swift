import Foundation

// MARK: - Shared Types for EmbedKit and VectorStoreKit Integration

/// A document that contains text and can be embedded
public struct EmbeddingDocument: Sendable {
    public let id: String
    public let text: String
    public let metadata: [String: String]
    public let createdAt: Date
    
    public init(
        id: String = UUID().uuidString,
        text: String,
        metadata: [String: String] = [:],
        createdAt: Date = Date()
    ) {
        self.id = id
        self.text = text
        self.metadata = metadata
        self.createdAt = createdAt
    }
}

/// Embedded document with its vector representation
public struct EmbeddedDocument: Sendable {
    public let document: EmbeddingDocument
    public let embedding: EmbeddingVector
    public let modelIdentifier: ModelIdentifier
    public let embeddedAt: Date
    
    public init(
        document: EmbeddingDocument,
        embedding: EmbeddingVector,
        modelIdentifier: ModelIdentifier,
        embeddedAt: Date = Date()
    ) {
        self.document = document
        self.embedding = embedding
        self.modelIdentifier = modelIdentifier
        self.embeddedAt = embeddedAt
    }
}

/// Search result containing document and similarity score
public struct SearchResult: Sendable {
    public let document: EmbeddingDocument
    public let embedding: EmbeddingVector
    public let score: Float
    public let rank: Int
    
    public init(
        document: EmbeddingDocument,
        embedding: EmbeddingVector,
        score: Float,
        rank: Int
    ) {
        self.document = document
        self.embedding = embedding
        self.score = score
        self.rank = rank
    }
}

/// Filter for metadata-based queries
public struct MetadataFilter: Sendable {
    public enum Condition: Sendable {
        case equals(String, String)
        case contains(String, String)
        case startsWith(String, String)
        case endsWith(String, String)
        case greaterThan(String, String)
        case lessThan(String, String)
        case inSet(String, Set<String>)
    }
    
    public let conditions: [Condition]
    public let requireAll: Bool
    
    public init(conditions: [Condition], requireAll: Bool = true) {
        self.conditions = conditions
        self.requireAll = requireAll
    }
    
    /// Helper to create a single condition filter
    public static func single(_ condition: Condition) -> MetadataFilter {
        MetadataFilter(conditions: [condition])
    }
}

// MARK: - Index Configuration

/// Configuration for vector index operations
public struct IndexConfiguration: Sendable {
    public let name: String
    public let dimension: Int
    public let metric: SimilarityMetric
    public let capacity: Int
    public let enableMetadataFiltering: Bool
    
    public enum SimilarityMetric: String, Sendable {
        case cosine
        case euclidean
        case dotProduct
    }
    
    public init(
        name: String,
        dimension: Int,
        metric: SimilarityMetric = .cosine,
        capacity: Int = 100_000,
        enableMetadataFiltering: Bool = true
    ) {
        self.name = name
        self.dimension = dimension
        self.metric = metric
        self.capacity = capacity
        self.enableMetadataFiltering = enableMetadataFiltering
    }
}

// MARK: - Batch Operations

/// Configuration for batch indexing operations
public struct BatchIndexConfiguration: Sendable {
    public let batchSize: Int
    public let parallelism: Int
    public let skipDuplicates: Bool
    public let updateExisting: Bool
    
    public init(
        batchSize: Int = 100,
        parallelism: Int = 4,
        skipDuplicates: Bool = true,
        updateExisting: Bool = false
    ) {
        self.batchSize = batchSize
        self.parallelism = parallelism
        self.skipDuplicates = skipDuplicates
        self.updateExisting = updateExisting
    }
}

// MARK: - Pipeline Integration Types

/// Result of an indexing operation
public struct IndexResult: Sendable {
    public let documentId: String
    public let embedding: EmbeddingVector
    public let indexedAt: Date
    public let duration: TimeInterval
    
    public init(
        documentId: String,
        embedding: EmbeddingVector,
        indexedAt: Date = Date(),
        duration: TimeInterval
    ) {
        self.documentId = documentId
        self.embedding = embedding
        self.indexedAt = indexedAt
        self.duration = duration
    }
}

/// Result of a batch indexing operation
public struct BatchIndexResult: Sendable {
    public let successful: [IndexResult]
    public let failed: [(document: EmbeddingDocument, error: String)]
    public let totalDuration: TimeInterval
    public let averageDuration: TimeInterval
    
    public init(
        successful: [IndexResult],
        failed: [(document: EmbeddingDocument, error: String)],
        totalDuration: TimeInterval
    ) {
        self.successful = successful
        self.failed = failed
        self.totalDuration = totalDuration
        self.averageDuration = successful.isEmpty ? 0 : totalDuration / Double(successful.count)
    }
    
    public var successRate: Double {
        let total = successful.count + failed.count
        return total == 0 ? 0 : Double(successful.count) / Double(total)
    }
}

// MARK: - Error Types

/// Errors that can occur during vector store operations
public enum VectorStoreError: Error, LocalizedError {
    case dimensionMismatch(expected: Int, actual: Int)
    case documentNotFound(id: String)
    case indexNotFound(name: String)
    case duplicateDocument(id: String)
    case invalidQuery(reason: String)
    case storageError(underlying: Error)
    case embeddingError(underlying: Error)
    
    public var errorDescription: String? {
        switch self {
        case .dimensionMismatch(let expected, let actual):
            return "Dimension mismatch: expected \(expected), got \(actual)"
        case .documentNotFound(let id):
            return "Document not found: \(id)"
        case .indexNotFound(let name):
            return "Index not found: \(name)"
        case .duplicateDocument(let id):
            return "Duplicate document: \(id)"
        case .invalidQuery(let reason):
            return "Invalid query: \(reason)"
        case .storageError(let underlying):
            return "Storage error: \(underlying.localizedDescription)"
        case .embeddingError(let underlying):
            return "Embedding error: \(underlying.localizedDescription)"
        }
    }
}