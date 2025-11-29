// EmbedKit - Persistent Cache Types
// Configuration, statistics, and cached embedding structures

import Foundation

// MARK: - Cache Configuration

/// Configuration for persistent embedding cache.
public struct CacheConfiguration: Sendable {
    /// Maximum number of embeddings to store (0 = unlimited).
    public var maxEntries: Int

    /// Maximum cache size in bytes (0 = unlimited).
    public var maxSizeBytes: Int64

    /// Whether to enable semantic deduplication.
    /// When enabled, similar embeddings may share storage.
    public var enableSemanticDedup: Bool

    /// Similarity threshold for semantic deduplication (0.0 to 1.0).
    /// Embeddings with similarity >= threshold are considered duplicates.
    public var deduplicationThreshold: Float

    /// Automatically evict old entries when limits are exceeded.
    public var autoEvict: Bool

    /// Time-to-live for cache entries in seconds (0 = forever).
    public var ttlSeconds: TimeInterval

    /// Enable write-ahead logging for better crash recovery.
    public var enableWAL: Bool

    /// Default configuration with reasonable limits.
    public static let `default` = CacheConfiguration(
        maxEntries: 100_000,
        maxSizeBytes: 500 * 1024 * 1024,  // 500 MB
        enableSemanticDedup: false,
        deduplicationThreshold: 0.98,
        autoEvict: true,
        ttlSeconds: 0,
        enableWAL: true
    )

    /// In-memory only configuration (for testing).
    public static let inMemory = CacheConfiguration(
        maxEntries: 10_000,
        maxSizeBytes: 100 * 1024 * 1024,  // 100 MB
        enableSemanticDedup: false,
        deduplicationThreshold: 0.98,
        autoEvict: true,
        ttlSeconds: 0,
        enableWAL: false
    )

    public init(
        maxEntries: Int = 100_000,
        maxSizeBytes: Int64 = 500 * 1024 * 1024,
        enableSemanticDedup: Bool = false,
        deduplicationThreshold: Float = 0.98,
        autoEvict: Bool = true,
        ttlSeconds: TimeInterval = 0,
        enableWAL: Bool = true
    ) {
        self.maxEntries = maxEntries
        self.maxSizeBytes = maxSizeBytes
        self.enableSemanticDedup = enableSemanticDedup
        self.deduplicationThreshold = deduplicationThreshold
        self.autoEvict = autoEvict
        self.ttlSeconds = ttlSeconds
        self.enableWAL = enableWAL
    }
}

// MARK: - Cache Statistics

/// Statistics about cache usage and performance.
public struct CacheStatistics: Sendable, Codable {
    /// Total number of cache lookups.
    public var totalLookups: Int

    /// Number of exact match hits.
    public var exactHits: Int

    /// Number of semantic similarity hits.
    public var semanticHits: Int

    /// Number of cache misses.
    public var misses: Int

    /// Number of entries currently in cache.
    public var entryCount: Int

    /// Total size of cached data in bytes.
    public var sizeBytes: Int64

    /// Number of evictions performed.
    public var evictions: Int

    /// Timestamp of last cache operation.
    public var lastAccessTime: Date?

    /// Timestamp when statistics were reset.
    public var resetTime: Date

    /// Overall hit rate (exact + semantic hits / total lookups).
    public var hitRate: Double {
        guard totalLookups > 0 else { return 0 }
        return Double(exactHits + semanticHits) / Double(totalLookups)
    }

    /// Exact match hit rate.
    public var exactHitRate: Double {
        guard totalLookups > 0 else { return 0 }
        return Double(exactHits) / Double(totalLookups)
    }

    /// Average entry size in bytes.
    public var averageEntrySize: Int64 {
        guard entryCount > 0 else { return 0 }
        return sizeBytes / Int64(entryCount)
    }

    public init() {
        self.totalLookups = 0
        self.exactHits = 0
        self.semanticHits = 0
        self.misses = 0
        self.entryCount = 0
        self.sizeBytes = 0
        self.evictions = 0
        self.lastAccessTime = nil
        self.resetTime = Date()
    }

    /// Reset all statistics.
    public mutating func reset() {
        self = CacheStatistics()
    }
}

// MARK: - Cached Embedding

/// An embedding stored in the persistent cache.
public struct CachedEmbedding: Sendable {
    /// Unique identifier in the cache.
    public let id: Int64

    /// The original text that was embedded.
    public let text: String

    /// Hash of the normalized text (for fast lookup).
    public let textHash: String

    /// The embedding vector.
    public let embedding: Embedding

    /// Model that generated this embedding.
    public let modelID: ModelID

    /// When this entry was created.
    public let createdAt: Date

    /// When this entry was last accessed.
    public let accessedAt: Date

    /// Number of times this entry has been accessed.
    public let accessCount: Int

    /// Estimated size of this entry in bytes.
    public var estimatedSize: Int64 {
        // text bytes + vector bytes + overhead
        let textSize = Int64(text.utf8.count)
        let vectorSize = Int64(embedding.dimensions * MemoryLayout<Float>.size)
        let overhead: Int64 = 200  // metadata, hash, etc.
        return textSize + vectorSize + overhead
    }

    public init(
        id: Int64,
        text: String,
        textHash: String,
        embedding: Embedding,
        modelID: ModelID,
        createdAt: Date,
        accessedAt: Date,
        accessCount: Int
    ) {
        self.id = id
        self.text = text
        self.textHash = textHash
        self.embedding = embedding
        self.modelID = modelID
        self.createdAt = createdAt
        self.accessedAt = accessedAt
        self.accessCount = accessCount
    }
}

// MARK: - Cache Result

/// Result of a cache lookup operation.
public enum CacheResult: Sendable {
    /// Exact text match found.
    case exactMatch(CachedEmbedding)

    /// Semantically similar embedding found.
    case semanticMatch(CachedEmbedding, similarity: Float)

    /// No matching entry found.
    case miss

    /// Whether the lookup was successful.
    public var isHit: Bool {
        switch self {
        case .exactMatch, .semanticMatch:
            return true
        case .miss:
            return false
        }
    }

    /// The cached embedding if found.
    public var embedding: CachedEmbedding? {
        switch self {
        case .exactMatch(let cached), .semanticMatch(let cached, _):
            return cached
        case .miss:
            return nil
        }
    }
}

// MARK: - Cache Errors

/// Errors that can occur during cache operations.
public enum CacheError: Error, LocalizedError, Sendable {
    case databaseError(String)
    case serializationError(String)
    case capacityExceeded(current: Int64, max: Int64)
    case entryNotFound(String)
    case invalidConfiguration(String)
    case ioError(String)

    public var errorDescription: String? {
        switch self {
        case .databaseError(let msg):
            return "Database error: \(msg)"
        case .serializationError(let msg):
            return "Serialization error: \(msg)"
        case .capacityExceeded(let current, let max):
            return "Cache capacity exceeded: \(current) bytes (max: \(max))"
        case .entryNotFound(let key):
            return "Cache entry not found: \(key)"
        case .invalidConfiguration(let msg):
            return "Invalid cache configuration: \(msg)"
        case .ioError(let msg):
            return "I/O error: \(msg)"
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .databaseError:
            return "Check that the cache database file is not corrupted or locked by another process. Try clearing the cache or recreating the database."
        case .serializationError:
            return "The cached data format may be incompatible. Clear the cache and retry. If persisting custom types, ensure they conform to Codable."
        case .capacityExceeded:
            return "Increase the cache capacity limit in configuration, or call trimToSize() to remove old entries. Consider using a cleanup policy."
        case .entryNotFound:
            return "The requested cache entry does not exist or has expired. Re-compute the embedding and cache it again."
        case .invalidConfiguration:
            return "Review cache configuration parameters. Ensure directory paths are valid and capacity limits are reasonable."
        case .ioError:
            return "Check disk space and file permissions. Ensure the cache directory exists and is writable."
        }
    }
}

// MARK: - Text Hashing

#if canImport(CryptoKit)
import CryptoKit
#endif

/// Utilities for consistent text hashing.
public enum TextHasher {
    /// Compute a stable hash for cache key lookup.
    /// Normalizes text before hashing for better deduplication.
    public static func hash(_ text: String) -> String {
        let normalized = normalize(text)
        return sha256(normalized)
    }

    /// Normalize text for consistent hashing.
    /// - Lowercases
    /// - Trims whitespace
    /// - Collapses multiple spaces/newlines
    private static func normalize(_ text: String) -> String {
        text.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    /// Compute SHA-256 hash of a string.
    private static func sha256(_ string: String) -> String {
        let data = Data(string.utf8)

        #if canImport(CryptoKit)
        let hash = SHA256.hash(data: data)
        return hash.compactMap { String(format: "%02x", $0) }.joined()
        #else
        // Fallback: simple hash for platforms without CryptoKit
        var hash: UInt64 = 5381
        for byte in data {
            hash = ((hash << 5) &+ hash) &+ UInt64(byte)
        }
        return String(format: "%016llx", hash)
        #endif
    }
}

// MARK: - Vector Serialization

/// Utilities for serializing embedding vectors to/from binary data.
public enum VectorSerializer {
    /// Serialize a Float array to binary data.
    public static func serialize(_ vector: [Float]) -> Data {
        vector.withUnsafeBytes { Data($0) }
    }

    /// Deserialize binary data to a Float array.
    public static func deserialize(_ data: Data, dimensions: Int) -> [Float]? {
        guard data.count == dimensions * MemoryLayout<Float>.size else {
            return nil
        }

        return data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Float.self))
        }
    }
}
