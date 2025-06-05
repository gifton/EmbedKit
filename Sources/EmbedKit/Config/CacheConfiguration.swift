import Foundation

// MARK: - Cache Configuration

/// Configuration for caching behavior
public struct CacheConfiguration: Sendable {
    /// Maximum number of cached embeddings
    public let maxCacheSize: Int
    
    /// Cache TTL in seconds
    public let ttl: TimeInterval
    
    /// Whether to persist cache to disk
    public let persistToDisk: Bool
    
    /// Cache eviction policy
    public let evictionPolicy: CacheEvictionPolicy
    
    /// Whether to use memory-aware caching
    public let memoryAware: Bool
    
    public init(
        maxCacheSize: Int = 1000,
        ttl: TimeInterval = 3600,
        persistToDisk: Bool = false,
        evictionPolicy: CacheEvictionPolicy = .lru,
        memoryAware: Bool = true
    ) {
        self.maxCacheSize = maxCacheSize
        self.ttl = ttl
        self.persistToDisk = persistToDisk
        self.evictionPolicy = evictionPolicy
        self.memoryAware = memoryAware
    }
    
    // MARK: - Presets
    
    public static let `default` = CacheConfiguration()
    
    public static let aggressive = CacheConfiguration(
        maxCacheSize: 5000,
        ttl: 7200,
        persistToDisk: true,
        evictionPolicy: .lfu
    )
    
    public static let minimal = CacheConfiguration(
        maxCacheSize: 100,
        ttl: 300,
        persistToDisk: false,
        evictionPolicy: .lru,
        memoryAware: true
    )
}

/// Cache eviction policies
public enum CacheEvictionPolicy: String, Sendable {
    case lru = "lru" // Least Recently Used
    case lfu = "lfu" // Least Frequently Used
    case fifo = "fifo" // First In First Out
    case ttl = "ttl" // Time To Live based
}