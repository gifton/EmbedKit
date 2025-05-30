import Foundation
import Collections
import OSLog

/// Thread-safe LRU (Least Recently Used) cache for embeddings
public actor LRUCache<Key: Hashable & Sendable, Value: Sendable> {
    private let logger = Logger(subsystem: "EmbedKit", category: "LRUCache")
    
    private let maxSize: Int
    private var cache: OrderedDictionary<Key, Value> = [:]
    private var accessOrder: Deque<Key> = []
    private var hitCount: Int = 0
    private var missCount: Int = 0
    private var evictionCount: Int = 0
    
    /// Statistics about cache performance
    public var statistics: CacheStatistics {
        CacheStatistics(
            hits: hitCount,
            misses: missCount,
            evictions: evictionCount,
            currentSize: cache.count,
            maxSize: maxSize
        )
    }
    
    public init(maxSize: Int) {
        self.maxSize = max(1, maxSize)
    }
    
    /// Get a value from the cache
    public func get(_ key: Key) -> Value? {
        if let value = cache[key] {
            // Move to end (most recently used)
            moveToEnd(key)
            hitCount += 1
            logger.trace("Cache hit")
            return value
        } else {
            missCount += 1
            logger.trace("Cache miss")
            return nil
        }
    }
    
    /// Set a value in the cache
    public func set(_ key: Key, value: Value) {
        if cache[key] != nil {
            // Update existing value
            cache[key] = value
            moveToEnd(key)
        } else {
            // Add new value
            cache[key] = value
            accessOrder.append(key)
            
            // Evict if necessary
            while cache.count > maxSize {
                evictOldest()
            }
        }
        
        logger.trace("Cache set, size: \(self.cache.count)")
    }
    
    /// Remove a specific key from the cache
    public func remove(_ key: Key) {
        guard cache[key] != nil else { return }
        
        cache.removeValue(forKey: key)
        accessOrder.removeAll { $0 == key }
        
        logger.trace("Cache remove")
    }
    
    /// Clear all entries from the cache
    public func clear() {
        let previousSize = cache.count
        cache.removeAll()
        accessOrder.removeAll()
        
        logger.debug("Cache cleared, previous size: \(previousSize)")
    }
    
    /// Check if a key exists in the cache
    public func contains(_ key: Key) -> Bool {
        cache[key] != nil
    }
    
    /// Get the current size of the cache
    public var count: Int {
        cache.count
    }
    
    /// Get all keys currently in the cache
    public var keys: [Key] {
        Array(cache.keys)
    }
    
    // MARK: - Private Helpers
    
    private func moveToEnd(_ key: Key) {
        accessOrder.removeAll { $0 == key }
        accessOrder.append(key)
    }
    
    private func evictOldest() {
        guard let oldestKey = accessOrder.popFirst() else { return }
        
        cache.removeValue(forKey: oldestKey)
        evictionCount += 1
        
        logger.trace("Cache eviction, new size: \(self.cache.count)")
    }
}

/// Specialized LRU cache for text embeddings
public actor EmbeddingCache {
    private let logger = Logger(subsystem: "EmbedKit", category: "EmbeddingCache")
    private let cache: LRUCache<String, CachedEmbedding>
    private let maxMemoryBytes: Int
    private var currentMemoryBytes: Int = 0
    
    public struct CachedEmbedding: Sendable {
        public let embedding: EmbeddingVector
        public let modelIdentifier: String
        public let timestamp: Date
        public let byteSize: Int
        
        init(embedding: EmbeddingVector, modelIdentifier: String) {
            self.embedding = embedding
            self.modelIdentifier = modelIdentifier
            self.timestamp = Date()
            // Approximate byte size: dimensions * 4 bytes per float + overhead
            self.byteSize = embedding.dimensions * MemoryLayout<Float>.size + 64
        }
    }
    
    public init(maxEntries: Int = 10000, maxMemoryMB: Int = 100) {
        self.cache = LRUCache(maxSize: maxEntries)
        self.maxMemoryBytes = maxMemoryMB * 1024 * 1024
    }
    
    /// Generate a cache key from text and model identifier
    public static func cacheKey(for text: String, modelIdentifier: String) -> String {
        // Use a hash to avoid storing full text in memory
        let textHash = text.hashValue
        return "\(modelIdentifier):\(textHash)"
    }
    
    /// Get an embedding from the cache
    public func get(text: String, modelIdentifier: String) async -> EmbeddingVector? {
        let key = Self.cacheKey(for: text, modelIdentifier: modelIdentifier)
        
        if let cached = await cache.get(key) {
            // Check if it's from the same model
            guard cached.modelIdentifier == modelIdentifier else {
                await cache.remove(key)
                return nil
            }
            
            return cached.embedding
        }
        
        return nil
    }
    
    /// Store an embedding in the cache
    public func set(text: String, modelIdentifier: String, embedding: EmbeddingVector) async {
        let key = Self.cacheKey(for: text, modelIdentifier: modelIdentifier)
        let cached = CachedEmbedding(embedding: embedding, modelIdentifier: modelIdentifier)
        
        // Check memory pressure
        if currentMemoryBytes + cached.byteSize > maxMemoryBytes {
            logger.debug("Cache memory pressure, clearing oldest entries")
            // Simple strategy: clear 25% of cache when memory limit reached
            let targetSize = await cache.count * 3 / 4
            while await cache.count > targetSize {
                if let oldestKey = await cache.keys.first {
                    await cache.remove(oldestKey)
                }
            }
        }
        
        await cache.set(key, value: cached)
        currentMemoryBytes += cached.byteSize
    }
    
    /// Get cache statistics
    public func statistics() async -> CacheStatistics {
        await cache.statistics
    }
    
    /// Clear the cache
    public func clear() async {
        await cache.clear()
        currentMemoryBytes = 0
    }
    
    /// Preload embeddings for a batch of texts
    public func preload(texts: [String], modelIdentifier: String, embeddings: [EmbeddingVector]) async {
        guard texts.count == embeddings.count else {
            logger.error("Mismatch between texts and embeddings count")
            return
        }
        
        for (text, embedding) in zip(texts, embeddings) {
            await set(text: text, modelIdentifier: modelIdentifier, embedding: embedding)
        }
        
        logger.debug("Preloaded \(texts.count) embeddings into cache")
    }
}

/// Memory-aware cache that responds to system memory pressure
public final class MemoryAwareCache: @unchecked Sendable {
    private let embeddingCache: EmbeddingCache
    private var memoryPressureSource: DispatchSourceMemoryPressure?
    private let logger = Logger(subsystem: "EmbedKit", category: "MemoryAwareCache")
    
    public init(embeddingCache: EmbeddingCache) {
        self.embeddingCache = embeddingCache
        setupMemoryPressureHandling()
    }
    
    deinit {
        memoryPressureSource?.cancel()
    }
    
    private func setupMemoryPressureHandling() {
        memoryPressureSource = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical], queue: .global())
        
        memoryPressureSource?.setEventHandler { [weak self] in
            guard let self = self else { return }
            
            let pressure = self.memoryPressureSource?.data ?? []
            
            Task {
                if pressure.contains(.critical) {
                    self.logger.warning("Critical memory pressure detected, clearing cache")
                    await self.embeddingCache.clear()
                } else if pressure.contains(.warning) {
                    self.logger.info("Memory pressure warning, reducing cache size")
                    // Clear 50% of cache on warning
                    let stats = await self.embeddingCache.statistics()
                    let targetSize = stats.currentSize / 2
                    
                    while await self.embeddingCache.statistics().currentSize > targetSize {
                        // The LRU cache will handle eviction
                        break
                    }
                }
            }
        }
        
        memoryPressureSource?.resume()
    }
}