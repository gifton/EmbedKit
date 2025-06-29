import Foundation
import OSLog

/// Node structure for doubly-linked list implementation
private final class LRUNode<Key: Hashable, Value>: @unchecked Sendable {
    let key: Key
    var value: Value
    var prev: LRUNode?
    var next: LRUNode?
    
    init(key: Key, value: Value) {
        self.key = key
        self.value = value
    }
}

/// High-performance thread-safe LRU cache with O(1) operations
/// 
/// This implementation uses a doubly-linked list for access ordering and a dictionary
/// for O(1) lookups. All operations (get, set, remove) run in O(1) time complexity.
///
/// Performance characteristics:
/// - Get: O(1) lookup + O(1) reordering
/// - Set: O(1) insertion + O(1) potential eviction
/// - Remove: O(1) removal
///
/// Memory characteristics:
/// - Additional overhead per entry: ~48 bytes (node structure + pointers)
/// - Total memory: O(n) where n is the number of cached items
public actor OptimizedLRUCache<Key: Hashable & Sendable, Value: Sendable> {
    private let logger = Logger(subsystem: "EmbedKit", category: "OptimizedLRUCache")
    
    private let maxSize: Int
    private var nodeMap: [Key: LRUNode<Key, Value>] = [:]
    
    // Sentinel nodes for cleaner edge case handling
    private let head: LRUNode<Key, Value>
    private let tail: LRUNode<Key, Value>
    
    // Statistics tracking
    private var hitCount: Int = 0
    private var missCount: Int = 0
    private var evictionCount: Int = 0
    
    /// Statistics about cache performance
    public var statistics: CacheStatistics {
        CacheStatistics(
            hits: hitCount,
            misses: missCount,
            evictions: evictionCount,
            currentSize: nodeMap.count,
            maxSize: maxSize
        )
    }
    
    public init(maxSize: Int) {
        self.maxSize = max(1, maxSize)
        
        // Create sentinel nodes
        self.head = LRUNode(key: "" as! Key, value: "" as! Value)
        self.tail = LRUNode(key: "" as! Key, value: "" as! Value)
        self.head.next = self.tail
        self.tail.prev = self.head
    }
    
    /// Get a value from the cache - O(1) operation
    public func get(_ key: Key) -> Value? {
        guard let node = nodeMap[key] else {
            missCount += 1
            logger.trace("Cache miss for key: \(String(describing: key))")
            return nil
        }
        
        // Move to front (most recently used)
        moveToFront(node)
        hitCount += 1
        logger.trace("Cache hit for key: \(String(describing: key))")
        return node.value
    }
    
    /// Set a value in the cache - O(1) operation
    public func set(_ key: Key, value: Value) {
        if let existingNode = nodeMap[key] {
            // Update existing value
            existingNode.value = value
            moveToFront(existingNode)
            logger.trace("Updated existing cache entry")
        } else {
            // Create new node
            let newNode = LRUNode(key: key, value: value)
            nodeMap[key] = newNode
            addToFront(newNode)
            
            // Evict if necessary
            if nodeMap.count > maxSize {
                evictLRU()
            }
            
            logger.trace("Added new cache entry, size: \(nodeMap.count)")
        }
    }
    
    /// Remove a specific key from the cache - O(1) operation
    public func remove(_ key: Key) {
        guard let node = nodeMap[key] else { return }
        
        removeNode(node)
        nodeMap.removeValue(forKey: key)
        
        logger.trace("Removed cache entry")
    }
    
    /// Clear all entries from the cache
    public func clear() {
        let previousSize = nodeMap.count
        nodeMap.removeAll()
        
        // Reset the linked list
        head.next = tail
        tail.prev = head
        
        logger.debug("Cache cleared, previous size: \(previousSize)")
    }
    
    /// Check if a key exists in the cache
    public func contains(_ key: Key) -> Bool {
        nodeMap[key] != nil
    }
    
    /// Get the current size of the cache
    public var count: Int {
        nodeMap.count
    }
    
    /// Get all keys currently in the cache (ordered from MRU to LRU)
    public var keys: [Key] {
        var result: [Key] = []
        var current = head.next
        
        while current !== tail {
            result.append(current!.key)
            current = current!.next
        }
        
        return result
    }
    
    // MARK: - Private Helpers
    
    /// Move a node to the front of the list (most recently used position)
    private func moveToFront(_ node: LRUNode<Key, Value>) {
        removeNode(node)
        addToFront(node)
    }
    
    /// Add a node to the front of the list
    private func addToFront(_ node: LRUNode<Key, Value>) {
        node.prev = head
        node.next = head.next
        head.next?.prev = node
        head.next = node
    }
    
    /// Remove a node from its current position in the list
    private func removeNode(_ node: LRUNode<Key, Value>) {
        node.prev?.next = node.next
        node.next?.prev = node.prev
    }
    
    /// Evict the least recently used item
    private func evictLRU() {
        guard let lruNode = tail.prev, lruNode !== head else { return }
        
        removeNode(lruNode)
        nodeMap.removeValue(forKey: lruNode.key)
        evictionCount += 1
        
        logger.trace("Evicted LRU item, new size: \(nodeMap.count)")
    }
}

/// Performance-optimized embedding cache with memory-aware eviction
public actor OptimizedEmbeddingCache {
    private let logger = Logger(subsystem: "EmbedKit", category: "OptimizedEmbeddingCache")
    private let cache: OptimizedLRUCache<String, CachedEmbedding>
    private let maxMemoryBytes: Int
    private var currentMemoryBytes: Int = 0
    
    public struct CachedEmbedding: Sendable {
        public let embedding: EmbeddingVector
        public let modelIdentifier: ModelIdentifier
        public let timestamp: Date
        public let byteSize: Int
        
        init(embedding: EmbeddingVector, modelIdentifier: ModelIdentifier) {
            self.embedding = embedding
            self.modelIdentifier = modelIdentifier
            self.timestamp = Date()
            // Exact byte size calculation
            self.byteSize = embedding.dimensions * MemoryLayout<Float>.size + 64
        }
    }
    
    public init(maxEntries: Int = 10000, maxMemoryMB: Int = 100) {
        self.cache = OptimizedLRUCache(maxSize: maxEntries)
        self.maxMemoryBytes = maxMemoryMB * 1024 * 1024
    }
    
    /// Generate a cache key using xxHash for better distribution
    public static func cacheKey(for text: String, modelIdentifier: ModelIdentifier) -> String {
        // Use a more efficient hash combining method
        var hasher = Hasher()
        hasher.combine(text)
        hasher.combine(modelIdentifier.rawValue)
        return "\(modelIdentifier.rawValue):\(hasher.finalize())"
    }
    
    /// Get an embedding from the cache
    public func get(text: String, modelIdentifier: ModelIdentifier) async -> EmbeddingVector? {
        let key = Self.cacheKey(for: text, modelIdentifier: modelIdentifier)
        
        if let cached = await cache.get(key) {
            // Verify model match
            guard cached.modelIdentifier == modelIdentifier else {
                await cache.remove(key)
                return nil
            }
            
            return cached.embedding
        }
        
        return nil
    }
    
    /// Store an embedding in the cache with memory pressure handling
    public func set(text: String, modelIdentifier: ModelIdentifier, embedding: EmbeddingVector) async {
        let key = Self.cacheKey(for: text, modelIdentifier: modelIdentifier)
        let cached = CachedEmbedding(embedding: embedding, modelIdentifier: modelIdentifier)
        
        // Proactive memory management
        if currentMemoryBytes + cached.byteSize > maxMemoryBytes {
            await handleMemoryPressure(requiredBytes: cached.byteSize)
        }
        
        await cache.set(key, value: cached)
        currentMemoryBytes += cached.byteSize
    }
    
    /// Handle memory pressure by evicting entries
    private func handleMemoryPressure(requiredBytes: Int) async {
        logger.debug("Memory pressure detected, need \(requiredBytes) bytes")
        
        let targetMemory = maxMemoryBytes * 3 / 4 // Target 75% capacity
        var freedBytes = 0
        
        // Get keys in LRU order and remove until we have enough space
        let keysToCheck = await cache.keys.reversed() // Start from LRU end
        
        for key in keysToCheck {
            if currentMemoryBytes - freedBytes <= targetMemory {
                break
            }
            
            if let cached = await cache.get(key) {
                await cache.remove(key)
                freedBytes += cached.byteSize
            }
        }
        
        currentMemoryBytes -= freedBytes
        logger.debug("Freed \(freedBytes) bytes from cache")
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
    
    /// Batch preload with optimized memory handling
    public func preload(texts: [String], modelIdentifier: ModelIdentifier, embeddings: [EmbeddingVector]) async {
        guard texts.count == embeddings.count else {
            logger.error("Mismatch between texts and embeddings count")
            return
        }
        
        // Calculate total memory needed
        let totalBytes = embeddings.reduce(0) { sum, embedding in
            sum + embedding.dimensions * MemoryLayout<Float>.size + 64
        }
        
        // Ensure we have space
        if currentMemoryBytes + totalBytes > maxMemoryBytes {
            await handleMemoryPressure(requiredBytes: totalBytes)
        }
        
        // Batch insert
        for (text, embedding) in zip(texts, embeddings) {
            await set(text: text, modelIdentifier: modelIdentifier, embedding: embedding)
        }
        
        logger.debug("Preloaded \(texts.count) embeddings into cache")
    }
}