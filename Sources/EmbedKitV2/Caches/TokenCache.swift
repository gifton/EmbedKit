// EmbedKitV2 - Token Cache (Scaffold)

import Foundation

public struct TokenCacheStats: Sendable {
    public let hits: Int
    public let misses: Int
    public var total: Int { hits + misses }
    public var hitRate: Double { total > 0 ? Double(hits) / Double(total) : 0 }
}

public actor TokenCache<Key: Hashable & Sendable, Value: Sendable> {
    private let capacity: Int
    private var storage: [Key: Value] = [:]
    private var hits = 0
    private var misses = 0

    public init(capacity: Int) {
        precondition(capacity > 0, "capacity must be > 0")
        self.capacity = capacity
        self.storage.reserveCapacity(capacity)
    }

    public func get(_ key: Key) -> Value? {
        if let v = storage[key] { hits += 1; return v }
        misses += 1
        return nil
    }

    public func put(_ key: Key, _ value: Value) {
        // Scaffold: no eviction yet, just store until capacity
        if storage.count < capacity || storage[key] != nil {
            storage[key] = value
        } else {
            // Simple fallback: drop a random element (to be replaced with LRU)
            if let firstKey = storage.keys.first { storage.removeValue(forKey: firstKey) }
            storage[key] = value
        }
    }

    public func reset() {
        storage.removeAll(keepingCapacity: true)
        hits = 0; misses = 0
    }

    public func stats() -> TokenCacheStats { TokenCacheStats(hits: hits, misses: misses) }
}

