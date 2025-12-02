// EmbedKit - Token Cache

import Foundation

public struct TokenCacheStats: Sendable {
    public let hits: Int
    public let misses: Int
    public var total: Int { hits + misses }
    public var hitRate: Double { total > 0 ? Double(hits) / Double(total) : 0 }
}

public actor TokenCache<Key: Hashable & Sendable, Value: Sendable> {
    private let capacity: Int

    // LRU state: doubly-linked list via key references
    private struct Entry {
        var value: Value
        var prev: Key?
        var next: Key?
    }
    private var map: [Key: Entry] = [:]
    private var head: Key? // Most recently used
    private var tail: Key? // Least recently used

    // Stats
    private var hits = 0
    private var misses = 0

    public init(capacity: Int) {
        precondition(capacity > 0, "capacity must be > 0")
        self.capacity = capacity
        self.map.reserveCapacity(capacity)
    }

    public func get(_ key: Key) -> Value? {
        guard var entry = map[key] else { misses &+= 1; return nil }
        hits &+= 1
        moveToHead(key: key, entry: &entry)
        return entry.value
    }

    public func put(_ key: Key, _ value: Value) {
        if var entry = map[key] {
            entry.value = value
            map[key] = entry
            moveToHead(key: key, entry: &entry)
            return
        }

        // Evict if at capacity
        if map.count >= capacity, let lruKey = tail {
            remove(key: lruKey)
        }

        // Insert new entry at head
        let newEntry = Entry(value: value, prev: nil, next: head)
        if let h = head { map[h]?.prev = key }
        head = key
        if tail == nil { tail = key }
        map[key] = newEntry
    }

    public func reset() {
        map.removeAll(keepingCapacity: true)
        head = nil
        tail = nil
        hits = 0
        misses = 0
    }

    public func stats() -> TokenCacheStats { TokenCacheStats(hits: hits, misses: misses) }

    // MARK: - LRU internals

    private func moveToHead(key: Key, entry: inout Entry) {
        if head == key { return }
        // Unlink current position
        if let p = entry.prev { map[p]?.next = entry.next }
        if let n = entry.next { map[n]?.prev = entry.prev }
        if tail == key { tail = entry.prev }

        // Insert at head
        entry.prev = nil
        entry.next = head
        if let h = head { map[h]?.prev = key }
        head = key
        map[key] = entry
        if tail == nil { tail = key }
    }

    private func remove(key: Key) {
        guard let entry = map.removeValue(forKey: key) else { return }
        if let p = entry.prev { map[p]?.next = entry.next }
        if let n = entry.next { map[n]?.prev = entry.prev }
        if head == key { head = entry.next }
        if tail == key { tail = entry.prev }
    }
}
