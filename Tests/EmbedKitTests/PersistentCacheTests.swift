// Tests for Persistent Cache - Week 5 Batch 1
import Testing
import Foundation
@testable import EmbedKit

// MARK: - Test Helpers

/// Create a test embedding with given properties.
func makeTestEmbedding(
    vector: [Float]? = nil,
    modelID: ModelID = ModelID(provider: "test", name: "model", version: "1.0"),
    tokenCount: Int = 5
) -> Embedding {
    let vec = vector ?? (0..<128).map { Float($0) / 128.0 }
    return Embedding(
        vector: vec,
        metadata: EmbeddingMetadata(
            modelID: modelID,
            tokenCount: tokenCount,
            processingTime: 0.01,
            normalized: true
        )
    )
}

// MARK: - Cache Types Tests

@Suite("Persistent Cache - Types")
struct CacheTypesTests {

    @Test("CacheConfiguration has sensible defaults")
    func configurationDefaults() {
        let config = CacheConfiguration.default

        #expect(config.maxEntries == 100_000)
        #expect(config.maxSizeBytes == 500 * 1024 * 1024)
        #expect(config.autoEvict == true)
        #expect(config.enableWAL == true)
    }

    @Test("CacheConfiguration inMemory preset")
    func configurationInMemory() {
        let config = CacheConfiguration.inMemory

        #expect(config.maxEntries == 10_000)
        #expect(config.enableWAL == false)
    }

    @Test("CacheStatistics initializes to zero")
    func statisticsInit() {
        let stats = CacheStatistics()

        #expect(stats.totalLookups == 0)
        #expect(stats.exactHits == 0)
        #expect(stats.semanticHits == 0)
        #expect(stats.misses == 0)
        #expect(stats.hitRate == 0)
    }

    @Test("CacheStatistics calculates hit rate correctly")
    func statisticsHitRate() {
        var stats = CacheStatistics()
        stats.totalLookups = 100
        stats.exactHits = 60
        stats.semanticHits = 10
        stats.misses = 30

        #expect(abs(stats.hitRate - 0.7) < 0.001)
        #expect(abs(stats.exactHitRate - 0.6) < 0.001)
    }

    @Test("TextHasher produces consistent hashes")
    func textHasherConsistency() {
        let text = "Hello, World!"

        let hash1 = TextHasher.hash(text)
        let hash2 = TextHasher.hash(text)

        #expect(hash1 == hash2)
    }

    @Test("TextHasher normalizes text before hashing")
    func textHasherNormalization() {
        let text1 = "Hello World"
        let text2 = "  hello   world  "
        let text3 = "HELLO WORLD"

        let hash1 = TextHasher.hash(text1)
        let hash2 = TextHasher.hash(text2)
        let hash3 = TextHasher.hash(text3)

        // All should normalize to same hash
        #expect(hash1 == hash2)
        #expect(hash2 == hash3)
    }

    @Test("VectorSerializer roundtrips correctly")
    func vectorSerializerRoundtrip() {
        let original: [Float] = [1.0, 2.5, -3.14, 0.0, 999.99]

        let data = VectorSerializer.serialize(original)
        let restored = VectorSerializer.deserialize(data, dimensions: original.count)

        #expect(restored != nil)
        #expect(restored! == original)
    }

    @Test("CacheResult isHit property")
    func cacheResultIsHit() {
        let cached = CachedEmbedding(
            id: 1,
            text: "test",
            textHash: "abc",
            embedding: makeTestEmbedding(),
            modelID: ModelID(provider: "test", name: "m", version: "1"),
            createdAt: Date(),
            accessedAt: Date(),
            accessCount: 1
        )

        #expect(CacheResult.exactMatch(cached).isHit == true)
        #expect(CacheResult.semanticMatch(cached, similarity: 0.95).isHit == true)
        #expect(CacheResult.miss.isHit == false)
    }
}

// MARK: - Persistent Cache Core Tests

@Suite("Persistent Cache - Core Operations")
struct PersistentCacheCoreTests {

    @Test("Cache initializes with in-memory database")
    func initInMemory() async throws {
        let cache = try await PersistentCache(inMemory: .inMemory)

        #expect(cache.isInMemory == true)

        let stats = await cache.statistics
        #expect(stats.entryCount == 0)
    }

    @Test("Store and retrieve embedding")
    func storeAndRetrieve() async throws {
        let cache = try await PersistentCache(inMemory: .inMemory)
        let modelID = ModelID(provider: "test", name: "model", version: "1.0")
        let embedding = makeTestEmbedding(modelID: modelID)

        try await cache.store(text: "Hello world", embedding: embedding)

        let result = await cache.get(text: "Hello world", modelID: modelID)

        #expect(result.isHit == true)
        if case .exactMatch(let cached) = result {
            #expect(cached.embedding.dimensions == embedding.dimensions)
            #expect(cached.embedding.vector == embedding.vector)
        } else {
            Issue.record("Expected exact match")
        }
    }

    @Test("Cache miss for unknown text")
    func cacheMiss() async throws {
        let cache = try await PersistentCache(inMemory: .inMemory)
        let modelID = ModelID(provider: "test", name: "model", version: "1.0")

        let result = await cache.get(text: "Unknown text", modelID: modelID)

        #expect(result.isHit == false)
        #expect(result.embedding == nil)
    }

    @Test("Cache miss for wrong model")
    func cacheMissWrongModel() async throws {
        let cache = try await PersistentCache(inMemory: .inMemory)
        let modelID1 = ModelID(provider: "test", name: "model1", version: "1.0")
        let modelID2 = ModelID(provider: "test", name: "model2", version: "1.0")
        let embedding = makeTestEmbedding(modelID: modelID1)

        try await cache.store(text: "Hello world", embedding: embedding)

        let result = await cache.get(text: "Hello world", modelID: modelID2)

        #expect(result.isHit == false)
    }

    @Test("Duplicate store is idempotent")
    func duplicateStore() async throws {
        let cache = try await PersistentCache(inMemory: .inMemory)
        let modelID = ModelID(provider: "test", name: "model", version: "1.0")
        let embedding = makeTestEmbedding(modelID: modelID)

        try await cache.store(text: "Hello world", embedding: embedding)
        try await cache.store(text: "Hello world", embedding: embedding)

        let stats = await cache.statistics
        #expect(stats.entryCount == 1)
    }

    @Test("Store batch adds multiple entries")
    func storeBatch() async throws {
        let cache = try await PersistentCache(inMemory: .inMemory)
        let modelID = ModelID(provider: "test", name: "model", version: "1.0")

        let entries: [(text: String, embedding: Embedding)] = [
            ("Text one", makeTestEmbedding(modelID: modelID)),
            ("Text two", makeTestEmbedding(modelID: modelID)),
            ("Text three", makeTestEmbedding(modelID: modelID))
        ]

        try await cache.storeBatch(entries)

        let stats = await cache.statistics
        #expect(stats.entryCount == 3)
    }
}

// MARK: - Cache Statistics Tests

@Suite("Persistent Cache - Statistics")
struct PersistentCacheStatisticsTests {

    @Test("Statistics track lookups")
    func trackLookups() async throws {
        let cache = try await PersistentCache(inMemory: .inMemory)
        let modelID = ModelID(provider: "test", name: "model", version: "1.0")
        let embedding = makeTestEmbedding(modelID: modelID)

        try await cache.store(text: "Test", embedding: embedding)

        // Hit
        _ = await cache.get(text: "Test", modelID: modelID)
        // Miss
        _ = await cache.get(text: "Unknown", modelID: modelID)
        // Hit
        _ = await cache.get(text: "Test", modelID: modelID)

        let stats = await cache.statistics
        #expect(stats.totalLookups == 3)
        #expect(stats.exactHits == 2)
        #expect(stats.misses == 1)
    }

    @Test("Statistics reset clears counters")
    func statisticsReset() async throws {
        let cache = try await PersistentCache(inMemory: .inMemory)
        let modelID = ModelID(provider: "test", name: "model", version: "1.0")

        _ = await cache.get(text: "Test", modelID: modelID)
        _ = await cache.get(text: "Test2", modelID: modelID)

        await cache.resetStatistics()

        let stats = await cache.statistics
        #expect(stats.totalLookups == 0)
        #expect(stats.misses == 0)
    }
}

// MARK: - Cache Eviction Tests

@Suite("Persistent Cache - Eviction")
struct PersistentCacheEvictionTests {

    @Test("Clear removes all entries")
    func clearCache() async throws {
        let cache = try await PersistentCache(inMemory: .inMemory)
        let modelID = ModelID(provider: "test", name: "model", version: "1.0")

        for i in 0..<10 {
            try await cache.store(text: "Text \(i)", embedding: makeTestEmbedding(modelID: modelID))
        }

        var stats = await cache.statistics
        #expect(stats.entryCount == 10)

        try await cache.clear()

        stats = await cache.statistics
        #expect(stats.entryCount == 0)
    }

    @Test("Remove deletes specific entry")
    func removeEntry() async throws {
        let cache = try await PersistentCache(inMemory: .inMemory)
        let modelID = ModelID(provider: "test", name: "model", version: "1.0")

        try await cache.store(text: "Keep me", embedding: makeTestEmbedding(modelID: modelID))
        try await cache.store(text: "Delete me", embedding: makeTestEmbedding(modelID: modelID))

        let hash = TextHasher.hash("Delete me")
        let removed = try await cache.remove(textHash: hash)

        #expect(removed == true)

        let stats = await cache.statistics
        #expect(stats.entryCount == 1)

        // Verify correct one was removed
        let result = await cache.get(text: "Keep me", modelID: modelID)
        #expect(result.isHit == true)

        let result2 = await cache.get(text: "Delete me", modelID: modelID)
        #expect(result2.isHit == false)
    }

    @Test("LRU eviction removes oldest accessed")
    func lruEviction() async throws {
        let cache = try await PersistentCache(inMemory: .inMemory)
        let modelID = ModelID(provider: "test", name: "model", version: "1.0")

        // Add entries with some delay
        for i in 0..<5 {
            try await cache.store(text: "Text \(i)", embedding: makeTestEmbedding(modelID: modelID))
        }

        // Access the first one to make it "recent"
        _ = await cache.get(text: "Text 0", modelID: modelID)

        // Evict 3 oldest
        let evicted = try await cache.evictLRU(count: 3)

        #expect(evicted == 3)

        let stats = await cache.statistics
        #expect(stats.entryCount == 2)

        // Text 0 should still be there (was accessed recently)
        let result = await cache.get(text: "Text 0", modelID: modelID)
        #expect(result.isHit == true)
    }
}

// MARK: - Cache Convenience Tests

@Suite("Persistent Cache - Convenience")
struct PersistentCacheConvenienceTests {

    @Test("Contains checks existence")
    func containsCheck() async throws {
        let cache = try await PersistentCache(inMemory: .inMemory)
        let modelID = ModelID(provider: "test", name: "model", version: "1.0")

        try await cache.store(text: "Exists", embedding: makeTestEmbedding(modelID: modelID))

        let exists = await cache.contains(text: "Exists", modelID: modelID)
        let notExists = await cache.contains(text: "Not exists", modelID: modelID)

        #expect(exists == true)
        #expect(notExists == false)
    }

    @Test("GetEmbedding returns just the embedding")
    func getEmbeddingOnly() async throws {
        let cache = try await PersistentCache(inMemory: .inMemory)
        let modelID = ModelID(provider: "test", name: "model", version: "1.0")
        let original = makeTestEmbedding(modelID: modelID)

        try await cache.store(text: "Test", embedding: original)

        let retrieved = await cache.getEmbedding(text: "Test", modelID: modelID)

        #expect(retrieved != nil)
        #expect(retrieved!.vector == original.vector)
    }

    @Test("Warm cache stores multiple entries")
    func warmCache() async throws {
        let cache = try await PersistentCache(inMemory: .inMemory)
        let modelID = ModelID(provider: "test", name: "model", version: "1.0")

        let entries: [(text: String, embedding: Embedding)] = [
            ("Warm 1", makeTestEmbedding(modelID: modelID)),
            ("Warm 2", makeTestEmbedding(modelID: modelID)),
            ("Warm 3", makeTestEmbedding(modelID: modelID))
        ]

        try await cache.warmCache(entries: entries)

        let stats = await cache.statistics
        #expect(stats.entryCount == 3)
    }
}

// MARK: - Text Normalization Tests

@Suite("Persistent Cache - Text Normalization")
struct PersistentCacheNormalizationTests {

    @Test("Normalized text matches regardless of case")
    func caseInsensitiveMatch() async throws {
        let cache = try await PersistentCache(inMemory: .inMemory)
        let modelID = ModelID(provider: "test", name: "model", version: "1.0")

        try await cache.store(text: "Hello World", embedding: makeTestEmbedding(modelID: modelID))

        let result1 = await cache.get(text: "hello world", modelID: modelID)
        let result2 = await cache.get(text: "HELLO WORLD", modelID: modelID)
        let result3 = await cache.get(text: "HeLLo WoRLD", modelID: modelID)

        #expect(result1.isHit == true)
        #expect(result2.isHit == true)
        #expect(result3.isHit == true)
    }

    @Test("Normalized text matches regardless of whitespace")
    func whitespaceNormalizedMatch() async throws {
        let cache = try await PersistentCache(inMemory: .inMemory)
        let modelID = ModelID(provider: "test", name: "model", version: "1.0")

        try await cache.store(text: "Hello World", embedding: makeTestEmbedding(modelID: modelID))

        let result1 = await cache.get(text: "  Hello   World  ", modelID: modelID)
        let result2 = await cache.get(text: "Hello\nWorld", modelID: modelID)
        let result3 = await cache.get(text: "\t Hello \t World \t", modelID: modelID)

        #expect(result1.isHit == true)
        #expect(result2.isHit == true)
        #expect(result3.isHit == true)
    }
}

// MARK: - Semantic Deduplication Tests

@Suite("Persistent Cache - Semantic Dedup")
struct PersistentCacheSemanticTests {

    @Test("Semantic match finds similar embeddings")
    func semanticMatch() async throws {
        var config = CacheConfiguration.inMemory
        config.enableSemanticDedup = true
        config.deduplicationThreshold = 0.9

        let cache = try await PersistentCache(inMemory: config)
        let modelID = ModelID(provider: "test", name: "model", version: "1.0")

        // Store a normalized vector
        let originalVec: [Float] = [0.6, 0.8, 0.0, 0.0]  // Unit vector
        let original = Embedding(
            vector: originalVec,
            metadata: EmbeddingMetadata(
                modelID: modelID,
                tokenCount: 5,
                processingTime: 0.01,
                normalized: true
            )
        )

        try await cache.store(text: "Original text", embedding: original)

        // Query with very similar vector
        let similarVec: [Float] = [0.61, 0.79, 0.01, 0.0]  // Slightly different
        let similar = Embedding(
            vector: similarVec,
            metadata: EmbeddingMetadata(
                modelID: modelID,
                tokenCount: 5,
                processingTime: 0.01,
                normalized: true
            )
        )

        let result = await cache.getSemantic(
            text: "Different text",
            embedding: similar,
            modelID: modelID,
            threshold: 0.9
        )

        // Should find semantic match since vectors are very similar
        if case .semanticMatch(_, let similarity) = result {
            #expect(similarity > 0.9)
        } else if case .exactMatch = result {
            // Also acceptable if exact match
        } else {
            // Vectors are similar enough that we should get a match
            // Calculate similarity to check
            let sim = original.similarity(to: similar)
            if sim >= 0.9 {
                Issue.record("Expected semantic match but got miss, similarity was \(sim)")
            }
        }
    }

    @Test("Semantic match respects threshold")
    func semanticMatchThreshold() async throws {
        var config = CacheConfiguration.inMemory
        config.enableSemanticDedup = true
        config.deduplicationThreshold = 0.99  // Very high threshold

        let cache = try await PersistentCache(inMemory: config)
        let modelID = ModelID(provider: "test", name: "model", version: "1.0")

        let originalVec: [Float] = [1.0, 0.0, 0.0, 0.0]
        let original = Embedding(
            vector: originalVec,
            metadata: EmbeddingMetadata(modelID: modelID, tokenCount: 5, processingTime: 0.01, normalized: true)
        )

        try await cache.store(text: "Original", embedding: original)

        // Query with different vector (90% similar but below 99% threshold)
        let differentVec: [Float] = [0.9, 0.436, 0.0, 0.0]  // ~90% similarity
        let different = Embedding(
            vector: differentVec,
            metadata: EmbeddingMetadata(modelID: modelID, tokenCount: 5, processingTime: 0.01, normalized: true)
        )

        let result = await cache.getSemantic(
            text: "Different",
            embedding: different,
            modelID: modelID,
            threshold: 0.99
        )

        // Should be miss because similarity is below 0.99
        #expect(result.isHit == false)
    }
}

// MARK: - Access Tracking Tests

@Suite("Persistent Cache - Access Tracking")
struct PersistentCacheAccessTests {

    @Test("Access count increments on retrieval")
    func accessCountIncrement() async throws {
        let cache = try await PersistentCache(inMemory: .inMemory)
        let modelID = ModelID(provider: "test", name: "model", version: "1.0")

        try await cache.store(text: "Test", embedding: makeTestEmbedding(modelID: modelID))

        // Access multiple times
        for _ in 0..<5 {
            _ = await cache.get(text: "Test", modelID: modelID)
        }

        // Get the cached entry
        let result = await cache.get(text: "Test", modelID: modelID)
        if case .exactMatch(let cached) = result {
            #expect(cached.accessCount >= 5)
        } else {
            Issue.record("Expected exact match")
        }
    }

    @Test("Last access time updates on retrieval")
    func accessTimeUpdate() async throws {
        let cache = try await PersistentCache(inMemory: .inMemory)
        let modelID = ModelID(provider: "test", name: "model", version: "1.0")

        try await cache.store(text: "Test", embedding: makeTestEmbedding(modelID: modelID))

        let result1 = await cache.get(text: "Test", modelID: modelID)
        guard case .exactMatch(let cached1) = result1 else {
            Issue.record("Expected exact match")
            return
        }
        let time1 = cached1.accessedAt

        // Small delay
        try await Task.sleep(nanoseconds: 10_000_000)  // 10ms

        let result2 = await cache.get(text: "Test", modelID: modelID)
        guard case .exactMatch(let cached2) = result2 else {
            Issue.record("Expected exact match")
            return
        }
        let time2 = cached2.accessedAt

        #expect(time2 > time1)
    }
}
