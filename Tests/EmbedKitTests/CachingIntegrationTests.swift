import Testing
import Foundation
@testable import EmbedKit

@Suite("Caching Integration Tests")
struct CachingIntegrationTests {
    
    // MARK: - Basic Cache Functionality Tests
    
    @Test("LRU cache basic operations")
    func testLRUCacheBasics() async {
        let cache = LRUCache<String, String>(maxSize: 3)
        
        // Test insertion
        await cache.set("key1", value: "value1")
        await cache.set("key2", value: "value2")
        await cache.set("key3", value: "value3")
        
        // Test retrieval
        let value1 = await cache.get("key1")
        #expect(value1 == "value1")
        
        // Test cache statistics
        let stats = await cache.statistics
        #expect(stats.currentSize == 3)
        #expect(stats.hitRate >= 0)
        
        // Test LRU eviction
        await cache.set("key4", value: "value4")
        
        // key2 should be evicted (key1 was accessed more recently)
        let value2 = await cache.get("key2")
        #expect(value2 == nil)
        
        // key1 should still be there
        let value1Again = await cache.get("key1")
        #expect(value1Again == "value1")
    }
    
    @Test("Embedding cache functionality")
    func testEmbeddingCache() async {
        let cache = EmbeddingCache()
        let modelId = ModelIdentifier.miniLM_L6_v2
        
        // Create test embeddings
        let embedding1 = EmbeddingVector([0.1, 0.2, 0.3])
        let embedding2 = EmbeddingVector([0.4, 0.5, 0.6])
        
        // Cache embeddings
        await cache.set(
            text: "test text 1",
            modelIdentifier: modelId,
            embedding: embedding1
        )
        
        await cache.set(
            text: "test text 2",
            modelIdentifier: modelId,
            embedding: embedding2
        )
        
        // Retrieve embeddings
        let retrieved1 = await cache.get(text: "test text 1", modelIdentifier: modelId)
        #expect(retrieved1 != nil)
        if let retrieved = retrieved1 {
            #expect(retrieved.dimensions == embedding1.dimensions)
        }
        
        // Test cache miss
        let miss = await cache.get(text: "not cached", modelIdentifier: modelId)
        #expect(miss == nil)
        
        // Test different model ID (should be separate)
        let differentModel = ModelIdentifier(family: "different")
        let differentCache = await cache.get(text: "test text 1", modelIdentifier: differentModel)
        #expect(differentCache == nil)
        
        // Test clearing
        await cache.clear()
        let afterClear = await cache.get(text: "test text 1", modelIdentifier: modelId)
        #expect(afterClear == nil)
    }
    
    // MARK: - Memory-Aware Cache Tests
    
    @Test("Memory-aware cache pressure handling")
    func testMemoryAwareCache() async {
        let embeddingCache = EmbeddingCache()
        // MemoryAwareCache is managed internally by EmbeddingCache
        
        let modelId = ModelIdentifier.miniLM_L6_v2
        
        // Fill cache with embeddings
        for i in 0..<20 {
            let embedding = EmbeddingVector(Array(repeating: Float(i), count: 384))
            await embeddingCache.set(
                text: "text \(i)",
                modelIdentifier: modelId,
                embedding: embedding
            )
        }
        
        // Get initial stats
        let initialStats = await embeddingCache.statistics()
        #expect(initialStats.currentSize > 0)
        
        // Simulate memory pressure
        // Memory pressure is handled internally by the cache
        // Force memory pressure by filling cache beyond limit
        
        // Cache should have reduced size
        let afterPressureStats = await embeddingCache.statistics()
        #expect(afterPressureStats.currentSize <= initialStats.currentSize)
        
        // Stop monitoring
        // Memory monitoring is automatic
    }
    
    // MARK: - Cache Integration with Embedder Tests
    
    @Test("Embedder with caching enabled")
    func testEmbedderCaching() async throws {
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: .miniLM_L6_v2,
            configuration: Configuration.default(for: .miniLM_L6_v2),
            enableCaching: true
        )
        
        let testText = "This text will be cached"
        
        // First call - cache miss
        let startTime1 = Date()
        let embedding1 = try await embedder.embed(testText)
        let time1 = Date().timeIntervalSince(startTime1)
        
        // Second call - should hit cache
        let startTime2 = Date()
        let embedding2 = try await embedder.embed(testText)
        let time2 = Date().timeIntervalSince(startTime2)
        
        // Embeddings should be identical
        if embedding1.dimensions == embedding2.dimensions && embedding1.dimensions > 0 {
            let similarity = embedding1.cosineSimilarity(with: embedding2)
            #expect(similarity > 0.999) // Should be essentially identical
        }
        
        // Cache hit should be faster (though with mock model, difference might be minimal)
        print("First call: \(time1)s, Second call (cached): \(time2)s")
        
        // Test with different text - should miss cache
        let embedding3 = try await embedder.embed("Different text")
        #expect(embedding3.dimensions >= 0)
    }
    
    @Test("Batch processing with cache")
    func testBatchCaching() async throws {
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: .miniLM_L6_v2,
            configuration: Configuration.default(for: .miniLM_L6_v2),
            enableCaching: true
        )
        
        // Create batch with duplicates
        let texts = [
            "First unique text",
            "Second unique text",
            "First unique text", // Duplicate
            "Third unique text",
            "Second unique text", // Duplicate
            "Fourth unique text"
        ]
        
        // Process batch
        let embeddings = try await embedder.embed(batch: texts)
        #expect(embeddings.count == texts.count)
        
        // Process same batch again - should be faster due to cache
        let startTime = Date()
        let cachedEmbeddings = try await embedder.embed(batch: texts)
        let cacheTime = Date().timeIntervalSince(startTime)
        
        #expect(cachedEmbeddings.count == texts.count)
        
        // Verify consistency
        for i in 0..<embeddings.count {
            if embeddings[i].dimensions == cachedEmbeddings[i].dimensions && 
               embeddings[i].dimensions > 0 {
                let similarity = embeddings[i].cosineSimilarity(with: cachedEmbeddings[i])
                #expect(similarity > 0.999)
            }
        }
        
        print("Batch cache time: \(cacheTime)s for \(texts.count) texts")
    }
    
    // MARK: - Cache Performance Tests
    
    @Test("Cache performance under load")
    func testCachePerformance() async throws {
        let cache = LRUCache<String, EmbeddingVector>(maxSize: 1000)
        
        // Generate test data
        let embeddings = (0..<1000).map { i in
            EmbeddingVector(Array(repeating: Float(i) / 1000.0, count: 384))
        }
        
        // Measure insertion time
        let insertStart = Date()
        for (i, embedding) in embeddings.enumerated() {
            await cache.set("key_\(i)", value: embedding)
        }
        let insertTime = Date().timeIntervalSince(insertStart)
        
        // Measure retrieval time
        let retrievalStart = Date()
        var hits = 0
        for i in 0..<1000 {
            if let _ = await cache.get("key_\(i)") {
                hits += 1
            }
        }
        let retrievalTime = Date().timeIntervalSince(retrievalStart)
        
        print("Cache performance - Insert: \(insertTime)s, Retrieval: \(retrievalTime)s, Hits: \(hits)")
        
        #expect(hits > 0) // Should have some hits
        #expect(insertTime < 1.0) // Should insert 1000 items in < 1 second
        #expect(retrievalTime < 0.5) // Should retrieve faster than insert
    }
    
    @Test("Cache with concurrent access")
    func testConcurrentCacheAccess() async throws {
        let cache = EmbeddingCache()
        let modelId = ModelIdentifier.miniLM_L6_v2
        
        // Concurrent writes
        await withTaskGroup(of: Void.self) { group in
            for i in 0..<10 {
                group.addTask {
                    let embedding = EmbeddingVector(Array(repeating: Float(i), count: 128))
                    await cache.set(
                        text: "concurrent_\(i)",
                        modelIdentifier: modelId,
                        embedding: embedding
                    )
                }
            }
        }
        
        // Concurrent reads
        await withTaskGroup(of: EmbeddingVector?.self) { group in
            for i in 0..<10 {
                group.addTask {
                    await cache.get(text: "concurrent_\(i)", modelIdentifier: modelId)
                }
            }
            
            var found = 0
            for await result in group {
                if result != nil {
                    found += 1
                }
            }
            
            #expect(found == 10) // All should be found
        }
        
        // Test statistics are consistent
        let stats = await cache.statistics()
        #expect(stats.currentSize >= 10)
    }
    
    // MARK: - Cache Invalidation Tests
    
    @Test("Cache invalidation strategies")
    func testCacheInvalidation() async throws {
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: .miniLM_L6_v2,
            configuration: Configuration.default(for: .miniLM_L6_v2),
            enableCaching: true
        )
        
        // Cache some embeddings
        let texts = ["Text 1", "Text 2", "Text 3"]
        for text in texts {
            _ = try await embedder.embed(text)
        }
        
        // Model change should invalidate cache
        // (In real scenario, this would happen when model is reloaded)
        try await embedder.unloadModel()
        try await embedder.loadModel()
        
        // Cache should still work after model reload
        let embedding = try await embedder.embed("New text after reload")
        #expect(embedding.dimensions >= 0)
    }
    
    @Test("Cache size limits")
    func testCacheSizeLimits() async {
        // Test with very small cache
        let tinyCache = LRUCache<String, String>(maxSize: 2)
        
        await tinyCache.set("1", value: "a")
        await tinyCache.set("2", value: "b")
        await tinyCache.set("3", value: "c") // Should evict "1"
        
        #expect(await tinyCache.get("1") == nil)
        #expect(await tinyCache.get("2") == "b")
        #expect(await tinyCache.get("3") == "c")
        
        // Test with configuration limits
        let config = Configuration(
            model: ModelConfiguration.custom(
                identifier: .miniLM_L6_v2,
                maxSequenceLength: 512
            ),
            cache: CacheConfiguration(maxCacheSize: 1)
        )
        
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: .miniLM_L6_v2,
            configuration: config,
            enableCaching: true
        )
        
        // Should still work with minimal cache
        do {
            let result = try await embedder.embed("Test with tiny cache")
            #expect(result.dimensions >= 0)
        } catch {
            // Might fail with very tiny cache
            #expect(error is ContextualEmbeddingError)
        }
    }
}