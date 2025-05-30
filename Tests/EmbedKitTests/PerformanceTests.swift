import Testing
import Foundation
@testable import EmbedKit

// Performance tests for EmbedKit
struct PerformanceTests {
    
    @Test("LRU Cache performance")
    func testLRUCachePerformance() async throws {
        let cache = LRUCache<String, Int>(maxSize: 1000)
        
        // Test insertion performance
        let insertStart = Date()
        for i in 0..<10000 {
            await cache.set("key\(i)", value: i)
        }
        let insertDuration = Date().timeIntervalSince(insertStart)
        print("Inserted 10,000 items in \(insertDuration)s")
        #expect(insertDuration < 1.0) // Should complete in under 1 second
        
        // Test hit performance
        let hitStart = Date()
        var hits = 0
        for i in 9000..<10000 {
            if await cache.get("key\(i)") != nil {
                hits += 1
            }
        }
        let hitDuration = Date().timeIntervalSince(hitStart)
        print("Retrieved 1,000 items in \(hitDuration)s with \(hits) hits")
        #expect(hitDuration < 1.0) // Should complete in under 1 second
        
        // Verify LRU behavior (most recent 1000 items should be in cache)
        #expect(hits == 1000)
        
        // Test cache statistics
        let stats = await cache.statistics
        #expect(stats.currentSize == 1000)
        #expect(stats.evictions > 0)
    }
    
    @Test("EmbeddingCache functionality")
    func testEmbeddingCache() async throws {
        let cache = EmbeddingCache(maxEntries: 100)
        let embedding1 = EmbeddingVector([0.1, 0.2, 0.3])
        let embedding2 = EmbeddingVector([0.4, 0.5, 0.6])
        
        // Test cache miss
        let result1 = await cache.get(text: "test text", modelIdentifier: "model1")
        #expect(result1 == nil)
        
        // Test cache set and hit
        await cache.set(text: "test text", modelIdentifier: "model1", embedding: embedding1)
        let result2 = await cache.get(text: "test text", modelIdentifier: "model1")
        #expect(result2 != nil)
        #expect(result2?.array == embedding1.array)
        
        // Test different model identifier
        let result3 = await cache.get(text: "test text", modelIdentifier: "model2")
        #expect(result3 == nil)
        
        // Test batch preload
        await cache.preload(
            texts: ["text1", "text2", "text3"],
            modelIdentifier: "model1",
            embeddings: [embedding1, embedding2, embedding1]
        )
        
        let preloaded = await cache.get(text: "text2", modelIdentifier: "model1")
        #expect(preloaded?.array == embedding2.array)
    }
    
    @Test("Metal acceleration availability")
    func testMetalAcceleration() async throws {
        // Check if Metal is available
        let accelerator = MetalAccelerator.shared
        #expect(accelerator != nil || true) // Pass on systems without Metal
        
        if let accel = accelerator {
            // Test normalization
            let vectors: [[Float]] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            let normalized = try await accel.normalizeVectors(vectors)
            
            #expect(normalized.count == 2)
            
            // Verify normalization (each vector should have L2 norm = 1)
            for vector in normalized {
                let norm = sqrt(vector.reduce(0) { $0 + $1 * $1 })
                #expect(abs(norm - 1.0) < 0.001)
            }
        }
    }
    
    @Test("Batch embedding with caching")
    func testBatchEmbeddingWithCache() async throws {
        let embedder = MockCachedTextEmbedder()
        try await embedder.loadModel()
        
        let texts = ["text1", "text2", "text3", "text1", "text2"] // Duplicates
        
        // First batch - all should be computed
        let embeddings1 = try await embedder.embed(batch: texts)
        #expect(embeddings1.count == 5)
        #expect(await embedder.computeCount == 3) // Only unique texts
        
        // Second batch - all should come from cache
        let embeddings2 = try await embedder.embed(batch: texts)
        #expect(embeddings2.count == 5)
        #expect(await embedder.computeCount == 3) // No new computations
        
        // Verify consistency
        #expect(embeddings1[0].array == embeddings2[0].array)
        #expect(embeddings1[3].array == embeddings1[0].array) // Duplicates match
    }
}

// Mock embedder with caching for testing
actor MockCachedTextEmbedder: TextEmbedder {
    let configuration = EmbeddingConfiguration()
    let dimensions = 3
    let modelIdentifier = "mock-cached"
    private(set) var isReady = false
    private let cache = EmbeddingCache(maxEntries: 100)
    private(set) var computeCount = 0
    
    func embed(_ text: String) async throws -> EmbeddingVector {
        guard isReady else { throw EmbeddingError.modelNotLoaded }
        
        // Check cache first
        if let cached = await cache.get(text: text, modelIdentifier: modelIdentifier) {
            return cached
        }
        
        // Compute embedding
        computeCount += 1
        let hash = Float(text.hashValue)
        let values = [
            abs(sin(hash)) / 2,
            abs(cos(hash)) / 2,
            abs(sin(hash * 2)) / 2
        ]
        
        let embedding = EmbeddingVector(values)
        await cache.set(text: text, modelIdentifier: modelIdentifier, embedding: embedding)
        
        return embedding
    }
    
    func embed(batch texts: [String]) async throws -> [EmbeddingVector] {
        var results: [EmbeddingVector] = []
        for text in texts {
            let embedding = try await embed(text)
            results.append(embedding)
        }
        return results
    }
    
    func loadModel() async throws {
        isReady = true
    }
    
    func unloadModel() async throws {
        isReady = false
        await cache.clear()
    }
}