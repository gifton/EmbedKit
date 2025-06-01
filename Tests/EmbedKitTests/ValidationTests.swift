import Testing
import Foundation
@testable import EmbedKit

/// Comprehensive validation tests for EmbedKit
@Suite("EmbedKit Validation Tests")
struct ValidationTests {
    let logger = EmbedKitLogger.custom("ValidationTests")
    
    // MARK: - Model Loading Tests
    
    @Test("Model loads correctly")
    func testModelLoading() async throws {
        logger.start("Model loading test")
        
        // Create a mock embedder
        let embedder = MockTextEmbedder(dimensions: 768)
        
        // Test loading
        try await embedder.loadModel()
        
        // Verify model is loaded
        #expect(await embedder.isModelLoaded == true)
        #expect(await embedder.dimensions == 768)
        
        // Test configuration
        let config = await embedder.configuration
        #expect(config.maxSequenceLength == 512)
        #expect(config.batchSize == 32)
        
        logger.success("Model loaded successfully with correct configuration")
    }
    
    @Test("Model unloading works")
    func testModelUnloading() async throws {
        let embedder = MockTextEmbedder(dimensions: 384)
        
        // Load and then unload
        try await embedder.loadModel()
        #expect(await embedder.isModelLoaded == true)
        
        try await embedder.unloadModel()
        #expect(await embedder.isModelLoaded == false)
        
        logger.success("Model unloading verified")
    }
    
    // MARK: - Embedding Generation Tests
    
    @Test("Single text embedding generation")
    func testSingleEmbedding() async throws {
        logger.start("Single embedding generation test")
        
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        
        // Generate embedding
        let text = "The quick brown fox jumps over the lazy dog"
        let embedding = try await embedder.embed(text)
        
        // Verify embedding properties
        #expect(embedding.dimensions == 768)
        #expect(embedding.vector.count == 768)
        
        // Check values are normalized
        let magnitude = embedding.magnitude()
        #expect(abs(magnitude - 1.0) < 0.01) // Should be close to 1.0
        
        // Verify embedding values are in reasonable range
        for value in embedding.vector {
            #expect(value >= -1.0 && value <= 1.0)
        }
        
        logger.success("Single embedding generated with correct dimensions and normalization")
    }
    
    @Test("Batch embedding generation")
    func testBatchEmbedding() async throws {
        logger.start("Batch embedding generation test")
        
        let embedder = MockTextEmbedder(dimensions: 384)
        try await embedder.loadModel()
        
        let texts = [
            "First text",
            "Second text with more words",
            "Third text that is even longer than the second one",
            "Fourth short text"
        ]
        
        let embeddings = try await embedder.embed(batch: texts)
        
        // Verify batch results
        #expect(embeddings.count == texts.count)
        
        for (index, embedding) in embeddings.enumerated() {
            #expect(embedding.dimensions == 384)
            #expect(embedding.vector.count == 384)
            
            // Each embedding should be unique
            if index > 0 {
                let similarity = embedding.cosineSimilarity(to: embeddings[0])
                #expect(similarity < 0.99) // Not identical
            }
        }
        
        logger.success("Batch embeddings generated correctly with unique vectors")
    }
    
    @Test("Empty text handling")
    func testEmptyTextEmbedding() async throws {
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        
        // Empty text should still generate an embedding
        let embedding = try await embedder.embed("")
        #expect(embedding.dimensions == 768)
        
        logger.info("Empty text handled gracefully")
    }
    
    @Test("Long text truncation")
    func testLongTextTruncation() async throws {
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        
        // Create text longer than max sequence length
        let longText = String(repeating: "word ", count: 1000)
        
        // Should not throw, but truncate
        let embedding = try await embedder.embed(longText)
        #expect(embedding.dimensions == 768)
        
        logger.info("Long text truncated and processed successfully")
    }
    
    // MARK: - Similarity Calculation Tests
    
    @Test("Cosine similarity calculation")
    func testCosineSimilarity() async throws {
        logger.start("Cosine similarity test")
        
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        
        // Test identical texts
        let text1 = "Machine learning is fascinating"
        let embedding1 = try await embedder.embed(text1)
        let embedding1Copy = try await embedder.embed(text1)
        
        let identicalSimilarity = embedding1.cosineSimilarity(to: embedding1Copy)
        #expect(identicalSimilarity > 0.99) // Should be very close to 1.0
        
        // Test different texts
        let text2 = "Deep learning models are powerful"
        let text3 = "The weather is nice today"
        
        let embedding2 = try await embedder.embed(text2)
        let embedding3 = try await embedder.embed(text3)
        
        let relatedSimilarity = embedding1.cosineSimilarity(to: embedding2)
        let unrelatedSimilarity = embedding1.cosineSimilarity(to: embedding3)
        
        // Related texts should have higher similarity than unrelated
        #expect(relatedSimilarity > unrelatedSimilarity)
        #expect(relatedSimilarity > 0.5) // Somewhat similar
        #expect(unrelatedSimilarity < 0.5) // Not very similar
        
        logger.success("Cosine similarity calculations verified")
    }
    
    @Test("Euclidean distance calculation")
    func testEuclideanDistance() async throws {
        let embedder = MockTextEmbedder(dimensions: 384)
        try await embedder.loadModel()
        
        let embedding1 = try await embedder.embed("Test text one")
        let embedding2 = try await embedder.embed("Test text two")
        let embedding3 = try await embedder.embed("Completely different content")
        
        let distance12 = embedding1.euclideanDistance(to: embedding2)
        let distance13 = embedding1.euclideanDistance(to: embedding3)
        
        // More different texts should have larger distance
        #expect(distance13 > distance12)
        
        // Distance to self should be zero
        let selfDistance = embedding1.euclideanDistance(to: embedding1)
        #expect(selfDistance < 0.001)
        
        logger.success("Euclidean distance calculations verified")
    }
    
    @Test("Dot product calculation")
    func testDotProduct() async throws {
        let embedder = MockTextEmbedder(dimensions: 256)
        try await embedder.loadModel()
        
        let embedding1 = try await embedder.embed("Vector one")
        let embedding2 = try await embedder.embed("Vector two")
        
        let dotProduct = embedding1.dotProduct(with: embedding2)
        
        // For normalized vectors, dot product should be between -1 and 1
        #expect(dotProduct >= -1.0 && dotProduct <= 1.0)
        
        // Self dot product should equal magnitude squared (≈1 for normalized)
        let selfDot = embedding1.dotProduct(with: embedding1)
        #expect(abs(selfDot - 1.0) < 0.01)
        
        logger.success("Dot product calculations verified")
    }
    
    // MARK: - Performance Tests
    
    @Test("Embedding generation performance")
    func testEmbeddingPerformance() async throws {
        logger.start("Performance benchmark test")
        
        let embedder = MockTextEmbedder(dimensions: 768)
        try await embedder.loadModel()
        
        let testTexts = (1...100).map { "Test text number \($0) with some content" }
        
        // Single embedding performance
        let singleStart = CFAbsoluteTimeGetCurrent()
        for text in testTexts {
            _ = try await embedder.embed(text)
        }
        let singleDuration = CFAbsoluteTimeGetCurrent() - singleStart
        let singleThroughput = Double(testTexts.count) / singleDuration
        
        // Batch embedding performance
        let batchStart = CFAbsoluteTimeGetCurrent()
        _ = try await embedder.embed(batch: testTexts)
        let batchDuration = CFAbsoluteTimeGetCurrent() - batchStart
        let batchThroughput = Double(testTexts.count) / batchDuration
        
        let speedup = batchThroughput / singleThroughput
        
        logger.performance("Single embeddings", duration: singleDuration, throughput: singleThroughput)
        logger.performance("Batch embeddings", duration: batchDuration, throughput: batchThroughput)
        logger.info("Batch speedup: \(String(format: "%.2fx", speedup))")
        
        // Batch should be faster
        #expect(speedup > 1.5)
    }
    
    // MARK: - Metal Acceleration Tests
    
    @Test("Metal acceleration availability")
    func testMetalAvailability() async throws {
        logger.start("Metal acceleration test")
        
        if let metal = MetalAccelerator.shared {
            logger.success("Metal acceleration available")
            
            // Test normalization
            let vectors = [[Float](repeating: 1.0, count: 768)]
            let normalized = try await metal.normalizeVectors(vectors)
            
            #expect(normalized.count == 1)
            #expect(normalized[0].count == 768)
            
            // Check normalization worked
            let magnitude = sqrt(normalized[0].reduce(0) { $0 + $1 * $1 })
            #expect(abs(magnitude - 1.0) < 0.01)
            
            logger.success("Metal normalization verified")
        } else {
            logger.warning("Metal acceleration not available on this device")
        }
    }
    
    @Test("Metal cosine similarity")
    func testMetalCosineSimilarity() async throws {
        guard let metal = MetalAccelerator.shared else {
            logger.info("Skipping Metal test - not available")
            return
        }
        
        let queries = [
            [Float](repeating: 0.5, count: 384),
            [Float](repeating: -0.5, count: 384)
        ]
        
        let keys = [
            [Float](repeating: 0.5, count: 384),
            [Float](repeating: 0.7, count: 384)
        ]
        
        let similarities = try await metal.cosineSimilarityMatrix(
            queries: queries,
            keys: keys
        )
        
        #expect(similarities.count == queries.count)
        #expect(similarities[0].count == keys.count)
        
        logger.success("Metal cosine similarity matrix computed")
    }
    
    // MARK: - Cache Tests
    
    @Test("LRU cache functionality")
    func testLRUCache() async throws {
        logger.start("LRU cache test")
        
        let cache = EmbeddingCache(maxEntries: 3)
        let embedder = MockTextEmbedder(dimensions: 256)
        try await embedder.loadModel()
        
        // Generate and cache embeddings
        let texts = ["one", "two", "three", "four"]
        var embeddings: [EmbeddingVector] = []
        
        for text in texts {
            let embedding = try await embedder.embed(text)
            embeddings.append(embedding)
            await cache.set(text: text, modelIdentifier: "test-model", embedding: embedding)
        }
        
        // First three should be evicted when fourth is added
        let cachedOne = await cache.get(text: "one", modelIdentifier: "test-model")
        #expect(cachedOne == nil) // Should be evicted
        
        let cachedFour = await cache.get(text: "four", modelIdentifier: "test-model")
        #expect(cachedFour != nil) // Should be present
        
        let stats = await cache.statistics()
        #expect(stats.currentSize == 3)
        
        logger.success("LRU cache eviction verified")
    }
    
    // MARK: - Error Handling Tests
    
    @Test("Model not loaded error")
    func testModelNotLoadedError() async throws {
        let embedder = MockTextEmbedder(dimensions: 768)
        
        // Try to embed without loading model
        do {
            _ = try await embedder.embed("test")
            Issue.record("Expected error but none was thrown")
        } catch {
            #expect(error is EmbeddingError)
            logger.success("Model not loaded error caught correctly")
        }
    }
    
    @Test("Invalid dimensions error")
    func testInvalidDimensionsError() async throws {
        // Test with invalid dimensions
        let embedder = MockTextEmbedder(dimensions: 0)
        
        do {
            try await embedder.loadModel()
            Issue.record("Expected error for invalid dimensions")
        } catch {
            logger.success("Invalid dimensions error caught")
        }
    }
}

// MARK: - Test Utilities

extension EmbeddingVector {
    /// Calculate magnitude of the vector
    func magnitude() -> Float {
        return sqrt(vector.reduce(0) { $0 + $1 * $1 })
    }
}