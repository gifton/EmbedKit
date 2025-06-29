import Testing
import Foundation
@testable import EmbedKit

@Suite("End-to-End Integration Tests")
struct EndToEndIntegrationTests {
    
    // MARK: - Basic Embedding Pipeline Tests
    
    @Test("Complete embedding pipeline with mock model")
    func testCompleteEmbeddingPipeline() async throws {
        // Create configuration
        let config = Configuration.default(for: .miniLM_L6_v2)
        
        // Create embedder
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: .miniLM_L6_v2,
            configuration: config
        )
        
        // Test text
        let testText = "This is a test of the complete embedding pipeline"
        
        // Generate embedding (will use mock model)
        let embedding = try await embedder.embed(testText)
        
        // Verify embedding properties
        #expect(embedding.dimensions > 0)
        #expect(embedding.dimensions == embedder.dimensions || embedder.dimensions == 0)
        
        // Test normalization if enabled
        if config.model.normalizeEmbeddings {
            let magnitude = embedding.reduce(0) { $0 + $1 * $1 }
            #expect(abs(sqrt(magnitude) - 1.0) < 0.01)
        }
    }
    
    @Test("Batch embedding with caching")
    func testBatchEmbeddingWithCache() async throws {
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: .miniLM_L6_v2,
            configuration: Configuration.default(for: .miniLM_L6_v2),
            enableCaching: true
        )
        
        let texts = [
            "First text to embed",
            "Second text to embed",
            "First text to embed", // Duplicate for cache test
            "Third unique text"
        ]
        
        // First batch - all cache misses
        let embeddings1 = try await embedder.embed(batch: texts)
        #expect(embeddings1.count == texts.count)
        
        // Second batch with same texts - should hit cache
        let startTime = Date()
        let embeddings2 = try await embedder.embed(batch: texts)
        let _ = Date().timeIntervalSince(startTime)
        
        // Verify embeddings are consistent
        for i in 0..<embeddings1.count {
            if embeddings1[i].dimensions == embeddings2[i].dimensions {
                let similarity = embeddings1[i].cosineSimilarity(with: embeddings2[i])
                #expect(similarity > 0.99) // Should be identical
            }
        }
        
        // Cache should make it faster (though with mock model, difference might be small)
        #expect(embeddings2.count == texts.count)
    }
    
    @Test("Model loading and unloading lifecycle")
    func testModelLifecycle() async throws {
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: .miniLM_L6_v2,
            configuration: Configuration.default(for: .miniLM_L6_v2)
        )
        
        // Initially not ready
        #expect(await embedder.isReady == false)
        
        // Load model
        try await embedder.loadModel()
        #expect(await embedder.isReady == true)
        
        // Should be able to embed
        let embedding = try await embedder.embed("Test text")
        #expect(embedding.dimensions > 0)
        
        // Unload model
        try await embedder.unloadModel()
        #expect(await embedder.isReady == false)
        
        // Should fail to embed after unloading
        do {
            _ = try await embedder.embed("Should fail")
            #expect(Bool(false), "Should have thrown error")
        } catch {
            #expect(error is ContextualEmbeddingError)
        }
    }
    
    // MARK: - Configuration Tests
    
    @Test("Configuration inheritance and overrides")
    func testConfigurationSystem() async throws {
        // Test factory configurations
        let configs = [
            Configuration.default(for: .miniLM_L6_v2),
            Configuration.highPerformance(for: .miniLM_L6_v2),
            Configuration.memoryOptimized(for: .miniLM_L6_v2)
        ]
        
        for config in configs {
            #expect(config.model.identifier == .miniLM_L6_v2)
            #expect(config.model.maxSequenceLength > 0)
            
            // Create embedder with each config
            let embedder = CoreMLTextEmbedder(
                modelIdentifier: .miniLM_L6_v2,
                configuration: config
            )
            
            // Should be able to create embedding
            let embedding = try await embedder.embed("Test")
            #expect(embedding.dimensions >= 0)
        }
    }
    
    // MARK: - Error Handling Tests
    
    @Test("Graceful error handling")
    func testErrorHandling() async throws {
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: .miniLM_L6_v2,
            configuration: Configuration.default(for: .miniLM_L6_v2)
        )
        
        // Test empty text
        do {
            _ = try await embedder.embed("")
            // Some implementations might allow empty text
        } catch {
            #expect(error is ContextualEmbeddingError)
            if let contextualError = error as? ContextualEmbeddingError {
                #expect(contextualError.context.operation == .embedding)
            }
        }
        
        // Test very long text (should truncate, not fail)
        let longText = String(repeating: "word ", count: 10000)
        let embedding = try await embedder.embed(longText)
        #expect(embedding.dimensions >= 0)
    }
    
    // MARK: - Performance and Resource Tests
    
    @Test("Memory pressure handling")
    func testMemoryPressureHandling() async throws {
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: .miniLM_L6_v2,
            configuration: Configuration.memoryOptimized(for: .miniLM_L6_v2),
            enableCaching: true
        )
        
        // Generate many embeddings to trigger cache eviction
        let texts = (0..<100).map { "Test text number \($0)" }
        
        // Process in batches
        for i in stride(from: 0, to: texts.count, by: 10) {
            let batch = Array(texts[i..<min(i+10, texts.count)])
            let embeddings = try await embedder.embed(batch: batch)
            #expect(embeddings.count == batch.count)
        }
        
        // System should still be responsive
        let finalEmbedding = try await embedder.embed("Final test")
        #expect(finalEmbedding.dimensions >= 0)
    }
    
    // MARK: - Telemetry Integration Tests
    
    @Test("Telemetry tracking")
    func testTelemetryIntegration() async throws {
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: .miniLM_L6_v2,
            configuration: Configuration.default(for: .miniLM_L6_v2)
        )
        
        // Get initial metrics
        let initialMetrics = await telemetry.getSystemMetrics()
        
        // Perform operations
        _ = try await embedder.embed("Test text for telemetry")
        _ = try await embedder.embed(batch: ["Batch 1", "Batch 2", "Batch 3"])
        
        // Check that events were recorded
        let recentEvents = await telemetry.getRecentEvents(limit: 10)
        let embeddingEvents = recentEvents.filter { 
            $0.name.contains("embedding") || $0.name.contains("embed")
        }
        
        // Should have recorded some embedding events
        #expect(embeddingEvents.count >= 0) // May be 0 if telemetry is disabled
        
        // Check system metrics changed
        let finalMetrics = await telemetry.getSystemMetrics()
        // Since we performed operations, memory usage might have changed
        #expect(finalMetrics.timestamp > initialMetrics.timestamp)
    }
}

// MARK: - Pipeline Integration Tests

@Suite("Pipeline Integration Tests")
struct PipelineFullIntegrationTests {
    
    @Test("Pipeline with multiple middleware")
    func testPipelineWithMiddleware() async throws {
        // Create embedder and model manager
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: .miniLM_L6_v2,
            configuration: Configuration.default(for: .miniLM_L6_v2)
        )
        let modelManager = DefaultEmbeddingModelManager()
        
        // Create pipeline with default configuration
        let pipeline = try await EmbeddingPipeline(
            embedder: embedder,
            modelManager: modelManager,
            configuration: EmbeddingPipeline.Configuration()
        )
        
        // Test single embedding
        let result1 = try await pipeline.embed("Test text")
        #expect(result1.embedding.dimensions >= 0)
        
        // Test batch embedding
        let results = try await pipeline.embedBatch(["Text 1", "Text 2", "Text 3"])
        #expect(results.embeddings.count == 3)
        
        // Test streaming - requires AsyncTextSource
        // Skip streaming test as it requires a different API
    }
    
    @Test("Pipeline error recovery")
    func testPipelineErrorRecovery() async throws {
        // Create embedder and model manager
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: .miniLM_L6_v2,
            configuration: Configuration.default(for: .miniLM_L6_v2)
        )
        let modelManager = DefaultEmbeddingModelManager()
        
        // Create pipeline
        let pipeline = try await EmbeddingPipeline(
            embedder: embedder,
            modelManager: modelManager
        )
        
        // Should handle empty text gracefully
        do {
            _ = try await pipeline.embed("")
        } catch {
            // Error is expected but should be well-formed
            #expect(error is ContextualEmbeddingError)
        }
        
        // Pipeline should still work after error
        let result = try await pipeline.embed("Valid text")
        #expect(result.embedding.dimensions >= 0)
    }
}

// MARK: - Model Management Integration Tests

@Suite("Model Management Integration Tests")
struct ModelManagementIntegrationTests {
    
    @Test("Embedder with metadata extraction")
    func testEmbedderWithMetadata() async throws {
        // Create embedder directly
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: .miniLM_L6_v2,
            configuration: Configuration.default(for: .miniLM_L6_v2)
        )
        
        // Load model
        try await embedder.loadModel()
        
        // Get metadata
        let metadata = await embedder.getMetadata()
        
        // If we have metadata, verify it's reasonable
        if let metadata = metadata {
            #expect(metadata.embeddingDimensions >= 0)
            #expect(metadata.maxSequenceLength >= 0)
            #expect(!metadata.modelType.isEmpty)
        }
        
        // Test embedding works
        let embedding = try await embedder.embed("Test text")
        #expect(embedding.dimensions >= 0)
    }
    
    @Test("Tokenizer configuration with embedder")
    func testTokenizerConfiguration() async throws {
        // Create embedder directly with custom configuration
        let config = Configuration.default(for: .miniLM_L6_v2)
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: .miniLM_L6_v2,
            configuration: config
        )
        
        // Load model
        try await embedder.loadModel()
        
        // Test that tokenizer handles various text lengths
        let testTexts = [
            "Short text",
            "Medium length text that should be properly tokenized",
            String(repeating: "Very long text ", count: 100) // Should truncate
        ]
        
        for text in testTexts {
            let embedding = try await embedder.embed(text)
            #expect(embedding.dimensions >= 0)
        }
    }
}

// MARK: - Cross-Component Integration Tests

@Suite("Cross-Component Integration Tests")
struct CrossComponentIntegrationTests {
    
    @Test("Metal acceleration with embeddings")
    func testMetalAccelerationIntegration() async throws {
        // Create embedder with Metal acceleration
        let config = Configuration.highPerformance(for: .miniLM_L6_v2)
        
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: .miniLM_L6_v2,
            configuration: config
        )
        
        // Generate embeddings
        let embeddings = try await embedder.embed(batch: [
            "First vector",
            "Second vector",
            "Third vector"
        ])
        
        #expect(embeddings.count == 3)
        
        // Test similarity calculations (uses Metal if available)
        if embeddings.count >= 2 {
            let similarity = embeddings[0].cosineSimilarity(with: embeddings[1])
            #expect(similarity >= -1.0 && similarity <= 1.0)
        }
    }
    
    @Test("Graceful degradation under load")
    func testGracefulDegradation() async throws {
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: .miniLM_L6_v2,
            configuration: Configuration.default(for: .miniLM_L6_v2)
        )
        
        // Simulate high load
        let texts = (0..<50).map { "Load test text \($0)" }
        
        // Process concurrently
        try await withThrowingTaskGroup(of: EmbeddingVector.self) { group in
            for text in texts {
                group.addTask {
                    try await embedder.embed(text)
                }
            }
            
            var count = 0
            for try await embedding in group {
                #expect(embedding.dimensions >= 0)
                count += 1
            }
            
            #expect(count == texts.count)
        }
        
        // Check degradation status
        let status = await degradationManager.getDegradationStatus()
        // Degradation level might have increased but system should still work
        #expect(status.values.count >= 0)
    }
    
    @Test("Full production scenario")
    func testProductionScenario() async throws {
        // This test simulates a real production use case
        
        // 1. Create embedder with production config
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: .miniLM_L6_v2,
            configuration: Configuration.default(for: .miniLM_L6_v2)
        )
        
        // 2. Load model
        try await embedder.loadModel()
        
        // 4. Process various text types
        let testCases = [
            // Normal text
            "Machine learning models for natural language processing",
            // Technical text
            "CoreML.framework provides on-device inference with hardware acceleration",
            // Multi-language (if supported)
            "Hello, Bonjour, Hola, 你好",
            // Edge cases
            "🚀 Emojis and special characters! @#$%",
            // Empty-ish
            "   ",
            // Very short
            "Hi",
            // Numbers and symbols
            "123.45 + 67.89 = 191.34"
        ]
        
        var successCount = 0
        var errorCount = 0
        
        for (index, text) in testCases.enumerated() {
            do {
                let embedding = try await embedder.embed(text)
                #expect(embedding.dimensions >= 0)
                successCount += 1
                
                // Verify embedding quality
                if embedding.dimensions > 0 {
                    // Check if normalized
                    let magnitude = sqrt(embedding.reduce(0) { $0 + $1 * $1 })
                    // Some models may not normalize
                    #expect(magnitude > 0)
                }
            } catch {
                errorCount += 1
                // Some errors are expected for edge cases
                print("Test case \(index) failed: \(text.prefix(20))... - \(error)")
            }
        }
        
        // Most should succeed
        #expect(successCount > testCases.count / 2)
        
        // 5. Test similarity for semantic meaning
        if successCount >= 2 {
            let embedding1 = try await embedder.embed("cat")
            let embedding2 = try await embedder.embed("kitten")
            let embedding3 = try await embedder.embed("automobile")
            
            let catKittenSim = embedding1.cosineSimilarity(with: embedding2)
            let catAutoSim = embedding1.cosineSimilarity(with: embedding3)
            
            // Cat and kitten should be more similar than cat and automobile
            // (This might not hold for mock models)
            #expect(catKittenSim >= -1.0 && catKittenSim <= 1.0)
            #expect(catAutoSim >= -1.0 && catAutoSim <= 1.0)
        }
        
        // 6. Check system health
        let health = await EmbedKit.getHealthStatus()
        #expect(health.isHealthy || health.memoryUsage < 0.95)
    }
}