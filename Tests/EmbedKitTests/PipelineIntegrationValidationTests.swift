import Testing
import Foundation
import PipelineKit
@testable import EmbedKit

/// Validation tests for PipelineKit integration
@Suite("PipelineKit Integration Validation")
struct PipelineIntegrationValidationTests {
    let logger = EmbedKitLogger.custom("PipelineValidation")
    
    // MARK: - Basic Pipeline Tests
    
    @Test("Pipeline executes single embedding command")
    func testSingleEmbeddingPipeline() async throws {
        logger.start("Single embedding pipeline test")
        
        // Create minimal pipeline
        let pipeline = try EmbeddingPipeline.createMinimalPipeline()
        
        // Execute embedding command
        let command = EmbedTextCommand(text: "Hello, PipelineKit!")
        let result = try await pipeline.execute(command)
        
        // Verify result
        #expect(result.embedding.dimensions == 768)
        #expect(result.text == "Hello, PipelineKit!")
        #expect(result.cached == false)
        
        logger.success("Single embedding through pipeline successful")
    }
    
    @Test("Pipeline executes batch embedding command")
    func testBatchEmbeddingPipeline() async throws {
        logger.start("Batch embedding pipeline test")
        
        let pipeline = try EmbeddingPipeline.createBalancedPipeline()
        
        let texts = [
            "First document",
            "Second document with more content",
            "Third document"
        ]
        
        let command = BatchEmbedCommand(texts: texts)
        let result = try await pipeline.execute(command)
        
        #expect(result.embeddings.count == texts.count)
        #expect(result.processedCount == texts.count)
        #expect(result.failedCount == 0)
        
        // Verify each embedding
        for (index, embedding) in result.embeddings.enumerated() {
            #expect(embedding.dimensions == 768)
            logger.info("Embedding \(index + 1): \(embedding.dimensions) dimensions")
        }
        
        logger.success("Batch embedding through pipeline successful")
    }
    
    // MARK: - Middleware Tests
    
    @Test("Cache middleware functions correctly")
    func testCacheMiddleware() async throws {
        logger.start("Cache middleware test")
        
        // Create pipeline with cache
        let config = EmbeddingPipelineConfiguration(
            enableCache: true,
            enableTelemetry: false,
            enableValidation: true
        )
        let pipeline = try EmbeddingPipeline(configuration: config)
        
        let text = "Cacheable text"
        let command = EmbedTextCommand(text: text)
        
        // First execution - should miss cache
        let result1 = try await pipeline.execute(command)
        #expect(result1.cached == false)
        logger.info("First execution: cache miss")
        
        // Second execution - should hit cache
        let result2 = try await pipeline.execute(command)
        #expect(result2.cached == true)
        #expect(result2.embedding.dimensions == result1.embedding.dimensions)
        logger.info("Second execution: cache hit")
        
        // Verify embeddings are identical
        let similarity = result1.embedding.cosineSimilarity(to: result2.embedding)
        #expect(similarity > 0.9999) // Should be virtually identical
        
        logger.success("Cache middleware verified")
    }
    
    @Test("Validation middleware rejects invalid input")
    func testValidationMiddleware() async throws {
        logger.start("Validation middleware test")
        
        let pipeline = try EmbeddingPipeline.createBalancedPipeline()
        
        // Test empty text
        do {
            let command = EmbedTextCommand(text: "")
            _ = try await pipeline.execute(command)
            Issue.record("Expected validation error for empty text")
        } catch {
            logger.success("Empty text rejected correctly")
        }
        
        // Test text that's too long
        let longText = String(repeating: "word ", count: 10000)
        let longCommand = EmbedTextCommand(text: longText)
        
        // Should succeed but truncate
        let result = try await pipeline.execute(longCommand)
        #expect(result.embedding.dimensions == 768)
        logger.info("Long text handled with truncation")
    }
    
    @Test("Metal acceleration middleware")
    func testMetalAccelerationMiddleware() async throws {
        logger.start("Metal acceleration middleware test")
        
        let config = EmbeddingPipelineConfiguration(
            enableMetalAcceleration: true,
            enableCache: false
        )
        let pipeline = try EmbeddingPipeline(configuration: config)
        
        // Process batch to test Metal acceleration
        let texts = (1...10).map { "Test text \($0)" }
        let command = BatchEmbedCommand(texts: texts)
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let result = try await pipeline.execute(command)
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        
        #expect(result.embeddings.count == texts.count)
        
        logger.performance("Batch with Metal", duration: duration, throughput: Double(texts.count) / duration)
        logger.success("Metal acceleration middleware verified")
    }
    
    // MARK: - Model Management Tests
    
    @Test("Model loading through pipeline")
    func testModelLoadingPipeline() async throws {
        logger.start("Model loading pipeline test")
        
        let pipeline = try EmbeddingPipeline.createBalancedPipeline()
        
        // Load model
        let loadCommand = LoadModelCommand(
            modelId: "test-model",
            version: "1.0"
        )
        let loadResult = try await pipeline.execute(loadCommand)
        
        #expect(loadResult.success == true)
        #expect(loadResult.metadata.name == "test-model")
        #expect(loadResult.metadata.version == "1.0")
        
        logger.success("Model loaded through pipeline")
    }
    
    @Test("Model swapping through pipeline")
    func testModelSwappingPipeline() async throws {
        logger.start("Model swapping test")
        
        let pipeline = try EmbeddingPipeline.createBalancedPipeline()
        
        // Load first model
        let load1 = LoadModelCommand(modelId: "model-v1", version: "1.0")
        _ = try await pipeline.execute(load1)
        
        // Generate embedding with first model
        let embed1 = EmbedTextCommand(text: "Test with model 1")
        let result1 = try await pipeline.execute(embed1)
        
        // Swap to second model
        let swap = SwapModelCommand(
            modelId: "model-v2",
            version: "2.0",
            gracePeriod: 0.1
        )
        let swapResult = try await pipeline.execute(swap)
        #expect(swapResult.success == true)
        
        // Generate embedding with second model
        let embed2 = EmbedTextCommand(text: "Test with model 2")
        let result2 = try await pipeline.execute(embed2)
        
        // Embeddings should be different (different models)
        let similarity = result1.embedding.cosineSimilarity(to: result2.embedding)
        #expect(similarity < 0.95) // Should be somewhat different
        
        logger.success("Model swapping verified")
    }
    
    // MARK: - Streaming Tests
    
    @Test("Streaming embeddings through pipeline")
    func testStreamingPipeline() async throws {
        logger.start("Streaming pipeline test")
        
        let pipeline = try EmbeddingPipeline.createHighPerformancePipeline()
        
        // Create stream of texts
        let texts = (1...20).map { "Streaming text number \($0)" }
        let stream = AsyncStream { continuation in
            Task {
                for text in texts {
                    continuation.yield(text)
                    try await Task.sleep(nanoseconds: 10_000_000) // 10ms
                }
                continuation.finish()
            }
        }
        
        let command = StreamEmbedCommand(textStream: stream)
        let result = try await pipeline.execute(command)
        
        // Collect results
        var embeddings: [EmbeddingVector] = []
        for try await embedding in result.embeddings {
            embeddings.append(embedding)
            logger.processing("Received embedding", progress: Double(embeddings.count) / Double(texts.count))
        }
        
        #expect(embeddings.count == texts.count)
        logger.success("Streaming pipeline processed \(embeddings.count) embeddings")
    }
    
    // MARK: - Error Handling Tests
    
    @Test("Pipeline handles errors gracefully")
    func testPipelineErrorHandling() async throws {
        logger.start("Error handling test")
        
        let pipeline = try EmbeddingPipeline.createBalancedPipeline()
        
        // Test with invalid model command
        let invalidLoad = LoadModelCommand(
            modelId: "",  // Invalid
            version: "1.0"
        )
        
        do {
            _ = try await pipeline.execute(invalidLoad)
            Issue.record("Expected error for invalid model ID")
        } catch {
            logger.success("Invalid model ID error caught")
        }
        
        // Test recovery after error
        let validCommand = EmbedTextCommand(text: "Valid text after error")
        let result = try await pipeline.execute(validCommand)
        #expect(result.embedding.dimensions == 768)
        
        logger.success("Pipeline recovered after error")
    }
    
    // MARK: - Performance Comparison Tests
    
    @Test("Pipeline performance comparison")
    func testPipelinePerformanceComparison() async throws {
        logger.start("Performance comparison test")
        
        let texts = (1...50).map { "Performance test text \($0)" }
        
        // Test different pipeline configurations
        let configs = [
            ("Minimal", try EmbeddingPipeline.createMinimalPipeline()),
            ("Balanced", try EmbeddingPipeline.createBalancedPipeline()),
            ("High Performance", try EmbeddingPipeline.createHighPerformancePipeline())
        ]
        
        for (name, pipeline) in configs {
            let command = BatchEmbedCommand(texts: texts)
            
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = try await pipeline.execute(command)
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            
            let throughput = Double(texts.count) / duration
            logger.performance("\(name) pipeline", duration: duration, throughput: throughput)
            
            #expect(result.embeddings.count == texts.count)
        }
        
        logger.success("Performance comparison completed")
    }
    
    // MARK: - Integration Tests
    
    @Test("Full integration test")
    func testFullIntegration() async throws {
        logger.start("Full integration test")
        
        // Create a development pipeline with all features
        let pipeline = try EmbeddingPipeline.createDevelopmentPipeline()
        
        // 1. Load model
        let loadResult = try await pipeline.execute(
            LoadModelCommand(modelId: "integration-test", version: "1.0")
        )
        #expect(loadResult.success)
        logger.success("✓ Model loaded")
        
        // 2. Single embedding
        let singleResult = try await pipeline.execute(
            EmbedTextCommand(text: "Single embedding test")
        )
        #expect(singleResult.embedding.dimensions == 768)
        logger.success("✓ Single embedding generated")
        
        // 3. Batch embedding
        let batchResult = try await pipeline.execute(
            BatchEmbedCommand(texts: ["Batch 1", "Batch 2", "Batch 3"])
        )
        #expect(batchResult.embeddings.count == 3)
        logger.success("✓ Batch embeddings generated")
        
        // 4. Cache test
        let cachedResult = try await pipeline.execute(
            EmbedTextCommand(text: "Single embedding test") // Same as #2
        )
        #expect(cachedResult.cached == true)
        logger.success("✓ Cache hit verified")
        
        // 5. Clear cache
        _ = try await pipeline.execute(ClearCacheCommand())
        logger.success("✓ Cache cleared")
        
        // 6. Verify cache was cleared
        let afterClearResult = try await pipeline.execute(
            EmbedTextCommand(text: "Single embedding test")
        )
        #expect(afterClearResult.cached == false)
        logger.success("✓ Cache clear verified")
        
        logger.complete("Full integration test", result: "All components working correctly")
    }
}

// MARK: - Test Helpers

extension EmbeddingPipeline {
    /// Create a test pipeline with minimal configuration
    static func createTestPipeline() throws -> EmbeddingPipeline {
        let config = EmbeddingPipelineConfiguration(
            enableCache: true,
            enableMetalAcceleration: false,
            enableTelemetry: false,
            enableValidation: true,
            enableRateLimiting: false,
            enableMonitoring: false
        )
        return try EmbeddingPipeline(configuration: config)
    }
}