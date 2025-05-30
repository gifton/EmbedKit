import Testing
import Foundation
@testable import EmbedKit
import PipelineKit

@Suite("Pipeline Integration Tests")
struct PipelineIntegrationTests {
    
    @Test("Basic embedding through pipeline")
    func testBasicEmbedding() async throws {
        // Create pipeline
        let modelManager = EmbeddingModelManager()
        let embedder = MockTextEmbedder()
        let pipeline = try await EmbeddingPipeline(
            embedder: embedder,
            modelManager: modelManager,
            configuration: .init(enableCache: false)
        )
        
        // Test embedding
        let result = try await pipeline.embed("Hello, world!")
        
        #expect(result.embedding.dimensions == 384)
        #expect(result.modelIdentifier == "mock-model")
        #expect(!result.fromCache)
    }
    
    @Test("Batch embedding with cache")
    func testBatchEmbeddingWithCache() async throws {
        // Create pipeline with cache enabled
        let modelManager = EmbeddingModelManager()
        let embedder = MockTextEmbedder()
        let pipeline = try await EmbeddingPipeline(
            embedder: embedder,
            modelManager: modelManager,
            configuration: .init(enableCache: true)
        )
        
        let texts = ["Text 1", "Text 2", "Text 3"]
        
        // First batch - all cache misses
        let result1 = try await pipeline.embedBatch(texts)
        #expect(result1.embeddings.count == 3)
        #expect(result1.cacheHitRate == 0.0)
        
        // Second batch - all cache hits
        let result2 = try await pipeline.embedBatch(texts)
        #expect(result2.embeddings.count == 3)
        #expect(result2.cacheHitRate == 1.0)
    }
    
    @Test("Command validation")
    func testCommandValidation() async throws {
        // Test empty text validation
        let emptyCommand = EmbedTextCommand(text: "")
        #expect(throws: ValidationError.self) {
            try emptyCommand.validate()
        }
        
        // Test long text validation
        let longText = String(repeating: "a", count: 15_000)
        let longCommand = EmbedTextCommand(text: longText)
        #expect(throws: ValidationError.self) {
            try longCommand.validate()
        }
        
        // Test valid command
        let validCommand = EmbedTextCommand(text: "Valid text")
        #expect(throws: Never.self) {
            try validCommand.validate()
        }
    }
    
    @Test("Streaming embeddings")
    func testStreamingEmbeddings() async throws {
        let modelManager = EmbeddingModelManager()
        let embedder = MockTextEmbedder()
        let pipeline = try await EmbeddingPipeline(
            embedder: embedder,
            modelManager: modelManager
        )
        
        let texts = (1...10).map { "Text \($0)" }
        let source = ArrayTextSource(texts)
        
        var results: [StreamingEmbeddingResult] = []
        let stream = try await pipeline.streamEmbeddings(
            from: source,
            maxConcurrency: 5
        )
        
        for try await result in stream {
            results.append(result)
        }
        
        #expect(results.count == 10)
        #expect(results.allSatisfy { $0.embedding.dimensions == 384 })
    }
    
    @Test("Pipeline factories")
    func testPipelineFactories() async throws {
        let modelManager = EmbeddingModelManager()
        let embedder = MockTextEmbedder()
        
        // Test different factory configurations
        let highPerf = try await EmbeddingPipelineFactory.highPerformance(
            embedder: embedder,
            modelManager: modelManager
        )
        
        let balanced = try await EmbeddingPipelineFactory.balanced(
            embedder: embedder,
            modelManager: modelManager
        )
        
        let dev = try await EmbeddingPipelineFactory.development(
            embedder: embedder,
            modelManager: modelManager
        )
        
        let minimal = try await EmbeddingPipelineFactory.minimal(
            embedder: embedder,
            modelManager: modelManager
        )
        
        // All should be able to embed text
        for pipeline in [highPerf, balanced, dev, minimal] {
            let result = try await pipeline.embed("Test")
            #expect(result.embedding.dimensions > 0)
        }
    }
    
    @Test("Telemetry integration")
    func testTelemetryIntegration() async throws {
        let modelManager = EmbeddingModelManager()
        let embedder = MockTextEmbedder()
        let pipeline = try await EmbeddingPipelineFactory.development(
            embedder: embedder,
            modelManager: modelManager
        )
        
        // Reset telemetry
        await pipeline.resetTelemetry()
        
        // Perform operations
        _ = try await pipeline.embed("Test 1")
        _ = try await pipeline.embed("Test 2")
        _ = try await pipeline.embedBatch(["Batch 1", "Batch 2"])
        
        // Check statistics
        let stats = await pipeline.getStatistics()
        #expect(stats.isReady)
        #expect(stats.currentModel == "mock-model")
        
        // Check telemetry data exists
        let telemetryData = await pipeline.getTelemetryData()
        #expect(telemetryData != nil)
    }
    
    @Test("Error handling")
    func testErrorHandling() async throws {
        let modelManager = EmbeddingModelManager()
        let embedder = FailingTextEmbedder()
        let pipeline = try await EmbeddingPipeline(
            embedder: embedder,
            modelManager: modelManager
        )
        
        // Should throw embedding error
        #expect(throws: EmbeddingError.self) {
            _ = try await pipeline.embed("Will fail")
        }
    }
    
    @Test("Cache management")
    func testCacheManagement() async throws {
        let modelManager = EmbeddingModelManager()
        let embedder = MockTextEmbedder()
        let pipeline = try await EmbeddingPipeline(
            embedder: embedder,
            modelManager: modelManager
        )
        
        // Preload cache
        let texts = ["Preload 1", "Preload 2", "Preload 3"]
        let preloadResult = try await pipeline.preloadCache(texts: texts)
        #expect(preloadResult.textsProcessed == 3)
        
        // Check cache stats
        let stats1 = await pipeline.getCacheStatistics()
        #expect(stats1.currentSize >= 3)
        
        // Clear cache
        let clearResult = try await pipeline.clearCache()
        #expect(clearResult.entriesCleared >= 3)
        
        // Verify cache is empty
        let stats2 = await pipeline.getCacheStatistics()
        #expect(stats2.currentSize == 0)
    }
    
    @Test("Integration validation")
    func testIntegrationValidation() async throws {
        // This should validate the entire setup
        try await PipelineIntegration.validateSetup()
    }
    
    @Test("Quick start helper")
    func testQuickStart() async throws {
        let (pipeline, cleanup) = try await PipelineIntegration.quickStart()
        
        defer {
            Task {
                await cleanup()
            }
        }
        
        // Should be able to embed
        let result = try await pipeline.embed("Quick start test")
        #expect(result.embedding.dimensions > 0)
    }
}

// MARK: - Mock Implementations

actor MockTextEmbedder: TextEmbedder {
    let configuration = EmbeddingConfiguration()
    let dimensions = 384
    let modelIdentifier = "mock-model"
    var isReady = true
    
    func embed(_ text: String) async throws -> EmbeddingVector {
        // Simulate some processing time
        try await Task.sleep(nanoseconds: 10_000_000) // 10ms
        
        // Return mock embedding
        let values = (0..<dimensions).map { _ in Float.random(in: -1...1) }
        return EmbeddingVector(values)
    }
    
    func loadModel() async throws {
        isReady = true
    }
    
    func unloadModel() async throws {
        isReady = false
    }
}

actor FailingTextEmbedder: TextEmbedder {
    let configuration = EmbeddingConfiguration()
    let dimensions = 384
    let modelIdentifier = "failing-model"
    let isReady = true
    
    func embed(_ text: String) async throws -> EmbeddingVector {
        throw EmbeddingError.inferenceFailed("Mock failure")
    }
    
    func loadModel() async throws {
        // No-op
    }
    
    func unloadModel() async throws {
        // No-op
    }
}