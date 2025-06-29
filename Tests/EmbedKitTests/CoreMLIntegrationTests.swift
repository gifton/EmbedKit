import Testing
import Foundation
@preconcurrency import CoreML
@testable import EmbedKit

@Suite("Core ML Integration Tests")
struct CoreMLIntegrationTests {
    
    // Helper to check if a Core ML model is available
    static var isModelAvailable: Bool {
        Bundle.main.url(
            forResource: "all-MiniLM-L6-v2",
            withExtension: "mlpackage"
        ) != nil || Bundle.main.url(
            forResource: "all-MiniLM-L6-v2",
            withExtension: "mlmodelc"
        ) != nil
    }
    
    @Test("Core ML model integration", .enabled(if: isModelAvailable))
    func testCoreMLModelIntegration() async throws {
        
        // Create model loader
        let loader = CoreMLModelLoader()
        
        // Load model
        let (mlModel, metadata) = try await loader.loadPretrainedModel(.miniLM)
        
        // Verify metadata
        #expect(metadata.embeddingDimensions == 384)
        #expect(metadata.maxSequenceLength == 256)
        #expect(metadata.modelType == "coreml")
        
        // Create embedder with the model
        let modelId = ModelIdentifier(family: "all-MiniLM", variant: "L6-v2", version: "v1")
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: modelId,
            configuration: Configuration.default(for: modelId)
        )
        
        // Test embedding
        let text = "The quick brown fox jumps over the lazy dog."
        let embedding = try await embedder.embed(text)
        
        // Verify embedding
        #expect(embedding.dimensions == 384)
        
        // Test similarity
        let text2 = "A fast brown fox leaps over a sleepy dog."
        let embedding2 = try await embedder.embed(text2)
        
        let similarity = embedding.cosineSimilarity(with: embedding2)
        #expect(similarity > 0.8) // Should be similar
        
        // Test dissimilar texts
        let text3 = "Machine learning is transforming technology."
        let embedding3 = try await embedder.embed(text3)
        
        let similarity2 = embedding.cosineSimilarity(with: embedding3)
        #expect(similarity2 < 0.5) // Should be dissimilar
    }
    
    @Test("Batch processing performance", .enabled(if: isModelAvailable))
    func testBatchProcessingPerformance() async throws {
        
        let modelId = ModelIdentifier(family: "all-MiniLM", variant: "L6-v2", version: "v1")
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: modelId,
            configuration: Configuration(
                model: ModelConfiguration.custom(
                    identifier: modelId,
                    maxSequenceLength: 512
                ),
                resources: ResourceConfiguration(
                    batchSize: 32
                )
            )
        )
        
        try await embedder.loadModel()
        
        // Prepare test data
        let texts = (0..<100).map { i in
            "This is test sentence number \(i) for batch processing performance testing."
        }
        
        // Measure performance
        let startTime = CFAbsoluteTimeGetCurrent()
        let embeddings = try await embedder.embed(batch: texts)
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        
        #expect(embeddings.count == 100)
        
        let embeddingsPerSecond = Double(texts.count) / duration
        print("Performance: \(String(format: "%.1f", embeddingsPerSecond)) embeddings/second")
        
        // Should achieve at least 50 embeddings/second on modern hardware
        #expect(embeddingsPerSecond > 50)
    }
    
    @Test("Model memory footprint", .enabled(if: isModelAvailable))
    func testModelMemoryFootprint() async throws {
        let loader = CoreMLModelLoader()
        
        // Measure memory before loading
        let memoryBefore = ProcessInfo.processInfo.physicalMemory
        
        // Load model
        _ = try await loader.loadPretrainedModel(.miniLM)
        
        // Measure memory after loading
        let memoryAfter = ProcessInfo.processInfo.physicalMemory
        let memoryUsed = Int64(memoryBefore) - Int64(memoryAfter)
        let memoryUsedMB = Double(memoryUsed) / 1024 / 1024
        
        print("Model memory footprint: \(String(format: "%.1f", memoryUsedMB)) MB")
        
        // Model should use less than 100MB in memory
        #expect(memoryUsedMB < 100)
    }
}

// MARK: - Test Helpers

extension CoreMLIntegrationTests {
    /// Create a simple test model for unit testing
    static func createTestModel() throws -> URL {
        // This would create a minimal Core ML model for testing
        // In practice, you'd use coremltools in Python to create this
        fatalError("Implement test model creation")
    }
}