import XCTest
import CoreML
@testable import EmbedKit

final class CoreMLIntegrationTests: XCTestCase {
    
    /// Test loading and using a converted Core ML model
    func testCoreMLModelIntegration() async throws {
        // Skip if no model is available
        guard Bundle(for: Self.self).url(
            forResource: "all-MiniLM-L6-v2",
            withExtension: "mlpackage"
        ) != nil else {
            throw XCTSkip("No Core ML model found in test bundle")
        }
        
        // Create model loader
        let loader = CoreMLModelLoader()
        
        // Load model
        let (mlModel, metadata) = try await loader.loadPretrainedModel(.miniLM)
        
        // Verify metadata
        XCTAssertEqual(metadata.embeddingDimensions, 384)
        XCTAssertEqual(metadata.maxSequenceLength, 256)
        XCTAssertEqual(metadata.modelFormat, .coreML)
        
        // Create embedder with the model
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: "all-MiniLM-L6-v2",
            configuration: EmbeddingConfiguration()
        )
        
        // Test embedding
        let text = "The quick brown fox jumps over the lazy dog."
        let embedding = try await embedder.embed(text)
        
        // Verify embedding
        XCTAssertEqual(embedding.dimensions, 384)
        
        // Test similarity
        let text2 = "A fast brown fox leaps over a sleepy dog."
        let embedding2 = try await embedder.embed(text2)
        
        let similarity = embedding.cosineSimilarity(with: embedding2)
        XCTAssertGreaterThan(similarity, 0.8) // Should be similar
        
        // Test dissimilar texts
        let text3 = "Machine learning is transforming technology."
        let embedding3 = try await embedder.embed(text3)
        
        let similarity2 = embedding.cosineSimilarity(with: embedding3)
        XCTAssertLessThan(similarity2, 0.5) // Should be dissimilar
    }
    
    /// Test batch processing performance
    func testBatchProcessingPerformance() async throws {
        guard Bundle(for: Self.self).url(
            forResource: "all-MiniLM-L6-v2",
            withExtension: "mlpackage"
        ) != nil else {
            throw XCTSkip("No Core ML model found in test bundle")
        }
        
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: "all-MiniLM-L6-v2",
            configuration: EmbeddingConfiguration(
                batchSize: 32,
                useGPUAcceleration: true
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
        
        XCTAssertEqual(embeddings.count, 100)
        
        let embeddingsPerSecond = Double(texts.count) / duration
        print("Performance: \(String(format: "%.1f", embeddingsPerSecond)) embeddings/second")
        
        // Should achieve at least 50 embeddings/second on modern hardware
        XCTAssertGreaterThan(embeddingsPerSecond, 50)
    }
    
    /// Test model size and memory usage
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
        XCTAssertLessThan(memoryUsedMB, 100)
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