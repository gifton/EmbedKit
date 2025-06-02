import Testing
import Foundation
@testable import EmbedKit

/// Simple tests to verify basic functionality
@Suite("Simple Tests")
struct SimpleTests {
    
    @Test("Basic MockTextEmbedder functionality")
    func testMockTextEmbedder() async throws {
        let embedder = MockTextEmbedder(dimensions: 384)
        try await embedder.loadModel()
        
        let embedding = try await embedder.embed("Test text")
        #expect(embedding.dimensions == 384)
    }
    
    @Test("Configuration initialization")
    func testConfiguration() {
        let config = Configuration()
        #expect(config.resources.batchSize == 32)
        #expect(config.model.maxSequenceLength == 512)
    }
    
    @Test("ModelIdentifier creation")
    func testModelIdentifier() throws {
        let id = ModelIdentifier(family: "test", variant: "v1", version: "1.0")
        #expect(id.family == "test")
        #expect(id.variant == "v1") 
        #expect(id.version == "1.0")
    }
    
    @Test("EmbeddingVector operations")
    func testEmbeddingVector() async throws {
        let embedder = MockTextEmbedder(dimensions: 384)
        try await embedder.loadModel()
        
        let embedding1 = try await embedder.embed("Hello")
        let embedding2 = try await embedder.embed("Hello")
        let embedding3 = try await embedder.embed("Goodbye")
        
        // Same text should have high similarity
        let similarity1 = embedding1.cosineSimilarity(with: embedding2)
        #expect(similarity1 > 0.99)
        
        // Different text should have lower similarity
        let similarity2 = embedding1.cosineSimilarity(with: embedding3)
        #expect(similarity2 < similarity1)
    }
}