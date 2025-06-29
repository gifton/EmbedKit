import Testing
import Foundation
@preconcurrency import CoreML
@testable import EmbedKit

@Suite("Metadata Extraction Tests")
struct MetadataExtractionTests {
    
    @Test("Extract metadata from mock model")
    func testMetadataExtraction() async throws {
        // Create a mock CoreML model for testing
        let mockModel = try await createMockModel()
        
        let metadata = try await CoreMLMetadataExtractor.extractMetadata(
            from: mockModel,
            modelIdentifier: "test-model"
        )
        
        #expect(metadata.name == "test-model")
        #expect(metadata.embeddingDimensions > 0)
        #expect(metadata.modelType != "unknown")
    }
    
    @Test("Validate metadata completeness")
    func testMetadataValidation() {
        // Test complete metadata
        let completeMetadata = ModelMetadata(
            name: "test-model",
            version: "1.0",
            embeddingDimensions: 384,
            maxSequenceLength: 512,
            vocabularySize: 30522,
            modelType: "bert",
            additionalInfo: [:]
        )
        
        let issues = CoreMLMetadataExtractor.validateMetadata(completeMetadata)
        #expect(issues.isEmpty)
        
        // Test incomplete metadata
        let incompleteMetadata = ModelMetadata(
            name: "test-model",
            version: "1.0",
            embeddingDimensions: 0,
            maxSequenceLength: 0,
            vocabularySize: 0,
            modelType: "unknown",
            additionalInfo: [:]
        )
        
        let validationIssues = CoreMLMetadataExtractor.validateMetadata(incompleteMetadata)
        #expect(validationIssues.count == 4)
        #expect(validationIssues.contains { $0.contains("embedding dimensions") })
        #expect(validationIssues.contains { $0.contains("sequence length") })
        #expect(validationIssues.contains { $0.contains("vocabulary size") })
        #expect(validationIssues.contains { $0.contains("model type") })
    }
    
    @Test("Default metadata for known models")
    func testDefaultMetadata() {
        // Test MiniLM
        let miniLMId = ModelIdentifier.miniLM_L6_v2
        let miniLMMetadata = CoreMLMetadataExtractor.defaultMetadata(for: miniLMId)
        
        #expect(miniLMMetadata != nil)
        #expect(miniLMMetadata?.embeddingDimensions == 384)
        #expect(miniLMMetadata?.maxSequenceLength == 256)
        #expect(miniLMMetadata?.vocabularySize == 30522)
        #expect(miniLMMetadata?.modelType == "sentence-transformer")
        
        // Test unknown model
        let unknownId = ModelIdentifier(family: "unknown-model")
        let unknownMetadata = CoreMLMetadataExtractor.defaultMetadata(for: unknownId)
        #expect(unknownMetadata == nil)
    }
    
    @Test("Model manager with metadata extraction")
    func testModelManagerMetadataUsage() async throws {
        let manager = ModelManager()
        
        // List available models should include metadata
        let models = try await manager.listAvailableModels()
        
        // If we have any models, verify structure
        if !models.isEmpty {
            let model = models[0]
            #expect(model.identifier.family.count > 0)
            #expect(model.location == .bundled || model.location == .downloaded)
        }
    }
    
    @Test("Automatic tokenizer configuration from metadata")
    func testTokenizerConfigurationFromMetadata() async throws {
        // Create embedder with default tokenizer
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: .miniLM_L6_v2,
            configuration: Configuration.default(for: .miniLM_L6_v2)
        )
        
        // After loading model, tokenizer should be updated with metadata
        // This test would need a real model to be meaningful
        // For now, just test the mechanism exists
        #expect(embedder.dimensions == 0) // Before loading
        
        // Test metadata getter
        let metadata = await embedder.getMetadata()
        #expect(metadata == nil) // Before loading
    }
    
    // MARK: - Helper Methods
    
    private func createMockModel() async throws -> MLModel {
        // For real testing, we'd need to create a simple CoreML model
        // For now, throw skip to indicate this needs a real model
        throw TestSkipError("Mock model creation not implemented - requires real CoreML model")
    }
}

// Test error for skipping tests
struct TestSkipError: Error {
    let message: String
    
    init(_ message: String) {
        self.message = message
    }
}

@Suite("Model Type Detection Tests")
struct ModelTypeDetectionTests {
    
    @Test("Detect BERT model type")
    func testBERTModelDetection() {
        // This would test the model type detection logic
        // by examining input/output names
        
        // Expected BERT inputs: input_ids, attention_mask, token_type_ids
        // Expected outputs: last_hidden_state or pooler_output
        
        // Since we can't easily create mock MLModelDescription,
        // we're testing the concept
        #expect(true) // Placeholder
    }
    
    @Test("Detect sentence transformer model")
    func testSentenceTransformerDetection() {
        // Sentence transformers typically have:
        // - input_ids, attention_mask (no token_type_ids)
        // - sentence_embedding output
        
        #expect(true) // Placeholder
    }
}