import XCTest
@testable import EmbedKit

final class CoreComponentsTests: XCTestCase {
    
    // MARK: - Tokenizer Tests
    
    func testBertTokenizer() async throws {
        let vocab = [
            "[CLS]": 0,
            "[SEP]": 1,
            "[UNK]": 2,
            "hello": 3,
            "world": 4,
            "embed": 5,
            "##ding": 6
        ]
        
        let tokenizer = BertTokenizer(vocabulary: vocab)
        
        // Test encoding
        let text = "Hello world embedding"
        let ids = try await tokenizer.encode(text)
        
        // Expected: [CLS], hello, world, embed, ##ding, [SEP]
        // IDs: 0, 3, 4, 5, 6, 1
        XCTAssertEqual(ids, [0, 3, 4, 5, 6, 1])
        
        // Test decoding
        let decoded = try await tokenizer.decode(ids)
        XCTAssertEqual(decoded, "hello world embedding")
    }
    
    // MARK: - ModelManager Tests
    
    func testModelManager() async throws {
        let manager = ModelManager()
        let modelID = "test-model"
        
        // Load mock model
        let model = try await manager.loadModel(id: modelID) {
            MockEmbeddingModel(id: modelID)
        }
        
        XCTAssertNotNil(model)
        let id = await model.id
        XCTAssertEqual(id, modelID)
        
        // Verify retrieval
        let retrieved = await manager.getModel(id: modelID)
        XCTAssertNotNil(retrieved)
        if let retrievedModel = retrieved {
            let retrievedID = await retrievedModel.id
            XCTAssertEqual(retrievedID, modelID)
        } else {
            XCTFail("Model not retrieved")
        }
        
        // Verify embedding
        let embedding = try await retrieved!.embed("test")
        XCTAssertEqual(embedding.vector.count, 10)
        
        // Unload
        await manager.unloadModel(id: modelID)
        let unloaded = await manager.getModel(id: modelID)
        XCTAssertNil(unloaded)
    }
}

// Mock Model for testing
actor MockEmbeddingModel: EmbeddingModel {
    let id: String
    let dimension: Int = 10
    
    init(id: String) {
        self.id = id
    }
    
    func embed(_ text: String) async throws -> Embedding {
        return Embedding(
            vector: Array(repeating: 0.1, count: dimension),
            metadata: EmbeddingMetadata(
                modelID: id,
                tokenCount: 5,
                processingTime: 0.01
            )
        )
    }
    
    func embedBatch(_ texts: [String]) async throws -> BatchResult {
        var embeddings: [Embedding] = []
        for text in texts {
            embeddings.append(try await embed(text))
        }
        return BatchResult(
            embeddings: embeddings,
            totalTime: 0.05,
            totalTokens: 5 * texts.count
        )
    }
}

extension Sequence {
    func asyncMap<T>(
        _ transform: (Element) async throws -> T
    ) async rethrows -> [T] {
        var values = [T]()
        for element in self {
            try await values.append(transform(element))
        }
        return values
    }
}
