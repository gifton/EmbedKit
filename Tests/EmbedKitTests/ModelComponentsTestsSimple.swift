import Foundation
import Testing
import CoreML
@testable import EmbedKit

@Suite("Model Components Tests - Simple")
struct ModelComponentsTestsSimple {
    
    // MARK: - CoreMLBackend Tests
    
    @Test("CoreMLBackend initialization and state")
    func testCoreMLBackendInitialization() async throws {
        let backend = CoreMLBackend(identifier: "test-backend")
        
        // Initial state should be not loaded
        #expect(await !backend.isLoaded)
        
        // Should return nil for metadata when not loaded
        let metadata = await backend.metadata
        #expect(metadata == nil)
    }
    
    // MARK: - EmbeddingModelManager Tests
    
    @Test("DefaultEmbeddingModelManager initialization")
    func testDefaultEmbeddingModelManagerInitialization() async {
        let manager = DefaultEmbeddingModelManager()
        
        // Should start with no loaded models
        let loadedModels = await manager.loadedModels()
        #expect(loadedModels.isEmpty)
    }
    
    // MARK: - ModelVersioning Tests
    
    @Test("ModelVersion comparison")
    func testModelVersionComparison() {
        let v1 = ModelVersion(
            identifier: "test-model",
            version: "1.0.0",
            buildNumber: 100,
            createdAt: Date(),
            metadata: [:]
        )
        
        let v2 = ModelVersion(
            identifier: "test-model",
            version: "2.0.0",
            buildNumber: 200,
            createdAt: Date().addingTimeInterval(3600),
            metadata: [:]
        )
        
        let v1_1 = ModelVersion(
            identifier: "test-model",
            version: "1.0.0",
            buildNumber: 101,
            createdAt: Date().addingTimeInterval(1800),
            metadata: [:]
        )
        
        #expect(v2.isNewer(than: v1))
        #expect(!v1.isNewer(than: v2))
        #expect(v1_1.isNewer(than: v1)) // Same version but newer build
        
        #expect(v1.semanticVersion == "1.0.0.100")
        #expect(v2.semanticVersion == "2.0.0.200")
    }
    
    @Test("ModelVersion Hashable conformance")
    func testModelVersionHashable() {
        let date = Date()
        let v1 = ModelVersion(
            identifier: "test-model",
            version: "1.0.0",
            buildNumber: 100,
            createdAt: date,
            metadata: [:]
        )
        
        let v2 = ModelVersion(
            identifier: "test-model",
            version: "1.0.0",
            buildNumber: 100,
            createdAt: date,
            metadata: [:]
        )
        
        // Should be equal
        #expect(v1 == v2)
        
        // Should work in Set
        let set = Set([v1, v2])
        #expect(set.count == 1)
    }
    
    @Test("ModelVersionRegistry basic operations")
    func testModelVersionRegistryBasicOperations() async throws {
        let registry = ModelVersionRegistry()
        
        // Create test file
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("ModelVersionTest_\(UUID())")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        let modelFile = tempDir.appendingPathComponent("test_model.mlmodel")
        try "test model".data(using: .utf8)!.write(to: modelFile)
        
        // Register version
        let version = ModelVersion(
            identifier: "test-model",
            version: "1.0.0",
            buildNumber: 100,
            createdAt: Date(),
            metadata: ["key": "value"]
        )
        
        try await registry.register(version: version, modelURL: modelFile)
        
        // Get versions
        let versions = await registry.getVersions(for: "test-model")
        #expect(versions.count == 1)
        #expect(versions.first == version)
        
        // Get active version
        let activeVersion = await registry.getActiveVersion(for: "test-model")
        #expect(activeVersion == version)
        
        // Get model URL
        let modelURL = await registry.getModelURL(for: version)
        #expect(modelURL == modelFile)
        
        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }
    
    @Test("HotSwappableModelManager initialization")
    func testHotSwappableModelManagerInitialization() async {
        let registry = ModelVersionRegistry()
        let manager = HotSwappableModelManager(registry: registry, maxConcurrentModels: 3)
        
        // Basic initialization test - just ensure it can be created
        // Manager is created successfully, no assertion needed
    }
}

// MARK: - Mock Implementations

actor MockModelBackend: ModelBackend {
    let id: Int
    var unloadCalled = false
    var isLoadedValue = true
    
    init(id: Int = 0) {
        self.id = id
    }
    
    var identifier: String { "mock-backend-\(id)" }
    
    var isLoaded: Bool { isLoadedValue }
    
    var metadata: ModelMetadata? {
        ModelMetadata(
            name: "mock-model",
            version: "1.0.0",
            embeddingDimensions: 768,
            maxSequenceLength: 512,
            vocabularySize: 30522,
            modelType: "mock",
            additionalInfo: [:]
        )
    }
    
    func loadModel(from url: URL) async throws {
        isLoadedValue = true
    }
    
    func unloadModel() async throws {
        unloadCalled = true
        isLoadedValue = false
    }
    
    func generateEmbeddings(for input: TokenizedInput) async throws -> ModelOutput {
        // Return mock embedding output
        let embeddings = [[Float]](repeating: [Float](repeating: 0.1, count: 768), count: input.tokenIds.count)
        return ModelOutput(
            tokenEmbeddings: embeddings,
            attentionWeights: nil,
            metadata: ["model": "mock-\(id)"]
        )
    }
    
    func inputDimensions() async -> (sequence: Int, features: Int)? {
        return (512, 768)
    }
    
    func outputDimensions() async -> Int? {
        return 768
    }
}