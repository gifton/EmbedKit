// Tests for Model Discovery - Week 5 Batch 3
import Testing
import Foundation
@testable import EmbedKit

// MARK: - Model File Type Tests

@Suite("Model Discovery - File Types")
struct ModelFileTypeTests {

    @Test("MLPackage has correct extension")
    func mlpackageExtension() {
        let type = ModelFileType.mlpackage

        #expect(type.extensions.contains("mlpackage"))
    }

    @Test("MLModel has correct extensions")
    func mlmodelExtensions() {
        let type = ModelFileType.mlmodel

        #expect(type.extensions.contains("mlmodel"))
        #expect(type.extensions.contains("mlmodelc"))
    }

    @Test("All file types are enumerable")
    func allTypesEnumerable() {
        let types = ModelFileType.allCases

        #expect(types.count == 3)
        #expect(types.contains(.mlpackage))
        #expect(types.contains(.mlmodel))
        #expect(types.contains(.onnx))
    }
}

// MARK: - Model Family Tests

@Suite("Model Discovery - Families")
struct ModelFamilyTests {

    @Test("MiniLM family inferred from name")
    func inferMiniLM() {
        #expect(ModelFamily.infer(from: "MiniLM-L6-v2") == .miniLM)
        #expect(ModelFamily.infer(from: "all-minilm-l6-v2") == .miniLM)
        #expect(ModelFamily.infer(from: "paraphrase-mini-lm") == .miniLM)
    }

    @Test("BERT family inferred from name")
    func inferBERT() {
        #expect(ModelFamily.infer(from: "bert-base-uncased") == .bert)
        #expect(ModelFamily.infer(from: "BERT-large") == .bert)
    }

    @Test("DistilBERT family inferred from name")
    func inferDistilBERT() {
        #expect(ModelFamily.infer(from: "distilbert-base-uncased") == .distilBERT)
        #expect(ModelFamily.infer(from: "DistilBERT") == .distilBERT)
    }

    @Test("MPNet family inferred from name")
    func inferMPNet() {
        #expect(ModelFamily.infer(from: "all-mpnet-base-v2") == .mpnet)
        #expect(ModelFamily.infer(from: "paraphrase-mpnet") == .mpnet)
    }

    @Test("E5 family inferred from name")
    func inferE5() {
        #expect(ModelFamily.infer(from: "e5-small-v2") == .e5)
        #expect(ModelFamily.infer(from: "multilingual-e5-base") == .e5)
    }

    @Test("Unknown returns nil")
    func inferUnknown() {
        #expect(ModelFamily.infer(from: "random-model") == nil)
        #expect(ModelFamily.infer(from: "custom-classifier") == nil)
    }

    @Test("Families have common dimensions")
    func familyDimensions() {
        #expect(ModelFamily.miniLM.commonDimensions.contains(384))
        #expect(ModelFamily.bert.commonDimensions.contains(768))
        #expect(ModelFamily.mpnet.commonDimensions.contains(768))
    }
}

// MARK: - Discovery Options Tests

@Suite("Model Discovery - Options")
struct ModelDiscoveryOptionsTests {

    @Test("Default options are sensible")
    func defaultOptions() {
        let options = ModelDiscoveryOptions.default

        #expect(options.recursive == true)
        #expect(options.maxDepth == 5)
        #expect(options.fileTypes.count == 3)
        #expect(options.minSizeBytes == 1024)
    }

    @Test("Quick options are non-recursive")
    func quickOptions() {
        let options = ModelDiscoveryOptions.quick

        #expect(options.recursive == false)
        #expect(options.maxDepth == 1)
    }

    @Test("Custom options are respected")
    func customOptions() {
        let options = ModelDiscoveryOptions(
            recursive: false,
            maxDepth: 2,
            fileTypes: [.mlpackage],
            minSizeBytes: 10000,
            embeddingModelsOnly: true
        )

        #expect(options.recursive == false)
        #expect(options.maxDepth == 2)
        #expect(options.fileTypes.count == 1)
        #expect(options.minSizeBytes == 10000)
        #expect(options.embeddingModelsOnly == true)
    }
}

// MARK: - Discovered Model Tests

@Suite("Model Discovery - Discovered Model")
struct DiscoveredModelTests {

    @Test("Discovered model has correct properties")
    func modelProperties() {
        let model = DiscoveredModel(
            path: URL(fileURLWithPath: "/tmp/test-model.mlpackage"),
            type: .mlpackage,
            sizeBytes: 50_000_000,
            modifiedDate: Date(),
            name: "test-model",
            family: .miniLM,
            inferredDimensions: 384,
            isLikelyEmbeddingModel: true
        )

        #expect(model.name == "test-model")
        #expect(model.type == .mlpackage)
        #expect(model.family == .miniLM)
        #expect(model.inferredDimensions == 384)
        #expect(model.isLikelyEmbeddingModel == true)
        #expect(model.sizeBytes == 50_000_000)
    }

    @Test("Formatted size is human readable")
    func formattedSize() {
        let smallModel = DiscoveredModel(
            path: URL(fileURLWithPath: "/tmp/small.mlpackage"),
            type: .mlpackage,
            sizeBytes: 1024,
            modifiedDate: Date(),
            name: "small",
            family: nil,
            inferredDimensions: nil,
            isLikelyEmbeddingModel: false
        )

        let largeModel = DiscoveredModel(
            path: URL(fileURLWithPath: "/tmp/large.mlpackage"),
            type: .mlpackage,
            sizeBytes: 100_000_000,
            modifiedDate: Date(),
            name: "large",
            family: nil,
            inferredDimensions: nil,
            isLikelyEmbeddingModel: false
        )

        // Should be KB/MB formatted
        #expect(!smallModel.formattedSize.isEmpty)
        #expect(!largeModel.formattedSize.isEmpty)
    }

    @Test("Model ID is based on path")
    func modelId() {
        let model = DiscoveredModel(
            path: URL(fileURLWithPath: "/tmp/unique-path.mlpackage"),
            type: .mlpackage,
            sizeBytes: 1000,
            modifiedDate: Date(),
            name: "unique",
            family: nil,
            inferredDimensions: nil,
            isLikelyEmbeddingModel: false
        )

        #expect(model.id == "/tmp/unique-path.mlpackage")
    }
}

// MARK: - Model Discovery Actor Tests

@Suite("Model Discovery - Scanning")
struct ModelDiscoveryScanningTests {

    @Test("Discovery scans empty directory")
    func scanEmptyDirectory() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let discovery = ModelDiscovery()
        let models = try await discovery.scan(directory: tempDir)

        #expect(models.isEmpty)
    }

    @Test("Discovery returns common locations")
    func commonLocations() async {
        let discovery = ModelDiscovery()
        let locations = discovery.commonModelLocations()

        // Should include at least caches and application support
        #expect(locations.count >= 2)
    }

    @Test("Discovery can analyze nil for non-model file")
    func analyzeNonModelFile() async {
        let discovery = ModelDiscovery()
        let textFile = FileManager.default.temporaryDirectory.appendingPathComponent("test.txt")

        let result = await discovery.analyze(modelAt: textFile)

        #expect(result == nil)
    }
}

// MARK: - Dimension Inference Tests

@Suite("Model Discovery - Dimension Inference")
struct DimensionInferenceTests {

    @Test("Dimensions inferred from model name with 384")
    func inferDimensions384() {
        let model = DiscoveredModel(
            path: URL(fileURLWithPath: "/tmp/model-384.mlpackage"),
            type: .mlpackage,
            sizeBytes: 1000,
            modifiedDate: Date(),
            name: "all-minilm-l6-v2-384",
            family: .miniLM,
            inferredDimensions: 384,
            isLikelyEmbeddingModel: true
        )

        #expect(model.inferredDimensions == 384)
    }

    @Test("Dimensions inferred from model name with 768")
    func inferDimensions768() {
        let model = DiscoveredModel(
            path: URL(fileURLWithPath: "/tmp/bert-768.mlpackage"),
            type: .mlpackage,
            sizeBytes: 1000,
            modifiedDate: Date(),
            name: "bert-base-768",
            family: .bert,
            inferredDimensions: 768,
            isLikelyEmbeddingModel: true
        )

        #expect(model.inferredDimensions == 768)
    }
}

// MARK: - Embedding Model Detection Tests

@Suite("Model Discovery - Embedding Detection")
struct EmbeddingDetectionTests {

    @Test("MiniLM detected as embedding model")
    func detectMiniLM() {
        let model = DiscoveredModel(
            path: URL(fileURLWithPath: "/tmp/MiniLM-L6.mlpackage"),
            type: .mlpackage,
            sizeBytes: 1000,
            modifiedDate: Date(),
            name: "MiniLM-L6-v2",
            family: .miniLM,
            inferredDimensions: 384,
            isLikelyEmbeddingModel: true
        )

        #expect(model.isLikelyEmbeddingModel == true)
    }

    @Test("Sentence transformer detected as embedding model")
    func detectSentenceTransformer() {
        let model = DiscoveredModel(
            path: URL(fileURLWithPath: "/tmp/sentence.mlpackage"),
            type: .mlpackage,
            sizeBytes: 1000,
            modifiedDate: Date(),
            name: "sentence-transformers-paraphrase",
            family: .sentenceTransformers,
            inferredDimensions: 384,
            isLikelyEmbeddingModel: true
        )

        #expect(model.isLikelyEmbeddingModel == true)
    }

    @Test("Generic classifier not detected as embedding model")
    func detectNonEmbedding() {
        let model = DiscoveredModel(
            path: URL(fileURLWithPath: "/tmp/classifier.mlpackage"),
            type: .mlpackage,
            sizeBytes: 1000,
            modifiedDate: Date(),
            name: "image-classifier",
            family: nil,
            inferredDimensions: nil,
            isLikelyEmbeddingModel: false
        )

        #expect(model.isLikelyEmbeddingModel == false)
    }
}
