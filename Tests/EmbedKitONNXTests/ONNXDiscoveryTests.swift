// Tests for ONNX Model Discovery
import Testing
import Foundation
@testable import EmbedKit
@testable import EmbedKitONNX

// MARK: - ONNX File Type Tests

@Suite("ONNX Discovery - File Types")
struct ONNXFileTypeTests {

    @Test("ONNX file type is in all cases")
    func onnxInAllCases() {
        let types = ModelFileType.allCases

        #expect(types.contains(.onnx))
    }

    @Test("ONNX file type has correct extension")
    func onnxExtension() {
        let type = ModelFileType.onnx

        #expect(type.extensions.contains("onnx"))
    }

    @Test("ONNX requires ONNX module")
    func onnxRequiresModule() {
        #expect(ModelFileType.onnx.requiresONNXModule == true)
        #expect(ModelFileType.mlpackage.requiresONNXModule == false)
        #expect(ModelFileType.mlmodel.requiresONNXModule == false)
    }
}

// MARK: - ONNX Discovery Options Tests

@Suite("ONNX Discovery - Options")
struct ONNXDiscoveryOptionsTests {

    @Test("Default options include ONNX")
    func defaultIncludesONNX() {
        let options = ModelDiscoveryOptions.default

        #expect(options.fileTypes.contains(.onnx))
    }

    @Test("Can filter to ONNX only")
    func onnxOnlyFilter() {
        let options = ModelDiscoveryOptions(
            recursive: true,
            maxDepth: 5,
            fileTypes: [.onnx],
            minSizeBytes: 1024,
            embeddingModelsOnly: false
        )

        #expect(options.fileTypes.count == 1)
        #expect(options.fileTypes.contains(.onnx))
    }

    @Test("Can exclude ONNX")
    func excludeONNX() {
        let options = ModelDiscoveryOptions(
            fileTypes: [.mlpackage, .mlmodel]
        )

        #expect(!options.fileTypes.contains(.onnx))
    }
}

// MARK: - ONNX Model Family Tests

@Suite("ONNX Discovery - Families")
struct ONNXModelFamilyTests {

    @Test("HuggingFace model names detected")
    func huggingFaceNames() {
        // Common HuggingFace ONNX model naming patterns
        #expect(ModelFamily.infer(from: "all-MiniLM-L6-v2") == .miniLM)
        #expect(ModelFamily.infer(from: "sentence-transformers-all-mpnet-base-v2") == .mpnet)
        #expect(ModelFamily.infer(from: "distilbert-base-uncased") == .distilBERT)
    }

    @Test("E5 models detected")
    func e5Models() {
        #expect(ModelFamily.infer(from: "e5-small-v2") == .e5)
        #expect(ModelFamily.infer(from: "multilingual-e5-large") == .e5)
        #expect(ModelFamily.infer(from: "intfloat-e5-base") == .e5)
    }

    @Test("BGE models detected")
    func bgeModels() {
        #expect(ModelFamily.infer(from: "bge-small-en-v1.5") == .bge)
        #expect(ModelFamily.infer(from: "BAAI-bge-base") == .bge)
    }

    @Test("GTE models detected")
    func gteModels() {
        #expect(ModelFamily.infer(from: "gte-small") == .gte)
        #expect(ModelFamily.infer(from: "thenlper-gte-base") == .gte)
    }
}

// MARK: - Discovered ONNX Model Tests

@Suite("ONNX Discovery - Discovered Model")
struct DiscoveredONNXModelTests {

    @Test("ONNX model has correct type")
    func onnxModelType() {
        let model = DiscoveredModel(
            path: URL(fileURLWithPath: "/tmp/model.onnx"),
            type: .onnx,
            sizeBytes: 50_000_000,
            modifiedDate: Date(),
            name: "all-MiniLM-L6-v2",
            family: .miniLM,
            inferredDimensions: 384,
            isLikelyEmbeddingModel: true
        )

        #expect(model.type == .onnx)
        #expect(model.type.requiresONNXModule == true)
    }

    @Test("ONNX model ID includes path")
    func onnxModelId() {
        let model = DiscoveredModel(
            path: URL(fileURLWithPath: "/models/embed.onnx"),
            type: .onnx,
            sizeBytes: 1000,
            modifiedDate: Date(),
            name: "embed",
            family: nil,
            inferredDimensions: nil,
            isLikelyEmbeddingModel: false
        )

        #expect(model.id == "/models/embed.onnx")
    }
}
