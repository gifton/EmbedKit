// Tests for EmbedKitONNX Types
import Testing
import Foundation
@testable import EmbedKit
@testable import EmbedKitONNX

// MARK: - ONNX Input Tests

@Suite("ONNX Types - Input")
struct ONNXInputTests {

    @Test("ONNXInput initializes with basic properties")
    func basicInit() {
        let input = ONNXInput(
            tokenIDs: [101, 2054, 2003, 102],
            attentionMask: [1, 1, 1, 1]
        )

        #expect(input.tokenIDs.count == 4)
        #expect(input.attentionMask.count == 4)
        #expect(input.tokenTypeIDs == nil)
    }

    @Test("ONNXInput initializes with token type IDs")
    func withTokenTypeIDs() {
        let input = ONNXInput(
            tokenIDs: [101, 2054, 102],
            attentionMask: [1, 1, 1],
            tokenTypeIDs: [0, 0, 0]
        )

        #expect(input.tokenTypeIDs != nil)
        #expect(input.tokenTypeIDs?.count == 3)
    }

    @Test("ONNXInput creates from CoreMLInput")
    func fromCoreMLInput() {
        let coreMLInput = CoreMLInput(
            tokenIDs: [101, 2054, 102],
            attentionMask: [1, 1, 1]
        )

        let onnxInput = ONNXInput(from: coreMLInput)

        #expect(onnxInput.tokenIDs == coreMLInput.tokenIDs)
        #expect(onnxInput.attentionMask == coreMLInput.attentionMask)
    }
}

// MARK: - ONNX Output Tests

@Suite("ONNX Types - Output")
struct ONNXOutputTests {

    @Test("ONNXOutput initializes correctly")
    func basicInit() {
        let values: [Float] = [0.1, 0.2, 0.3, 0.4]
        let shape = [1, 2, 2]

        let output = ONNXOutput(values: values, shape: shape)

        #expect(output.values.count == 4)
        #expect(output.shape == [1, 2, 2])
    }

    @Test("ONNXOutput converts to CoreMLOutput")
    func toCoreMLOutput() {
        let onnxOutput = ONNXOutput(
            values: [0.1, 0.2, 0.3],
            shape: [1, 3]
        )

        let coreMLOutput = onnxOutput.toCoreMLOutput()

        #expect(coreMLOutput.values == onnxOutput.values)
        #expect(coreMLOutput.shape == onnxOutput.shape)
    }
}

// MARK: - ONNX Configuration Tests

@Suite("ONNX Types - Configuration")
struct ONNXConfigurationTests {

    @Test("Default configuration has sensible values")
    func defaultConfig() {
        let config = ONNXBackendConfiguration.default

        #expect(config.intraOpNumThreads == 0)  // Auto
        #expect(config.interOpNumThreads == 0)  // Auto
        #expect(config.useCoreMLProvider == true)
        #expect(config.graphOptimizationLevel == .all)
    }

    @Test("CPU-only configuration disables CoreML provider")
    func cpuOnlyConfig() {
        let config = ONNXBackendConfiguration.cpuOnly

        #expect(config.useCoreMLProvider == false)
        #expect(config.intraOpNumThreads > 0)
    }

    @Test("Custom configuration is respected")
    func customConfig() {
        let config = ONNXBackendConfiguration(
            intraOpNumThreads: 8,
            interOpNumThreads: 4,
            useCoreMLProvider: false,
            graphOptimizationLevel: .basic
        )

        #expect(config.intraOpNumThreads == 8)
        #expect(config.interOpNumThreads == 4)
        #expect(config.useCoreMLProvider == false)
        #expect(config.graphOptimizationLevel == .basic)
    }

    @Test("Graph optimization levels have correct raw values")
    func optimizationLevels() {
        #expect(ONNXBackendConfiguration.GraphOptimizationLevel.disabled.rawValue == 0)
        #expect(ONNXBackendConfiguration.GraphOptimizationLevel.basic.rawValue == 1)
        #expect(ONNXBackendConfiguration.GraphOptimizationLevel.extended.rawValue == 2)
        #expect(ONNXBackendConfiguration.GraphOptimizationLevel.all.rawValue == 99)
    }
}

// MARK: - ONNX Error Tests

@Suite("ONNX Types - Errors")
struct ONNXErrorTests {

    @Test("ONNXBackendError has descriptive messages")
    func errorDescriptions() {
        let errors: [ONNXBackendError] = [
            .modelNotLoaded,
            .modelLoadFailed("test failure"),
            .inferenceError("inference failed"),
            .invalidInput("bad input"),
            .invalidOutput("bad output"),
            .sessionCreationFailed("session error"),
            .unsupportedModel("unsupported")
        ]

        for error in errors {
            #expect(error.errorDescription != nil)
            #expect(!error.errorDescription!.isEmpty)
        }
    }

    @Test("Error messages include context")
    func errorContext() {
        let error = ONNXBackendError.modelLoadFailed("file not found")

        #expect(error.errorDescription?.contains("file not found") == true)
    }
}
