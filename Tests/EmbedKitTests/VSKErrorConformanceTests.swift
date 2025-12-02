// EmbedKit - VSKError Conformance Tests

import Testing
import Foundation
@testable import EmbedKit
import VectorCore

@Suite("EmbedKitError VSKError Conformance")
struct VSKErrorConformanceTests {

    // MARK: - Error Code Tests

    @Test("Error codes are in EmbedKit range (2000-2999)")
    func testErrorCodesInRange() {
        let errors: [EmbedKitError] = [
            .modelNotFound(ModelID(provider: "test", name: "model", version: "1.0")),
            .modelLoadFailed("test"),
            .tokenizationFailed("test"),
            .inferenceFailed("test"),
            .dimensionMismatch(expected: 384, got: 512),
            .deviceNotAvailable(.gpu),
            .inputTooLong(length: 1000, max: 512),
            .batchSizeExceeded(size: 100, max: 32),
            .processingTimeout,
            .invalidConfiguration("test"),
            .metalDeviceUnavailable,
            .metalBufferFailed,
            .metalPipelineNotFound("test_kernel"),
            .metalEncoderFailed,
            .metalTensorFailed("test")
        ]

        for error in errors {
            #expect(error.errorCode >= 2000, "Error code \(error.errorCode) should be >= 2000")
            #expect(error.errorCode < 3000, "Error code \(error.errorCode) should be < 3000")
        }
    }

    @Test("Error codes are unique")
    func testErrorCodesUnique() {
        let errors: [EmbedKitError] = [
            .modelNotFound(ModelID(provider: "test", name: "model", version: "1.0")),
            .modelLoadFailed("test"),
            .tokenizationFailed("test"),
            .inferenceFailed("test"),
            .dimensionMismatch(expected: 384, got: 512),
            .deviceNotAvailable(.gpu),
            .inputTooLong(length: 1000, max: 512),
            .batchSizeExceeded(size: 100, max: 32),
            .processingTimeout,
            .invalidConfiguration("test"),
            .metalDeviceUnavailable,
            .metalBufferFailed,
            .metalPipelineNotFound("test_kernel"),
            .metalEncoderFailed,
            .metalTensorFailed("test")
        ]

        var codes = Set<Int>()
        for error in errors {
            let inserted = codes.insert(error.errorCode).inserted
            #expect(inserted, "Duplicate error code: \(error.errorCode)")
        }
    }

    @Test("Model errors have codes 2000-2099")
    func testModelErrorCodes() {
        let modelNotFound = EmbedKitError.modelNotFound(
            ModelID(provider: "test", name: "model", version: "1.0")
        )
        let modelLoadFailed = EmbedKitError.modelLoadFailed("test")

        #expect(modelNotFound.errorCode == 2000)
        #expect(modelLoadFailed.errorCode == 2001)
    }

    @Test("Tokenization errors have codes 2100-2199")
    func testTokenizationErrorCodes() {
        let tokenizationFailed = EmbedKitError.tokenizationFailed("test")
        let inputTooLong = EmbedKitError.inputTooLong(length: 1000, max: 512)

        #expect(tokenizationFailed.errorCode == 2100)
        #expect(inputTooLong.errorCode == 2101)
    }

    @Test("Inference errors have codes 2200-2299")
    func testInferenceErrorCodes() {
        let inferenceFailed = EmbedKitError.inferenceFailed("test")
        let processingTimeout = EmbedKitError.processingTimeout
        let batchSizeExceeded = EmbedKitError.batchSizeExceeded(size: 100, max: 32)

        #expect(inferenceFailed.errorCode == 2200)
        #expect(processingTimeout.errorCode == 2201)
        #expect(batchSizeExceeded.errorCode == 2202)
    }

    @Test("Metal errors have codes 2600-2699")
    func testMetalErrorCodes() {
        let metalDeviceUnavailable = EmbedKitError.metalDeviceUnavailable
        let metalBufferFailed = EmbedKitError.metalBufferFailed
        let metalPipelineNotFound = EmbedKitError.metalPipelineNotFound("test")
        let metalEncoderFailed = EmbedKitError.metalEncoderFailed

        #expect(metalDeviceUnavailable.errorCode == 2600)
        #expect(metalBufferFailed.errorCode == 2601)
        #expect(metalPipelineNotFound.errorCode == 2602)
        #expect(metalEncoderFailed.errorCode == 2603)

        let metalTensorFailed = EmbedKitError.metalTensorFailed("test")
        #expect(metalTensorFailed.errorCode == 2604)
    }

    // MARK: - Domain Tests

    @Test("Domain is 'EmbedKit'")
    func testDomain() {
        let error = EmbedKitError.modelNotFound(
            ModelID(provider: "test", name: "model", version: "1.0")
        )
        #expect(error.domain == "EmbedKit")
    }

    // MARK: - Recoverability Tests

    @Test("Non-recoverable errors are correctly identified")
    func testNonRecoverableErrors() {
        let nonRecoverable: [EmbedKitError] = [
            .modelNotFound(ModelID(provider: "test", name: "model", version: "1.0")),
            .modelLoadFailed("test"),
            .invalidConfiguration("test"),
            .metalDeviceUnavailable,
            .metalPipelineNotFound("test"),
            .dimensionMismatch(expected: 384, got: 512)
        ]

        for error in nonRecoverable {
            #expect(!error.isRecoverable, "\(error) should not be recoverable")
        }
    }

    @Test("Recoverable errors are correctly identified")
    func testRecoverableErrors() {
        let recoverable: [EmbedKitError] = [
            .tokenizationFailed("test"),
            .inferenceFailed("test"),
            .deviceNotAvailable(.gpu),
            .inputTooLong(length: 1000, max: 512),
            .batchSizeExceeded(size: 100, max: 32),
            .processingTimeout,
            .metalBufferFailed,
            .metalEncoderFailed,
            .metalTensorFailed("test")
        ]

        for error in recoverable {
            #expect(error.isRecoverable, "\(error) should be recoverable")
        }
    }

    // MARK: - Context Tests

    @Test("Context contains error-specific information")
    func testContextContainsInfo() {
        let dimensionError = EmbedKitError.dimensionMismatch(expected: 384, got: 512)
        let context = dimensionError.context

        #expect(context.additionalInfo["expected"] == "384")
        #expect(context.additionalInfo["got"] == "512")
    }

    @Test("Context contains model ID for modelNotFound")
    func testContextModelID() {
        let modelID = ModelID(provider: "apple", name: "embedding", version: "1.0")
        let error = EmbedKitError.modelNotFound(modelID)
        let context = error.context

        #expect(context.additionalInfo["modelID"]?.contains("apple") == true)
        #expect(context.additionalInfo["modelID"]?.contains("embedding") == true)
    }

    @Test("Context contains message from errorDescription")
    func testContextContainsMessage() {
        let error = EmbedKitError.processingTimeout
        let context = error.context

        #expect(context.additionalInfo["message"] != nil)
        #expect(context.additionalInfo["message"]?.contains("timeout") == true)
    }

    @Test("Context contains reason for invalidConfiguration")
    func testContextInvalidConfiguration() {
        let error = EmbedKitError.invalidConfiguration("batch size must be positive")
        let context = error.context

        #expect(context.additionalInfo["reason"] == "batch size must be positive")
    }

    // MARK: - Recovery Suggestion Tests

    @Test("Recovery suggestions are provided")
    func testRecoverySuggestions() {
        let errors: [EmbedKitError] = [
            .modelNotFound(ModelID(provider: "test", name: "model", version: "1.0")),
            .inputTooLong(length: 1000, max: 512),
            .metalDeviceUnavailable
        ]

        for error in errors {
            #expect(error.recoverySuggestion != nil, "\(error) should have recovery suggestion")
            #expect(!error.recoverySuggestion!.isEmpty, "\(error) recovery suggestion should not be empty")
        }
    }

    // MARK: - Description Tests

    @Test("VSKError description format")
    func testDescriptionFormat() {
        let error = EmbedKitError.dimensionMismatch(expected: 384, got: 512)

        // VSKError description format: [domain:code] message
        let description = error.description
        #expect(description.contains("EmbedKit"))
        #expect(description.contains("2300"))  // dimension error code
    }

    // MARK: - Underlying Error Tests

    @Test("Underlying error is nil for EmbedKitError")
    func testUnderlyingErrorNil() {
        let error = EmbedKitError.inferenceFailed("test")
        #expect(error.underlyingError == nil)
    }
}
