// EmbedKit - Error Types

import Foundation
import VectorCore

/// Common errors exposed by EmbedKit
public enum EmbedKitError: LocalizedError, Sendable {
    case modelNotFound(ModelID)
    case modelLoadFailed(String)
    case tokenizationFailed(String)
    case inferenceFailed(String)
    case dimensionMismatch(expected: Int, got: Int)
    case deviceNotAvailable(ComputeDevice)
    case inputTooLong(length: Int, max: Int)
    case batchSizeExceeded(size: Int, max: Int)
    case processingTimeout
    case invalidConfiguration(String)

    // MARK: - Metal Acceleration Errors
    /// Metal device (GPU) is not available on this system
    case metalDeviceUnavailable
    /// Failed to create Metal buffer for GPU operation
    case metalBufferFailed
    /// Metal compute pipeline not found for the specified kernel
    case metalPipelineNotFound(String)
    /// Failed to create Metal command encoder
    case metalEncoderFailed

    public var errorDescription: String? {
        switch self {
        case .inferenceFailed(let reason):
            return "Inference failed: \(reason)"
        case .modelNotFound(let id):
            return "Model not found: \(id)"
        case .modelLoadFailed(let reason):
            return "Failed to load model: \(reason)"
        case .tokenizationFailed(let reason):
            return "Tokenization failed: \(reason)"
        case .dimensionMismatch(let expected, let got):
            return "Dimension mismatch: expected \(expected), got \(got)"
        case .deviceNotAvailable(let device):
            return "Device not available: \(device)"
        case .inputTooLong(let length, let max):
            return "Input too long: \(length) tokens (max: \(max))"
        case .batchSizeExceeded(let size, let max):
            return "Batch size exceeded: \(size) (max: \(max))"
        case .processingTimeout:
            return "Processing timeout"
        case .invalidConfiguration(let reason):
            return "Invalid configuration: \(reason)"
        case .metalDeviceUnavailable:
            return "Metal device (GPU) is not available"
        case .metalBufferFailed:
            return "Failed to create Metal buffer"
        case .metalPipelineNotFound(let kernel):
            return "Metal pipeline not found for kernel: \(kernel)"
        case .metalEncoderFailed:
            return "Failed to create Metal command encoder"
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .modelNotFound:
            return "Verify the model ID is correct and the model is registered with ModelManager. Use ModelManager.availableModels() to list registered models."

        case .modelLoadFailed:
            return "Check that the model file exists at the expected path, is not corrupted, and is compatible with the current iOS/macOS version. For CoreML models, ensure the .mlpackage or .mlmodelc is properly formatted."

        case .tokenizationFailed:
            return "Verify the vocabulary file is present and properly formatted. Check that the input text uses supported characters and encoding (UTF-8)."

        case .inferenceFailed:
            return "Try reducing batch size or input length. If using GPU, try switching to CPU with `.cpuOnly` compute preference. Check system memory availability."

        case .dimensionMismatch:
            return "Ensure the embedding model output dimensions match the expected configuration. Check that you're using compatible embeddings from the same model."

        case .deviceNotAvailable:
            return "The requested compute device is not available. Try using `.auto` or `.cpuOnly` as compute preference, or check that Metal is supported on this device."

        case .inputTooLong:
            return "Shorten the input text or configure a truncation strategy (`.end`, `.start`, or `.middle`) in EmbeddingConfiguration. Consider chunking long documents."

        case .batchSizeExceeded:
            return "Reduce batch size in BatchOptions or use AdaptiveBatcher which automatically manages batch sizes based on system resources."

        case .processingTimeout:
            return "Increase the timeout duration, reduce input size/batch size, or try using a lighter compute device. Check if the system is under heavy load."

        case .invalidConfiguration:
            return "Review the configuration parameters. Check for missing required values, incompatible option combinations, or invalid ranges."

        case .metalDeviceUnavailable:
            return "Metal GPU acceleration is not available on this device. Use `.cpuOnly` compute preference for CPU-based processing, which works on all devices."

        case .metalBufferFailed:
            return "The system may be low on GPU memory. Try reducing batch size, using smaller inputs, or freeing GPU resources from other applications."

        case .metalPipelineNotFound:
            return "The Metal shader library may be missing or corrupted. Ensure EmbedKitShaders.metallib is included in the app bundle. Try recompiling shaders with CompileMetalShaders.sh."

        case .metalEncoderFailed:
            return "The GPU command queue may be overloaded. Try reducing concurrent GPU operations or adding delays between batches."
        }
    }

    /// Additional context about why this error occurred
    public var failureReason: String? {
        switch self {
        case .modelNotFound:
            return "The requested model identifier was not found in the model registry."

        case .modelLoadFailed:
            return "The model file could not be loaded into memory or compiled for execution."

        case .tokenizationFailed:
            return "The tokenizer encountered an error while processing the input text."

        case .inferenceFailed:
            return "The model inference operation did not complete successfully."

        case .dimensionMismatch:
            return "The vector dimensions do not match the expected size for this operation."

        case .deviceNotAvailable:
            return "The specified compute device could not be initialized or is not supported."

        case .inputTooLong:
            return "The input exceeds the maximum token length supported by the model."

        case .batchSizeExceeded:
            return "The number of items in the batch exceeds the configured maximum."

        case .processingTimeout:
            return "The operation did not complete within the allowed time limit."

        case .invalidConfiguration:
            return "One or more configuration parameters are invalid or incompatible."

        case .metalDeviceUnavailable:
            return "No Metal-compatible GPU was found on this system."

        case .metalBufferFailed:
            return "The GPU could not allocate memory for the requested buffer."

        case .metalPipelineNotFound:
            return "The compiled Metal shader function could not be located."

        case .metalEncoderFailed:
            return "The GPU command encoder could not be created from the command buffer."
        }
    }
}

// MARK: - VSKError Conformance

extension EmbedKitError: VSKError {
    /// Error code in EmbedKit range (2000-2999)
    public var errorCode: Int {
        VSKErrorCodeRange.embedKit.lowerBound + errorCodeOffset
    }

    /// Error code offset within EmbedKit range
    private var errorCodeOffset: Int {
        switch self {
        // Model errors: 0-99
        case .modelNotFound: return 0
        case .modelLoadFailed: return 1

        // Tokenization errors: 100-199
        case .tokenizationFailed: return 100
        case .inputTooLong: return 101

        // Inference errors: 200-299
        case .inferenceFailed: return 200
        case .processingTimeout: return 201
        case .batchSizeExceeded: return 202

        // Dimension/data errors: 300-399
        case .dimensionMismatch: return 300

        // Device errors: 400-499
        case .deviceNotAvailable: return 400

        // Configuration errors: 500-599
        case .invalidConfiguration: return 500

        // Metal errors: 600-699
        case .metalDeviceUnavailable: return 600
        case .metalBufferFailed: return 601
        case .metalPipelineNotFound: return 602
        case .metalEncoderFailed: return 603
        }
    }

    /// Error domain identifier
    public var domain: String { "EmbedKit" }

    /// Whether this error can potentially be recovered from
    public var isRecoverable: Bool {
        switch self {
        // Not recoverable - requires configuration/setup changes
        case .modelNotFound, .modelLoadFailed, .invalidConfiguration,
             .metalDeviceUnavailable, .metalPipelineNotFound:
            return false

        // Recoverable - can retry with different parameters
        case .tokenizationFailed, .inferenceFailed, .dimensionMismatch,
             .deviceNotAvailable, .inputTooLong, .batchSizeExceeded,
             .processingTimeout, .metalBufferFailed, .metalEncoderFailed:
            return true
        }
    }

    /// Error context with additional debugging information
    public var context: ErrorContext {
        var info: [String: String] = [:]

        // Add error-specific context
        switch self {
        case .modelNotFound(let id):
            info["modelID"] = id.description
        case .modelLoadFailed(let reason):
            info["reason"] = reason
        case .tokenizationFailed(let reason):
            info["reason"] = reason
        case .inferenceFailed(let reason):
            info["reason"] = reason
        case .dimensionMismatch(let expected, let got):
            info["expected"] = "\(expected)"
            info["got"] = "\(got)"
        case .deviceNotAvailable(let device):
            info["device"] = device.rawValue
        case .inputTooLong(let length, let max):
            info["length"] = "\(length)"
            info["maxLength"] = "\(max)"
        case .batchSizeExceeded(let size, let max):
            info["batchSize"] = "\(size)"
            info["maxBatchSize"] = "\(max)"
        case .metalPipelineNotFound(let kernel):
            info["kernel"] = kernel
        case .processingTimeout, .invalidConfiguration,
             .metalDeviceUnavailable, .metalBufferFailed, .metalEncoderFailed:
            break
        }

        // Add message from errorDescription
        if let desc = errorDescription {
            info["message"] = desc
        }

        return ErrorContext(additionalInfo: info)
    }

    /// Underlying error (none for EmbedKitError)
    public var underlyingError: (any Error)? { nil }
}
