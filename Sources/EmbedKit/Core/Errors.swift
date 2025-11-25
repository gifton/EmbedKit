// EmbedKit - Error Types

import Foundation

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
}
