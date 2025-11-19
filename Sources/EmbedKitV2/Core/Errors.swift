// EmbedKitV2 - Error Types (Week 1 polish)

import Foundation

/// Common errors exposed by EmbedKitV2
public enum EmbedKitError: LocalizedError, Sendable {
    case modelNotFound(ModelID)
    case modelLoadFailed(String)
    case tokenizationFailed(String)
    case dimensionMismatch(expected: Int, got: Int)
    case deviceNotAvailable(ComputeDevice)
    case inputTooLong(length: Int, max: Int)
    case batchSizeExceeded(size: Int, max: Int)
    case processingTimeout
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
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
        }
    }
}

