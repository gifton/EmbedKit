import Foundation

/// Metal error types
public enum MetalError: LocalizedError {
    case deviceNotAvailable
    case commandQueueCreationFailed
    case bufferCreationFailed
    case pipelineNotFound(String)
    case encoderCreationFailed
    case commandBufferCreationFailed
    case invalidInput(String)
    case dimensionMismatch
    case functionNotFound(String)

    public var errorDescription: String? {
        switch self {
        case .deviceNotAvailable:
            return "Metal device not available"
        case .commandQueueCreationFailed:
            return "Failed to create Metal command queue"
        case .bufferCreationFailed:
            return "Failed to create Metal buffer"
        case .pipelineNotFound(let name):
            return "Metal compute pipeline '\(name)' not found"
        case .encoderCreationFailed:
            return "Failed to create compute encoder"
        case .commandBufferCreationFailed:
            return "Failed to create command buffer"
        case .invalidInput(let message):
            return "Invalid input: \(message)"
        case .dimensionMismatch:
            return "Vector dimensions do not match"
        case .functionNotFound(let name):
            return "Metal function '\(name)' not found in library"
        }
    }
}
