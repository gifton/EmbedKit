import Foundation
import VectorCore

/// Represents a generated embedding with its metadata
public struct Embedding: Sendable, Codable {
    /// The vector data
    public let vector: [Float]
    
    /// Metadata about the generation process
    public let metadata: EmbeddingMetadata
    
    public init(vector: [Float], metadata: EmbeddingMetadata) {
        self.vector = vector
        self.metadata = metadata
    }
}

public struct EmbeddingMetadata: Sendable, Codable {
    public let modelID: String
    public let tokenCount: Int
    public let processingTime: TimeInterval
    
    public init(modelID: String, tokenCount: Int, processingTime: TimeInterval) {
        self.modelID = modelID
        self.tokenCount = tokenCount
        self.processingTime = processingTime
    }
}

/// Result of a batch embedding operation
public struct BatchResult: Sendable {
    public let embeddings: [Embedding]
    public let totalTime: TimeInterval
    public let averageLatency: TimeInterval
    public let totalTokens: Int
    
    public init(embeddings: [Embedding], totalTime: TimeInterval, totalTokens: Int) {
        self.embeddings = embeddings
        self.totalTime = totalTime
        self.totalTokens = totalTokens
        self.averageLatency = totalTime / Double(embeddings.count)
    }
}

/// Configuration for model loading and inference
public struct ModelConfiguration: Sendable {
    public let device: ComputeDevice
    public let batchSize: Int
    public let quantize: Bool
    
    public static let `default` = ModelConfiguration(
        device: .cpu,
        batchSize: 32,
        quantize: false
    )
    
    public init(device: ComputeDevice, batchSize: Int, quantize: Bool) {
        self.device = device
        self.batchSize = batchSize
        self.quantize = quantize
    }
}

/// EmbedKit specific errors
public enum EmbedKitError: Error {
    case modelNotFound(String)
    case modelLoadFailed(String, Error)
    case tokenizationFailed(String)
    case inferenceFailed(String)
    case invalidInput(String)
}
