import Foundation
@preconcurrency import Metal
import Testing
@testable import EmbedKit

/// Test environment capability detection
public struct TestEnvironment {
    /// Check if Metal is available in the current test environment
    public static var hasMetalSupport: Bool {
        #if targetEnvironment(simulator)
        // Metal is not available in iOS/tvOS simulators
        return false
        #elseif os(Linux)
        // Metal is not available on Linux
        return false
        #else
        // Check if we can create a Metal device
        return MTLCreateSystemDefaultDevice() != nil
        #endif
    }
    
    /// Check if Core ML models are available for testing
    public static var hasCoreMLModels: Bool {
        // Check for test models in bundle or filesystem
        // For now, return false since we don't have test models bundled
        return false
    }
    
    /// Check if the system has enough memory for performance tests
    public static var hasHighMemory: Bool {
        ProcessInfo.processInfo.physicalMemory > 8_000_000_000 // 8GB
    }
    
    /// Check if we're running in CI environment
    public static var isCI: Bool {
        ProcessInfo.processInfo.environment["CI"] != nil ||
        ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != nil
    }
    
    /// Get appropriate test timeout based on environment
    public static var testTimeout: TimeInterval {
        isCI ? 60.0 : 30.0
    }
}

// MARK: - Test Traits

/// Custom trait for environment-based test execution
public struct EnvironmentTrait: TestTrait {
    let isEnabled: Bool
    let skipMessage: String
    
    public init(isEnabled: Bool, skipMessage: String) {
        self.isEnabled = isEnabled
        self.skipMessage = skipMessage
    }
}

extension Trait where Self == EnvironmentTrait {
    /// Requires Metal support to run
    public static var requiresMetal: Self {
        EnvironmentTrait(
            isEnabled: TestEnvironment.hasMetalSupport,
            skipMessage: "Metal not available in test environment"
        )
    }
    
    /// Requires Core ML models to be available
    public static var requiresCoreML: Self {
        EnvironmentTrait(
            isEnabled: TestEnvironment.hasCoreMLModels,
            skipMessage: "Core ML test models not available"
        )
    }
    
    /// Requires high memory for performance tests
    public static var requiresHighMemory: Self {
        EnvironmentTrait(
            isEnabled: TestEnvironment.hasHighMemory,
            skipMessage: "Insufficient memory for performance tests"
        )
    }
    
    /// Skip in CI environments
    public static var skipInCI: Self {
        EnvironmentTrait(
            isEnabled: !TestEnvironment.isCI,
            skipMessage: "Test skipped in CI environment"
        )
    }
}

// MARK: - Metal Test Helpers

/// Test environment errors
public enum TestEnvironmentError: Error, CustomStringConvertible {
    case metalNotSupported
    case metalDeviceCreationFailed
    
    public var description: String {
        switch self {
        case .metalNotSupported:
            return "Metal not supported in test environment"
        case .metalDeviceCreationFailed:
            return "Failed to create Metal device"
        }
    }
}

/// Helper to get Metal accelerator for tests with proper error handling
public func getTestMetalAccelerator() throws -> MetalAccelerator {
    guard TestEnvironment.hasMetalSupport else {
        throw TestEnvironmentError.metalNotSupported
    }
    
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw TestEnvironmentError.metalDeviceCreationFailed
    }
    
    return try MetalAccelerator(device: device)
}

/// Mock Metal accelerator for testing when Metal is not available
public actor MockMetalAccelerator: MetalAcceleratorProtocol {
    public let isAvailable: Bool = true
    
    public init() {}
    
    public func normalizeVectors(_ vectors: [[Float]]) async throws -> [[Float]] {
        // CPU implementation for testing
        return vectors.map { vector in
            let norm = sqrt(vector.reduce(0) { $0 + $1 * $1 })
            return norm > 0 ? vector.map { $0 / norm } : vector
        }
    }
    
    public func poolEmbeddings(
        _ tokenEmbeddings: [[Float]],
        strategy: PoolingStrategy,
        attentionMask: [Int]? = nil
    ) async throws -> [Float] {
        guard !tokenEmbeddings.isEmpty else {
            throw MetalError.invalidInput("Empty token embeddings")
        }
        
        let dimensions = tokenEmbeddings[0].count
        
        switch strategy {
        case .cls:
            return tokenEmbeddings[0]
        case .mean:
            var result = [Float](repeating: 0, count: dimensions)
            var count = 0
            for (i, embedding) in tokenEmbeddings.enumerated() {
                if attentionMask == nil || attentionMask![i] == 1 {
                    for j in 0..<dimensions {
                        result[j] += embedding[j]
                    }
                    count += 1
                }
            }
            return count > 0 ? result.map { $0 / Float(count) } : result
        case .max:
            var result = [Float](repeating: -Float.infinity, count: dimensions)
            for (i, embedding) in tokenEmbeddings.enumerated() {
                if attentionMask == nil || attentionMask![i] == 1 {
                    for j in 0..<dimensions {
                        result[j] = max(result[j], embedding[j])
                    }
                }
            }
            return result
        case .attentionWeighted:
            // Use uniform weights for testing
            let uniformWeights = [Float](repeating: 1.0 / Float(tokenEmbeddings.count), count: tokenEmbeddings.count)
            return try await attentionWeightedPooling(tokenEmbeddings, attentionWeights: uniformWeights)
        }
    }
    
    public func attentionWeightedPooling(
        _ tokenEmbeddings: [[Float]],
        attentionWeights: [Float]
    ) async throws -> [Float] {
        guard tokenEmbeddings.count == attentionWeights.count else {
            throw MetalError.invalidInput("Attention weights count must match sequence length")
        }
        
        let dimensions = tokenEmbeddings[0].count
        var result = [Float](repeating: 0.0, count: dimensions)
        let weightSum = attentionWeights.reduce(0, +)
        
        guard weightSum > 0 else {
            throw MetalError.invalidInput("Attention weights sum to zero")
        }
        
        for i in 0..<tokenEmbeddings.count {
            let weight = attentionWeights[i] / weightSum
            for j in 0..<dimensions {
                result[j] += tokenEmbeddings[i][j] * weight
            }
        }
        
        return result
    }
    
    public func cosineSimilarity(_ vectorA: [Float], _ vectorB: [Float]) async throws -> Float {
        guard vectorA.count == vectorB.count else {
            throw MetalError.dimensionMismatch
        }
        
        var dotProduct: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        
        for i in 0..<vectorA.count {
            dotProduct += vectorA[i] * vectorB[i]
            normA += vectorA[i] * vectorA[i]
            normB += vectorB[i] * vectorB[i]
        }
        
        let denominator = sqrt(normA * normB)
        return denominator > 0 ? dotProduct / denominator : 0
    }
    
    public func cosineSimilarityBatch(_ vectorPairs: [([Float], [Float])]) async throws -> [Float] {
        var results: [Float] = []
        for (vectorA, vectorB) in vectorPairs {
            let similarity = try await cosineSimilarity(vectorA, vectorB)
            results.append(similarity)
        }
        return results
    }
    
    public func cosineSimilarityMatrix(queries: [[Float]], keys: [[Float]]) async throws -> [[Float]] {
        var results: [[Float]] = []
        for query in queries {
            var row: [Float] = []
            for key in keys {
                let similarity = try await cosineSimilarity(query, key)
                row.append(similarity)
            }
            results.append(row)
        }
        return results
    }
    
    public func fastBatchNormalize(_ vectors: [[Float]], epsilon: Float = 1e-6) async throws -> [[Float]] {
        // Simple batch normalization for testing
        return try await normalizeVectors(vectors)
    }
    
    public func cosineSimilarity(query: [Float], keys: [[Float]]) async throws -> [Float] {
        let matrix = try await cosineSimilarityMatrix(queries: [query], keys: keys)
        return matrix[0]
    }
    
    public func handleMemoryPressure() async {
        // No-op for mock
    }
}