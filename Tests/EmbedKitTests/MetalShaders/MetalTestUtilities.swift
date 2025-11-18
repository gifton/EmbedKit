import XCTest
import Metal
@testable import EmbedKit

/// Utilities for Metal shader testing
///
/// This module provides helper functions for:
/// - Performance measurement and statistical analysis
/// - CPU reference implementations for validation
/// - Test data generation
/// - Accuracy assertions with appropriate tolerances
///
public enum MetalTestUtilities {

    // MARK: - Performance Measurement

    /// Performance statistics from multiple measurement iterations
    public struct PerformanceStats {
        public let times: [Double]  // milliseconds

        public var median: Double {
            let sorted = times.sorted()
            let count = sorted.count
            if count % 2 == 0 {
                return (sorted[count/2 - 1] + sorted[count/2]) / 2.0
            } else {
                return sorted[count/2]
            }
        }

        public var mean: Double {
            times.reduce(0, +) / Double(times.count)
        }

        public var min: Double {
            times.min() ?? 0
        }

        public var max: Double {
            times.max() ?? 0
        }

        public var stddev: Double {
            let m = mean
            let variance = times.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(times.count)
            return sqrt(variance)
        }

        public var p95: Double {
            let sorted = times.sorted()
            let index = Int(ceil(Double(sorted.count) * 0.95)) - 1
            return sorted[Swift.max(0, Swift.min(index, sorted.count - 1))]
        }

        /// Pretty print statistics
        public var description: String {
            """
            Performance Stats:
              Median: \(String(format: "%.2f", median))ms
              Mean:   \(String(format: "%.2f", mean))ms (±\(String(format: "%.2f", stddev)))
              Range:  [\(String(format: "%.2f", min)) - \(String(format: "%.2f", max))]ms
              P95:    \(String(format: "%.2f", p95))ms
            """
        }
    }

    /// Measure execution time of async operation
    ///
    /// - Parameters:
    ///   - warmup: Number of warm-up iterations (default: 1)
    ///   - iterations: Number of measurement iterations
    ///   - operation: Async operation to measure
    /// - Returns: Performance statistics
    ///
    public static func measure(
        warmup: Int = 1,
        iterations: Int,
        operation: () async throws -> Void
    ) async rethrows -> PerformanceStats {
        var times: [Double] = []

        // Warm-up iterations
        for _ in 0..<warmup {
            try await operation()
        }

        // Measurement iterations
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            try await operation()
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000  // Convert to ms
            times.append(elapsed)
        }

        return PerformanceStats(times: times)
    }

    // MARK: - CPU Reference Implementations

    /// CPU reference implementation of L2 normalization
    ///
    /// Algorithm: output[i] = input[i] / ||input||₂
    /// where ||input||₂ = √(Σ input[i]²)
    ///
    /// - Parameter vector: Input vector to normalize
    /// - Returns: Normalized vector with unit L2 norm
    ///
    public static func cpuNormalize(_ vector: [Float]) -> [Float] {
        // Compute L2 norm using numerically stable method
        let maxVal = vector.map { abs($0) }.max() ?? 0

        guard maxVal > 1e-8 else {
            // Zero vector
            return Array(repeating: 0, count: vector.count)
        }

        // Scale by max to prevent overflow
        let scale = 1.0 / maxVal
        var sumSquares: Float = 0
        for val in vector {
            let scaled = val * scale
            sumSquares += scaled * scaled
        }

        let norm = maxVal * sqrt(sumSquares)
        let invNorm = 1.0 / norm

        return vector.map { $0 * invNorm }
    }

    /// CPU reference implementation of mean pooling
    ///
    /// Algorithm: output[d] = Σ(embeddings[t][d] * mask[t]) / Σ(mask[t])
    ///
    /// - Parameters:
    ///   - embeddings: Token embeddings [sequenceLength, dimensions]
    ///   - mask: Optional attention mask [sequenceLength]
    /// - Returns: Pooled embedding [dimensions]
    ///
    public static func cpuMeanPool(_ embeddings: [[Float]], mask: [Int]?) -> [Float] {
        guard !embeddings.isEmpty else {
            return []
        }

        let dimensions = embeddings[0].count
        var result = Array(repeating: Float(0), count: dimensions)
        var count = 0

        for (idx, embedding) in embeddings.enumerated() {
            let isValid = mask?[idx] ?? 1
            if isValid == 1 {
                for d in 0..<dimensions {
                    result[d] += embedding[d]
                }
                count += 1
            }
        }

        guard count > 0 else {
            return result  // Return zeros if all masked
        }

        let scale = 1.0 / Float(count)
        return result.map { $0 * scale }
    }

    /// CPU reference implementation of max pooling
    ///
    /// Algorithm: output[d] = max(embeddings[t][d]) over valid tokens
    ///
    /// - Parameters:
    ///   - embeddings: Token embeddings [sequenceLength, dimensions]
    ///   - mask: Optional attention mask [sequenceLength]
    /// - Returns: Pooled embedding [dimensions]
    ///
    public static func cpuMaxPool(_ embeddings: [[Float]], mask: [Int]?) -> [Float] {
        guard !embeddings.isEmpty else {
            return []
        }

        let dimensions = embeddings[0].count
        var result = Array(repeating: -Float.infinity, count: dimensions)
        var foundValid = false

        for (idx, embedding) in embeddings.enumerated() {
            let isValid = mask?[idx] ?? 1
            if isValid == 1 {
                for d in 0..<dimensions {
                    result[d] = max(result[d], embedding[d])
                }
                foundValid = true
            }
        }

        if !foundValid {
            return Array(repeating: 0, count: dimensions)
        }

        return result
    }

    /// CPU reference implementation of attention-weighted pooling
    ///
    /// Algorithm: output[d] = Σ(embeddings[t][d] * weights[t]) / Σ(weights[t])
    ///
    /// - Parameters:
    ///   - embeddings: Token embeddings [sequenceLength, dimensions]
    ///   - weights: Attention weights [sequenceLength]
    /// - Returns: Pooled embedding [dimensions]
    ///
    public static func cpuAttentionPool(_ embeddings: [[Float]], weights: [Float]) -> [Float] {
        guard !embeddings.isEmpty else {
            return []
        }

        let dimensions = embeddings[0].count
        var result = Array(repeating: Float(0), count: dimensions)
        var weightSum: Float = 0

        for (idx, embedding) in embeddings.enumerated() {
            let weight = weights[idx]
            weightSum += weight

            for d in 0..<dimensions {
                result[d] += embedding[d] * weight
            }
        }

        guard weightSum > 1e-6 else {
            return result  // Return zeros if weights sum to zero
        }

        let scale = 1.0 / weightSum
        return result.map { $0 * scale }
    }

    /// CPU reference implementation of cosine similarity
    ///
    /// Algorithm: similarity = dot(v1, v2) / (||v1||₂ * ||v2||₂)
    ///
    /// - Parameters:
    ///   - v1: First vector
    ///   - v2: Second vector
    /// - Returns: Cosine similarity in [-1, 1]
    ///
    public static func cpuCosineSimilarity(_ v1: [Float], _ v2: [Float]) -> Float {
        assert(v1.count == v2.count, "Vectors must have same dimension")

        var dotProduct: Float = 0
        var norm1: Float = 0
        var norm2: Float = 0

        for i in 0..<v1.count {
            dotProduct += v1[i] * v2[i]
            norm1 += v1[i] * v1[i]
            norm2 += v2[i] * v2[i]
        }

        let normProduct = sqrt(norm1 * norm2)
        guard normProduct > 1e-8 else {
            return 0  // Handle zero vectors
        }

        let similarity = dotProduct / normProduct

        // Clamp to valid range to handle numerical errors
        return max(-1.0, min(1.0, similarity))
    }

    // MARK: - Test Data Generation

    /// Generate random vector with specified characteristics
    ///
    /// - Parameters:
    ///   - dimensions: Vector dimensions
    ///   - range: Value range (default: -1...1)
    /// - Returns: Random vector
    ///
    public static func randomVector(dimensions: Int, range: ClosedRange<Float> = -1...1) -> [Float] {
        return (0..<dimensions).map { _ in Float.random(in: range) }
    }

    /// Generate batch of random vectors
    ///
    /// - Parameters:
    ///   - batchSize: Number of vectors
    ///   - dimensions: Vector dimensions
    ///   - range: Value range (default: -1...1)
    /// - Returns: Batch of random vectors
    ///
    public static func randomBatch(batchSize: Int, dimensions: Int, range: ClosedRange<Float> = -1...1) -> [[Float]] {
        return (0..<batchSize).map { _ in randomVector(dimensions: dimensions, range: range) }
    }

    /// Generate unit vector (for testing normalized operations)
    ///
    /// - Parameter dimensions: Vector dimensions
    /// - Returns: Random unit vector (L2 norm = 1)
    ///
    public static func randomUnitVector(dimensions: Int) -> [Float] {
        let random = randomVector(dimensions: dimensions)
        return cpuNormalize(random)
    }

    /// Generate orthogonal vectors (for testing cosine similarity = 0)
    ///
    /// - Parameter dimensions: Vector dimensions (must be >= 2)
    /// - Returns: Tuple of two orthogonal unit vectors
    ///
    public static func orthogonalVectors(dimensions: Int) -> (v1: [Float], v2: [Float]) {
        assert(dimensions >= 2, "Need at least 2 dimensions for orthogonal vectors")

        var v1 = Array(repeating: Float(0), count: dimensions)
        var v2 = Array(repeating: Float(0), count: dimensions)

        v1[0] = 1.0
        v2[1] = 1.0

        return (v1, v2)
    }

    // MARK: - Assertion Helpers

    /// Assert two float arrays are approximately equal
    ///
    /// - Parameters:
    ///   - actual: Actual values
    ///   - expected: Expected values
    ///   - accuracy: Absolute tolerance
    ///   - message: Test failure message
    ///   - file: Source file
    ///   - line: Source line
    ///
    public static func assertEqual(
        _ actual: [Float],
        _ expected: [Float],
        accuracy: Float = 1e-5,
        _ message: String = "",
        file: StaticString = #file,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count,
                      "Array length mismatch: \(actual.count) vs \(expected.count). \(message)",
                      file: file, line: line)

        for (idx, (a, e)) in zip(actual, expected).enumerated() {
            XCTAssertEqual(a, e, accuracy: accuracy,
                          "Mismatch at index \(idx): \(a) vs \(e). \(message)",
                          file: file, line: line)
        }
    }

    public static func assertEqual(
        _ actual: ArraySlice<Float>,
        _ expected: [Float],
        accuracy: Float = 1e-5,
        _ message: String = "",
        file: StaticString = #file,
        line: UInt = #line
    ) {
        assertEqual(Array(actual), expected, accuracy: accuracy, message, file: file, line: line)
    }

    /// Assert vector has unit L2 norm
    ///
    /// - Parameters:
    ///   - vector: Vector to check
    ///   - accuracy: Absolute tolerance (default: 1e-4)
    ///   - message: Test failure message
    ///   - file: Source file
    ///   - line: Source line
    ///
    public static func assertUnitNorm(
        _ vector: [Float],
        accuracy: Float = 1e-4,
        _ message: String = "",
        file: StaticString = #file,
        line: UInt = #line
    ) {
        let norm = sqrt(vector.reduce(0) { $0 + $1 * $1 })
        XCTAssertEqual(norm, 1.0, accuracy: accuracy,
                      "Vector norm should be 1.0, got \(norm). \(message)",
                      file: file, line: line)
    }

    public static func assertUnitNorm(
        _ vector: ArraySlice<Float>,
        accuracy: Float = 1e-4,
        _ message: String = "",
        file: StaticString = #file,
        line: UInt = #line
    ) {
        assertUnitNorm(Array(vector), accuracy: accuracy, message, file: file, line: line)
    }

    /// Assert all values are finite (not NaN or Inf)
    ///
    /// - Parameters:
    ///   - values: Values to check
    ///   - message: Test failure message
    ///   - file: Source file
    ///   - line: Source line
    ///
    public static func assertFinite(
        _ values: [Float],
        _ message: String = "",
        file: StaticString = #file,
        line: UInt = #line
    ) {
        for (idx, val) in values.enumerated() {
            XCTAssertTrue(val.isFinite,
                         "Value at index \(idx) is not finite: \(val). \(message)",
                         file: file, line: line)
        }
    }

    /// Assert value is in valid range
    ///
    /// - Parameters:
    ///   - value: Value to check
    ///   - range: Valid range
    ///   - message: Test failure message
    ///   - file: Source file
    ///   - line: Source line
    ///
    public static func assertInRange(
        _ value: Float,
        _ range: ClosedRange<Float>,
        _ message: String = "",
        file: StaticString = #file,
        line: UInt = #line
    ) {
        XCTAssertGreaterThanOrEqual(value, range.lowerBound,
                                   "Value \(value) below range [\(range.lowerBound), \(range.upperBound)]. \(message)",
                                   file: file, line: line)
        XCTAssertLessThanOrEqual(value, range.upperBound,
                                "Value \(value) above range [\(range.lowerBound), \(range.upperBound)]. \(message)",
                                file: file, line: line)
    }

    // MARK: - Device Information

    /// Get information about the Metal device being tested
    ///
    /// - Returns: Device information string
    ///
    public static func deviceInfo() -> String? {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return nil
        }

        var info = "Metal Device: \(device.name)\n"

        #if arch(arm64)
        info += "Architecture: Apple Silicon\n"
        #else
        info += "Architecture: Intel/AMD\n"
        #endif

        if #available(macOS 13.0, iOS 16.0, *) {
            if device.supportsFamily(.metal3) {
                info += "Metal 3: Supported\n"
            } else {
                info += "Metal 3: Not Supported\n"
            }
        }

        info += "Max threads per threadgroup: \(device.maxThreadsPerThreadgroup)\n"
        info += "Recommended max working set size: \(device.recommendedMaxWorkingSetSize / 1024 / 1024) MB"

        return info
    }
}

// MARK: - Extensions for Convenience

extension Array where Element == Float {
    /// Compute L2 norm of vector
    var l2Norm: Float {
        sqrt(self.reduce(0) { $0 + $1 * $1 })
    }

    /// Check if all values are finite
    var allFinite: Bool {
        self.allSatisfy { $0.isFinite }
    }
}

extension ArraySlice where Element == Float {
    /// Compute L2 norm of vector
    var l2Norm: Float {
        sqrt(self.reduce(0) { $0 + $1 * $1 })
    }

    /// Check if all values are finite
    var allFinite: Bool {
        self.allSatisfy { $0.isFinite }
    }
}
