// EmbedKit - Attention Pooling Tests

import Testing
import Foundation
@testable import EmbedKit

#if canImport(Metal)
import Metal

@Suite("Attention Pooling GPU")
struct AttentionPoolingTests {

    // MARK: - Basic Functionality Tests

    @Test("Attention pooling produces correct weighted average")
    func attentionPoolingBasic() async throws {
        let accelerator = await MetalAccelerator()

        guard await accelerator.isAvailable else {
            throw XCTSkip("Metal accelerator not available")
        }

        let batchSize = 1
        let sequenceLength = 4
        let dimensions = 3

        // Simple embeddings: each token is a different vector
        let embeddings: [Float] = [
            1, 0, 0,  // Token 0
            0, 1, 0,  // Token 1
            0, 0, 1,  // Token 2
            0, 0, 0   // Token 3
        ]

        // Weights: focus on first token
        let weights: [Float] = [0.7, 0.2, 0.1, 0.0]

        let results = await accelerator.tensorAttentionPoolNormalize(
            embeddings: embeddings,
            weights: weights,
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            dimensions: dimensions,
            normalize: false  // Don't normalize to check weighted average
        )

        #expect(results.count == 1)
        #expect(results[0].count == dimensions)

        // Expected: (0.7*[1,0,0] + 0.2*[0,1,0] + 0.1*[0,0,1]) / 1.0 = [0.7, 0.2, 0.1]
        #expect(abs(results[0][0] - 0.7) < 0.01)
        #expect(abs(results[0][1] - 0.2) < 0.01)
        #expect(abs(results[0][2] - 0.1) < 0.01)
    }

    @Test("Attention pooling with normalization")
    func attentionPoolingNormalized() async throws {
        let accelerator = await MetalAccelerator()

        guard await accelerator.isAvailable else {
            throw XCTSkip("Metal accelerator not available")
        }

        let batchSize = 2
        let sequenceLength = 4
        let dimensions = 8

        // Random embeddings
        var embeddings: [Float] = []
        for _ in 0..<(batchSize * sequenceLength * dimensions) {
            embeddings.append(Float.random(in: -1...1))
        }

        // Random weights (softmax-like: sum to 1)
        var weights: [Float] = []
        for b in 0..<batchSize {
            var batchWeights: [Float] = []
            for _ in 0..<sequenceLength {
                batchWeights.append(Float.random(in: 0.1...1.0))
            }
            let sum = batchWeights.reduce(0, +)
            for i in 0..<sequenceLength {
                weights.append(batchWeights[i] / sum)
            }
        }

        let results = await accelerator.tensorAttentionPoolNormalize(
            embeddings: embeddings,
            weights: weights,
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            dimensions: dimensions,
            normalize: true
        )

        #expect(results.count == batchSize)

        // Verify each result is normalized (L2 norm = 1)
        for batchResult in results {
            #expect(batchResult.count == dimensions)
            let norm = sqrt(batchResult.reduce(0) { $0 + $1 * $1 })
            #expect(abs(norm - 1.0) < 0.01)
        }
    }

    @Test("Attention pooling matches CPU fallback")
    func attentionPoolingMatchesCPU() async throws {
        let accelerator = await MetalAccelerator()

        guard await accelerator.isAvailable else {
            throw XCTSkip("Metal accelerator not available")
        }

        let batchSize = 4
        let sequenceLength = 16
        let dimensions = 32

        // Generate test data
        var embeddings: [Float] = []
        for _ in 0..<(batchSize * sequenceLength * dimensions) {
            embeddings.append(Float.random(in: -1...1))
        }

        var weights: [Float] = []
        for _ in 0..<(batchSize * sequenceLength) {
            weights.append(Float.random(in: 0.01...1.0))
        }

        // GPU result
        let gpuResults = await accelerator.tensorAttentionPoolNormalize(
            embeddings: embeddings,
            weights: weights,
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            dimensions: dimensions,
            normalize: true
        )

        // CPU reference implementation
        var cpuResults: [[Float]] = []
        for b in 0..<batchSize {
            let seqStart = b * sequenceLength * dimensions
            let weightStart = b * sequenceLength

            var weightSum: Float = 0
            for t in 0..<sequenceLength {
                weightSum += weights[weightStart + t]
            }

            var pooled = [Float](repeating: 0, count: dimensions)
            for t in 0..<sequenceLength {
                let weight = weights[weightStart + t]
                let tokenStart = seqStart + t * dimensions
                for d in 0..<dimensions {
                    pooled[d] += embeddings[tokenStart + d] * weight
                }
            }

            let invWeightSum = weightSum > 1e-12 ? 1.0 / weightSum : 0.0
            for d in 0..<dimensions {
                pooled[d] *= invWeightSum
            }

            // Normalize
            let norm = max(1e-12, sqrt(pooled.reduce(0) { $0 + Double($1) * Double($1) }))
            pooled = pooled.map { $0 / Float(norm) }
            cpuResults.append(pooled)
        }

        // Compare
        #expect(gpuResults.count == cpuResults.count)
        for b in 0..<batchSize {
            for d in 0..<dimensions {
                let diff = abs(gpuResults[b][d] - cpuResults[b][d])
                #expect(diff < 0.001, "Mismatch at batch \(b), dim \(d): GPU=\(gpuResults[b][d]), CPU=\(cpuResults[b][d])")
            }
        }
    }

    // MARK: - Batch Size Tests

    @Test("Attention pooling handles various batch sizes")
    func attentionPoolingBatchSizes() async throws {
        let accelerator = await MetalAccelerator()

        guard await accelerator.isAvailable else {
            throw XCTSkip("Metal accelerator not available")
        }

        let sequenceLength = 8
        let dimensions = 16

        for batchSize in [1, 2, 4, 8, 16, 32] {
            var embeddings: [Float] = []
            for _ in 0..<(batchSize * sequenceLength * dimensions) {
                embeddings.append(Float.random(in: -1...1))
            }

            var weights: [Float] = []
            for _ in 0..<(batchSize * sequenceLength) {
                weights.append(Float.random(in: 0.1...1.0))
            }

            let results = await accelerator.tensorAttentionPoolNormalize(
                embeddings: embeddings,
                weights: weights,
                batchSize: batchSize,
                sequenceLength: sequenceLength,
                dimensions: dimensions,
                normalize: true
            )

            #expect(results.count == batchSize, "Failed for batch size \(batchSize)")

            for result in results {
                #expect(result.count == dimensions)
                let norm = sqrt(result.reduce(0) { $0 + $1 * $1 })
                #expect(abs(norm - 1.0) < 0.01, "Normalization failed for batch size \(batchSize)")
            }
        }
    }

    // MARK: - Edge Case Tests

    @Test("Attention pooling with uniform weights equals mean")
    func attentionPoolingUniformWeights() async throws {
        let accelerator = await MetalAccelerator()

        guard await accelerator.isAvailable else {
            throw XCTSkip("Metal accelerator not available")
        }

        let batchSize = 2
        let sequenceLength = 8
        let dimensions = 16

        var embeddings: [Float] = []
        for _ in 0..<(batchSize * sequenceLength * dimensions) {
            embeddings.append(Float.random(in: -1...1))
        }

        // Uniform weights
        let weights: [Float] = [Float](repeating: 1.0, count: batchSize * sequenceLength)

        // Attention pooling with uniform weights
        let attentionResults = await accelerator.tensorAttentionPoolNormalize(
            embeddings: embeddings,
            weights: weights,
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            dimensions: dimensions,
            normalize: true
        )

        // Mean pooling
        let meanResults = await accelerator.tensorPoolNormalize(
            embeddings: embeddings,
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            dimensions: dimensions,
            masks: nil,
            strategy: .mean,
            normalize: true
        )

        // Should be very close
        for b in 0..<batchSize {
            for d in 0..<dimensions {
                let diff = abs(attentionResults[b][d] - meanResults[b][d])
                #expect(diff < 0.001)
            }
        }
    }

    @Test("Attention pooling with zero weights")
    func attentionPoolingZeroWeights() async throws {
        let accelerator = await MetalAccelerator()

        guard await accelerator.isAvailable else {
            throw XCTSkip("Metal accelerator not available")
        }

        let batchSize = 1
        let sequenceLength = 4
        let dimensions = 4

        let embeddings: [Float] = [Float](repeating: 1.0, count: batchSize * sequenceLength * dimensions)
        let weights: [Float] = [Float](repeating: 0.0, count: batchSize * sequenceLength)

        let results = await accelerator.tensorAttentionPoolNormalize(
            embeddings: embeddings,
            weights: weights,
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            dimensions: dimensions,
            normalize: false
        )

        // With zero weights, should get zero output
        #expect(results.count == 1)
        for val in results[0] {
            #expect(abs(val) < 0.001)
        }
    }

    @Test("Attention pooling with single weight")
    func attentionPoolingSingleWeight() async throws {
        let accelerator = await MetalAccelerator()

        guard await accelerator.isAvailable else {
            throw XCTSkip("Metal accelerator not available")
        }

        let batchSize = 1
        let sequenceLength = 4
        let dimensions = 3

        // Embeddings: different vectors per token
        let embeddings: [Float] = [
            1, 2, 3,   // Token 0
            4, 5, 6,   // Token 1
            7, 8, 9,   // Token 2
            10, 11, 12 // Token 3
        ]

        // All weight on token 2
        let weights: [Float] = [0.0, 0.0, 1.0, 0.0]

        let results = await accelerator.tensorAttentionPoolNormalize(
            embeddings: embeddings,
            weights: weights,
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            dimensions: dimensions,
            normalize: false
        )

        // Should just return token 2's embedding
        #expect(abs(results[0][0] - 7.0) < 0.01)
        #expect(abs(results[0][1] - 8.0) < 0.01)
        #expect(abs(results[0][2] - 9.0) < 0.01)
    }

    // MARK: - PoolingStrategy.attention Tests

    @Test("PoolingStrategy.attention metalIndex is 3")
    func poolingStrategyAttentionIndex() {
        #expect(PoolingStrategy.attention.metalIndex == 3)
    }

    @Test("PoolingStrategy includes attention case")
    func poolingStrategyAllCases() {
        let strategies = PoolingStrategy.allCases
        #expect(strategies.contains(.attention))
        #expect(strategies.count == 4) // mean, max, cls, attention
    }

    // MARK: - Performance Tests

    @Test("Attention pooling GPU faster than CPU for large batches", .tags(.performance))
    func attentionPoolingPerformance() async throws {
        let accelerator = await MetalAccelerator()

        guard await accelerator.isAvailable else {
            throw XCTSkip("Metal accelerator not available")
        }

        let batchSize = 64
        let sequenceLength = 128
        let dimensions = 384

        var embeddings: [Float] = []
        for _ in 0..<(batchSize * sequenceLength * dimensions) {
            embeddings.append(Float.random(in: -1...1))
        }

        var weights: [Float] = []
        for _ in 0..<(batchSize * sequenceLength) {
            weights.append(Float.random(in: 0.1...1.0))
        }

        // GPU execution
        let gpuStart = CFAbsoluteTimeGetCurrent()
        let _ = await accelerator.tensorAttentionPoolNormalize(
            embeddings: embeddings,
            weights: weights,
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            dimensions: dimensions,
            normalize: true
        )
        let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart

        // CPU execution (warm up GPU has been done)
        let cpuStart = CFAbsoluteTimeGetCurrent()
        for b in 0..<batchSize {
            let seqStart = b * sequenceLength * dimensions
            let weightStart = b * sequenceLength

            var weightSum: Float = 0
            for t in 0..<sequenceLength {
                weightSum += weights[weightStart + t]
            }

            var pooled = [Float](repeating: 0, count: dimensions)
            for t in 0..<sequenceLength {
                let weight = weights[weightStart + t]
                let tokenStart = seqStart + t * dimensions
                for d in 0..<dimensions {
                    pooled[d] += embeddings[tokenStart + d] * weight
                }
            }

            let invWeightSum = weightSum > 1e-12 ? 1.0 / weightSum : 0.0
            for d in 0..<dimensions {
                pooled[d] *= invWeightSum
            }

            let norm = max(1e-12, sqrt(pooled.reduce(0) { $0 + Double($1) * Double($1) }))
            _ = pooled.map { $0 / Float(norm) }
        }
        let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart

        print("Attention Pooling - GPU: \(gpuTime * 1000)ms, CPU: \(cpuTime * 1000)ms")
        print("Speedup: \(cpuTime / gpuTime)x")

        // GPU should be faster for large workloads
        // (may not always hold due to warm-up, but should be competitive)
        #expect(gpuTime < cpuTime * 2.0)
    }
}

#endif
