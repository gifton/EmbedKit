import XCTest
import Metal
@testable import EmbedKit

final class NumericsSpecializationToggleTests: XCTestCase {
    var accelerator: MetalAccelerator!

    override func setUp() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal device not available")
        }

        accelerator = try XCTUnwrap(
            MetalAccelerator.shared,
            "MetalAccelerator.shared should be available"
        )
        try await accelerator.setupPipelines()
    }

    // MARK: - Fast vs Stable: Basic Mixed Magnitude Vector

    func testFastPathNormalization() async throws {
        // Ensure fast single-pass path
        await NumericsTestHelper.disableStableNormalization(accelerator, epsilon: 1e-8)

        let v: [Float] = [1e-10, 1e10, 1.0, 1e-5, 1e5, 3.0]
        let batch = try VectorBatch(vectors: [v])
        let result = try await accelerator.normalizeVectors(batch)
        let out = Array(result[0])

        // Finite and near unit norm
        MetalTestUtilities.assertFinite(out, "fast path mixed magnitudes")
        let norm = sqrt(out.reduce(0) { $0 + $1 * $1 })
        XCTAssertEqual(norm, 1.0, accuracy: 1e-3)
    }

    func testStablePathMatchesCPU() async throws {
        // Enable stable two-pass path
        await NumericsTestHelper.enableStableNormalization(accelerator, epsilon: 1e-8)

        let v: [Float] = [1e-10, 1e10, 1.0, 1e-5, 1e5, 3.0]
        let batch = try VectorBatch(vectors: [v])
        let result = try await accelerator.normalizeVectors(batch)
        let out = Array(result[0])

        // Compare against CPU stable reference
        let cpu = MetalTestUtilities.cpuNormalize(v)
        MetalTestUtilities.assertEqual(out, cpu, accuracy: 1e-4, "stable vs cpu")
    }

    // MARK: - Epsilon Threshold Behavior (Stable Path)

    func testStablePathEpsilonThresholdZeroing() async throws {
        // Large epsilon should treat near-zero magnitudes as zero
        await NumericsTestHelper.enableStableNormalization(accelerator, epsilon: 1e-4)

        // All values well below epsilon => zero result
        let tiny: [Float] = Array(repeating: 1e-6, count: 128)
        let batch = try VectorBatch(vectors: [tiny])
        let result = try await accelerator.normalizeVectors(batch)
        let out = Array(result[0])

        XCTAssertTrue(out.allSatisfy { $0 == 0.0 }, "stable+large epsilon should zero near-zero vector")
    }

    // MARK: - Batch-Optimized Kernel Parity (Stable)

    func testBatchOptimizedStableParityAcrossBoundaries() async throws {
        await NumericsTestHelper.enableStableNormalization(accelerator, epsilon: 1e-8)

        // Dimensions straddling small/medium/large boundaries
        let dims = [31, 32, 33, 63, 64, 65]
        for dim in dims {
            let vectors = makeDeterministicVectors(count: 8, dim: dim)
            let batch = try VectorBatch(vectors: vectors)
            let gpu = try await accelerator.normalizeVectors(batch)

            // Compare first few elements for each vector to CPU reference
            for i in 0..<vectors.count {
                let cpu = MetalTestUtilities.cpuNormalize(vectors[i])
                let out = Array(gpu[i])
                MetalTestUtilities.assertEqual(out, cpu, accuracy: 1e-3, "stable batch parity dim=\(dim) idx=\(i)")
            }
        }
    }

    // MARK: - Helpers

    private func makeDeterministicVectors(count: Int, dim: Int) -> [[Float]] {
        var result: [[Float]] = []
        result.reserveCapacity(count)
        for i in 0..<count {
            var v: [Float] = []
            v.reserveCapacity(dim)
            for j in 0..<dim {
                // Alternating extreme magnitudes with a deterministic pattern
                let base: Float = (j % 2 == 0) ? 1e-8 : 1e8
                let sign: Float = ((i + j) % 3 == 0) ? -1.0 : 1.0
                v.append(base * sign)
            }
            result.append(v)
        }
        return result
    }
}

