import XCTest
@testable import EmbedKit

/// Tests GPU vs CPU parity for pooling operations (mean, max)
final class MetalPoolingParityTests: XCTestCase {

    // MARK: - Test Helpers

    /// Deterministic pseudo-random number generator for reproducible tests
    private struct PRNG {
        var seed: UInt64

        mutating func next() -> Float {
            seed = seed &* 2862933555777941757 &+ 3037000493
            let x = Double((seed >> 33) & 0xFFFF_FFFF)
            return Float((x / Double(UInt32.max)) * 2.0 - 1.0)
        }

        mutating func nextPositive() -> Float {
            return abs(next()) + 0.001  // Ensure positive, non-zero
        }
    }

    /// Generate random flat embeddings array
    private func generateEmbeddings(sequenceLength: Int, dimensions: Int, seed: UInt64 = 0xDEADBEEF) -> [Float] {
        var rng = PRNG(seed: seed)
        return (0..<(sequenceLength * dimensions)).map { _ in rng.next() }
    }

    /// CPU mean pooling reference implementation
    private func cpuMeanPool(embeddings: [Float], sequenceLength: Int, dimensions: Int, mask: [Int]?) -> [Float] {
        var result = [Float](repeating: 0, count: dimensions)
        var count = 0
        for t in 0..<sequenceLength {
            let isValid = mask == nil || (mask![t] == 1)
            if isValid {
                for d in 0..<dimensions {
                    result[d] += embeddings[t * dimensions + d]
                }
                count += 1
            }
        }
        if count > 0 {
            let scale = 1.0 / Float(count)
            for d in 0..<dimensions {
                result[d] *= scale
            }
        }
        return result
    }

    /// CPU max pooling reference implementation
    private func cpuMaxPool(embeddings: [Float], sequenceLength: Int, dimensions: Int, mask: [Int]?) -> [Float] {
        var result = [Float](repeating: -.greatestFiniteMagnitude, count: dimensions)
        var foundValid = false
        for t in 0..<sequenceLength {
            let isValid = mask == nil || (mask![t] == 1)
            if isValid {
                foundValid = true
                for d in 0..<dimensions {
                    let val = embeddings[t * dimensions + d]
                    if val > result[d] {
                        result[d] = val
                    }
                }
            }
        }
        if !foundValid {
            return [Float](repeating: 0, count: dimensions)
        }
        return result
    }

    /// Assert two vectors are approximately equal
    private func assertVectorsEqual(_ a: [Float], _ b: [Float], tolerance: Float = 1e-4, file: StaticString = #file, line: UInt = #line) {
        XCTAssertEqual(a.count, b.count, "Vector lengths differ", file: file, line: line)
        for i in 0..<a.count {
            if abs(a[i] - b[i]) > tolerance {
                XCTFail("Mismatch at index \(i): GPU=\(a[i]) vs CPU=\(b[i]), diff=\(abs(a[i] - b[i]))", file: file, line: line)
                return
            }
        }
    }

    // MARK: - Mean Pooling Tests

    func testMeanPoolingParitySmall() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        let seqLen = 16, dims = 32
        let embeddings = generateEmbeddings(sequenceLength: seqLen, dimensions: dims)

        let cpuResult = cpuMeanPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: nil)
        let gpuResult = await acc.meanPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: nil)

        assertVectorsEqual(gpuResult, cpuResult)
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }

    func testMeanPoolingParityLarge() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        // Large enough to trigger GPU path (seqLen * dims >= 1024)
        let seqLen = 128, dims = 384
        let embeddings = generateEmbeddings(sequenceLength: seqLen, dimensions: dims, seed: 0xCAFEBABE)

        let cpuResult = cpuMeanPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: nil)
        let gpuResult = await acc.meanPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: nil)

        assertVectorsEqual(gpuResult, cpuResult)
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }

    func testMeanPoolingWithMask() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        let seqLen = 64, dims = 128
        let embeddings = generateEmbeddings(sequenceLength: seqLen, dimensions: dims, seed: 0x12345678)

        // Mask: first half valid, second half masked (simulating padding)
        let mask = (0..<seqLen).map { $0 < seqLen / 2 ? 1 : 0 }

        let cpuResult = cpuMeanPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: mask)
        let gpuResult = await acc.meanPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: mask)

        assertVectorsEqual(gpuResult, cpuResult)
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }

    func testMeanPoolingWithSparseMask() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        let seqLen = 32, dims = 64
        let embeddings = generateEmbeddings(sequenceLength: seqLen, dimensions: dims, seed: 0xABCDEF00)

        // Sparse mask: every other token valid
        let mask = (0..<seqLen).map { $0 % 2 == 0 ? 1 : 0 }

        let cpuResult = cpuMeanPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: mask)
        let gpuResult = await acc.meanPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: mask)

        assertVectorsEqual(gpuResult, cpuResult)
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }

    // MARK: - Max Pooling Tests

    func testMaxPoolingParitySmall() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        let seqLen = 16, dims = 32
        let embeddings = generateEmbeddings(sequenceLength: seqLen, dimensions: dims)

        let cpuResult = cpuMaxPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: nil)
        let gpuResult = await acc.maxPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: nil)

        assertVectorsEqual(gpuResult, cpuResult)
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }

    func testMaxPoolingParityLarge() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        // Large enough to trigger GPU path
        let seqLen = 128, dims = 384
        let embeddings = generateEmbeddings(sequenceLength: seqLen, dimensions: dims, seed: 0xFEEDFACE)

        let cpuResult = cpuMaxPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: nil)
        let gpuResult = await acc.maxPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: nil)

        assertVectorsEqual(gpuResult, cpuResult)
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }

    func testMaxPoolingWithMask() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        let seqLen = 64, dims = 128
        let embeddings = generateEmbeddings(sequenceLength: seqLen, dimensions: dims, seed: 0x87654321)

        // Mask: first 3/4 valid, last 1/4 masked
        let mask = (0..<seqLen).map { $0 < (seqLen * 3 / 4) ? 1 : 0 }

        let cpuResult = cpuMaxPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: mask)
        let gpuResult = await acc.maxPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: mask)

        assertVectorsEqual(gpuResult, cpuResult)
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }

    // MARK: - Edge Cases

    func testPoolingSingleToken() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        let seqLen = 1, dims = 64
        let embeddings = generateEmbeddings(sequenceLength: seqLen, dimensions: dims, seed: 0x11111111)

        // For single token, mean and max should return the same result
        let cpuMean = cpuMeanPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: nil)
        let gpuMean = await acc.meanPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: nil)
        assertVectorsEqual(gpuMean, cpuMean)

        let cpuMax = cpuMaxPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: nil)
        let gpuMax = await acc.maxPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: nil)
        assertVectorsEqual(gpuMax, cpuMax)

        // Single token: mean == max == input
        assertVectorsEqual(gpuMean, embeddings)
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }

    func testPoolingAllMasked() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        let seqLen = 16, dims = 32
        let embeddings = generateEmbeddings(sequenceLength: seqLen, dimensions: dims, seed: 0x22222222)

        // All tokens masked
        let mask = [Int](repeating: 0, count: seqLen)

        let cpuMean = cpuMeanPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: mask)
        let gpuMean = await acc.meanPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: mask)

        // Both should return zeros when all masked
        let zeros = [Float](repeating: 0, count: dims)
        assertVectorsEqual(gpuMean, zeros)
        assertVectorsEqual(cpuMean, zeros)
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }

    func testPoolingHighDimensional() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        // Test with high dimensions (e.g., 1024-dim embeddings)
        let seqLen = 32, dims = 1024
        let embeddings = generateEmbeddings(sequenceLength: seqLen, dimensions: dims, seed: 0x33333333)

        let cpuMean = cpuMeanPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: nil)
        let gpuMean = await acc.meanPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: nil)
        assertVectorsEqual(gpuMean, cpuMean)

        let cpuMax = cpuMaxPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: nil)
        let gpuMax = await acc.maxPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: nil)
        assertVectorsEqual(gpuMax, cpuMax)
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }

    func testPoolingLongSequence() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        // Test with long sequence (e.g., 512 tokens)
        let seqLen = 512, dims = 384
        let embeddings = generateEmbeddings(sequenceLength: seqLen, dimensions: dims, seed: 0x44444444)

        let cpuMean = cpuMeanPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: nil)
        let gpuMean = await acc.meanPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: nil)
        assertVectorsEqual(gpuMean, cpuMean)

        let cpuMax = cpuMaxPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: nil)
        let gpuMax = await acc.maxPool(embeddings: embeddings, sequenceLength: seqLen, dimensions: dims, mask: nil)
        assertVectorsEqual(gpuMax, cpuMax)
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }
}
