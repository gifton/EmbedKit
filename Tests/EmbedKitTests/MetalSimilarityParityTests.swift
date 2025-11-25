import XCTest
@testable import EmbedKit

/// Tests GPU vs CPU parity for cosine similarity matrix computation
final class MetalSimilarityParityTests: XCTestCase {

    // MARK: - Test Helpers

    /// Deterministic pseudo-random number generator for reproducible tests
    private struct PRNG {
        var seed: UInt64

        mutating func next() -> Float {
            seed = seed &* 2862933555777941757 &+ 3037000493
            let x = Double((seed >> 33) & 0xFFFF_FFFF)
            return Float((x / Double(UInt32.max)) * 2.0 - 1.0)
        }
    }

    /// Generate random vectors
    private func generateVectors(count: Int, dimensions: Int, seed: UInt64 = 0xDEADBEEF) -> [[Float]] {
        var rng = PRNG(seed: seed)
        return (0..<count).map { _ in
            (0..<dimensions).map { _ in rng.next() }
        }
    }

    /// CPU cosine similarity reference implementation
    private func cpuCosineSimilarityMatrix(_ vectors: [[Float]]) -> [[Float]] {
        let n = vectors.count
        guard n > 0 else { return [] }

        // Compute norms
        let norms = vectors.map { row -> Float in
            let s = row.reduce(0) { $0 + Double($1) * Double($1) }
            return max(1e-12, Float(s).squareRoot())
        }

        // Compute similarity matrix
        var out = Array(repeating: Array(repeating: Float(0), count: n), count: n)
        for i in 0..<n {
            out[i][i] = 1.0  // Self-similarity is always 1
            for j in (i+1)..<n {
                let dot = zip(vectors[i], vectors[j]).reduce(0) { $0 + $1.0 * $1.1 }
                let cos = dot / (norms[i] * norms[j])
                out[i][j] = cos
                out[j][i] = cos
            }
        }
        return out
    }

    /// Assert two matrices are approximately equal
    private func assertMatricesEqual(_ a: [[Float]], _ b: [[Float]], tolerance: Float = 1e-4, file: StaticString = #file, line: UInt = #line) {
        XCTAssertEqual(a.count, b.count, "Matrix row counts differ", file: file, line: line)
        for i in 0..<a.count {
            XCTAssertEqual(a[i].count, b[i].count, "Matrix column counts differ at row \(i)", file: file, line: line)
            for j in 0..<a[i].count {
                if abs(a[i][j] - b[i][j]) > tolerance {
                    XCTFail("Mismatch at [\(i),\(j)]: GPU=\(a[i][j]) vs CPU=\(b[i][j]), diff=\(abs(a[i][j] - b[i][j]))", file: file, line: line)
                    return
                }
            }
        }
    }

    // MARK: - Basic Parity Tests

    func testCosineSimilarityParitySmall() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        // Small matrix (below GPU threshold, tests CPU path)
        let vectors = generateVectors(count: 8, dimensions: 64)

        let cpuResult = cpuCosineSimilarityMatrix(vectors)
        let gpuResult = await acc.cosineSimilarityMatrix(vectors)

        assertMatricesEqual(gpuResult, cpuResult)
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }

    func testCosineSimilarityParityMedium() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        // Medium matrix (above GPU threshold of 32)
        let vectors = generateVectors(count: 64, dimensions: 128, seed: 0xCAFEBABE)

        let cpuResult = cpuCosineSimilarityMatrix(vectors)
        let gpuResult = await acc.cosineSimilarityMatrix(vectors)

        assertMatricesEqual(gpuResult, cpuResult)
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }

    func testCosineSimilarityParityLarge() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        // Large matrix (typical batch size with 384-dim embeddings)
        let vectors = generateVectors(count: 128, dimensions: 384, seed: 0xFEEDFACE)

        let cpuResult = cpuCosineSimilarityMatrix(vectors)
        let gpuResult = await acc.cosineSimilarityMatrix(vectors)

        assertMatricesEqual(gpuResult, cpuResult)
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }

    // MARK: - Edge Cases

    func testCosineSimilaritySingleVector() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        // Single vector: 1x1 matrix with self-similarity = 1.0
        let vectors = generateVectors(count: 1, dimensions: 64, seed: 0x11111111)

        let cpuResult = cpuCosineSimilarityMatrix(vectors)
        let gpuResult = await acc.cosineSimilarityMatrix(vectors)

        XCTAssertEqual(gpuResult.count, 1)
        XCTAssertEqual(gpuResult[0].count, 1)
        XCTAssertEqual(gpuResult[0][0], 1.0, accuracy: 1e-5)
        assertMatricesEqual(gpuResult, cpuResult)
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }

    func testCosineSimilarityTwoVectors() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        // Two vectors: 2x2 symmetric matrix
        let vectors = generateVectors(count: 2, dimensions: 64, seed: 0x22222222)

        let cpuResult = cpuCosineSimilarityMatrix(vectors)
        let gpuResult = await acc.cosineSimilarityMatrix(vectors)

        // Verify symmetry
        XCTAssertEqual(gpuResult[0][1], gpuResult[1][0], accuracy: 1e-6)
        // Verify diagonal is 1.0
        XCTAssertEqual(gpuResult[0][0], 1.0, accuracy: 1e-5)
        XCTAssertEqual(gpuResult[1][1], 1.0, accuracy: 1e-5)

        assertMatricesEqual(gpuResult, cpuResult)
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }

    func testCosineSimilarityIdenticalVectors() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        // Identical vectors should have similarity 1.0
        var rng = PRNG(seed: 0x33333333)
        let template = (0..<64).map { _ in rng.next() }
        let vectors = [[Float]](repeating: template, count: 16)

        let gpuResult = await acc.cosineSimilarityMatrix(vectors)

        // All similarities should be 1.0
        for i in 0..<vectors.count {
            for j in 0..<vectors.count {
                XCTAssertEqual(gpuResult[i][j], 1.0, accuracy: 1e-4, "Expected similarity 1.0 at [\(i),\(j)]")
            }
        }
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }

    func testCosineSimilarityOrthogonalVectors() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        // Create orthogonal vectors using standard basis
        let dims = 64
        var vectors: [[Float]] = []
        for i in 0..<min(dims, 32) {
            var v = [Float](repeating: 0, count: dims)
            v[i] = 1.0
            vectors.append(v)
        }

        let gpuResult = await acc.cosineSimilarityMatrix(vectors)

        // Diagonal should be 1.0, off-diagonal should be 0.0
        for i in 0..<vectors.count {
            for j in 0..<vectors.count {
                let expected: Float = (i == j) ? 1.0 : 0.0
                XCTAssertEqual(gpuResult[i][j], expected, accuracy: 1e-5, "Expected \(expected) at [\(i),\(j)]")
            }
        }
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }

    func testCosineSimilarityOppositeVectors() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        // Vector and its negation should have similarity -1.0
        var rng = PRNG(seed: 0x44444444)
        let v1 = (0..<64).map { _ in rng.next() }
        let v2 = v1.map { -$0 }
        let vectors = [v1, v2]

        let gpuResult = await acc.cosineSimilarityMatrix(vectors)

        XCTAssertEqual(gpuResult[0][0], 1.0, accuracy: 1e-5)
        XCTAssertEqual(gpuResult[1][1], 1.0, accuracy: 1e-5)
        XCTAssertEqual(gpuResult[0][1], -1.0, accuracy: 1e-4)
        XCTAssertEqual(gpuResult[1][0], -1.0, accuracy: 1e-4)
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }

    // MARK: - Matrix Properties

    func testCosineSimilaritySymmetry() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        let vectors = generateVectors(count: 64, dimensions: 128, seed: 0x55555555)
        let gpuResult = await acc.cosineSimilarityMatrix(vectors)

        // Verify matrix is symmetric
        for i in 0..<vectors.count {
            for j in 0..<vectors.count {
                XCTAssertEqual(gpuResult[i][j], gpuResult[j][i], accuracy: 1e-6, "Matrix not symmetric at [\(i),\(j)]")
            }
        }
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }

    func testCosineSimilarityDiagonal() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        let vectors = generateVectors(count: 64, dimensions: 128, seed: 0x66666666)
        let gpuResult = await acc.cosineSimilarityMatrix(vectors)

        // Verify diagonal is all 1.0 (self-similarity)
        for i in 0..<vectors.count {
            XCTAssertEqual(gpuResult[i][i], 1.0, accuracy: 1e-4, "Diagonal should be 1.0 at [\(i),\(i)]")
        }
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }

    func testCosineSimilarityBounds() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        let vectors = generateVectors(count: 64, dimensions: 128, seed: 0x77777777)
        let gpuResult = await acc.cosineSimilarityMatrix(vectors)

        // Verify all values are in [-1, 1]
        for i in 0..<vectors.count {
            for j in 0..<vectors.count {
                XCTAssertGreaterThanOrEqual(gpuResult[i][j], -1.0 - 1e-5, "Value below -1 at [\(i),\(j)]")
                XCTAssertLessThanOrEqual(gpuResult[i][j], 1.0 + 1e-5, "Value above 1 at [\(i),\(j)]")
            }
        }
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }

    // MARK: - High Dimensional Tests

    func testCosineSimilarityHighDimensional() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        // Test with 1024-dimensional vectors
        let vectors = generateVectors(count: 32, dimensions: 1024, seed: 0x88888888)

        let cpuResult = cpuCosineSimilarityMatrix(vectors)
        let gpuResult = await acc.cosineSimilarityMatrix(vectors)

        assertMatricesEqual(gpuResult, cpuResult)
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }

    // MARK: - Tiling Tests

    func testCosineSimilarityTiledComputation() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        // Large enough to trigger tiling (N > 1024)
        // Using smaller dimensions to keep test fast
        let vectors = generateVectors(count: 256, dimensions: 64, seed: 0x99999999)

        let cpuResult = cpuCosineSimilarityMatrix(vectors)
        let gpuResult = await acc.cosineSimilarityMatrix(vectors)

        assertMatricesEqual(gpuResult, cpuResult, tolerance: 1e-3)  // Slightly higher tolerance for tiled computation
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }

    func testCosineSimilarityExplicitTileSize() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        // Test with explicit small tile size to force tiling
        let vectors = generateVectors(count: 128, dimensions: 64, seed: 0xAAAAAAAA)

        let cpuResult = cpuCosineSimilarityMatrix(vectors)
        let gpuResult = await acc.cosineSimilarityMatrix(vectors, tileSize: 32)

        assertMatricesEqual(gpuResult, cpuResult, tolerance: 1e-3)
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }

    // MARK: - Empty Input

    func testCosineSimilarityEmptyInput() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available") }

        let vectors: [[Float]] = []
        let gpuResult = await acc.cosineSimilarityMatrix(vectors)

        XCTAssertEqual(gpuResult.count, 0)
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }
}
