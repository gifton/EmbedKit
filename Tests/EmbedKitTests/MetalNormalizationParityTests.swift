import XCTest
@testable import EmbedKit

final class MetalNormalizationParityTests: XCTestCase {
    func testL2NormalizationParity() async throws {
        #if canImport(Metal)
        let acc = await MetalAccelerator()
        guard await acc.isAvailable else { throw XCTSkip("Metal not available or metallib not loadable") }

        // Generate a small batch of random vectors
        let n = 8, d = 64
        var input: [[Float]] = []
        input.reserveCapacity(n)
        var seed: UInt64 = 0xDEADBEEF
        func rand() -> Float {
            seed = seed &* 2862933555777941757 &+ 3037000493
            let x = Double((seed >> 33) & 0xFFFF_FFFF)
            return Float((x / Double(UInt32.max)) * 2.0 - 1.0)
        }
        for _ in 0..<n { input.append((0..<d).map { _ in rand() }) }

        // CPU baseline
        func cpu(_ v: [[Float]]) -> [[Float]] {
            v.map { row in
                let norm = max(1e-12, sqrt(row.reduce(0) { $0 + Double($1) * Double($1) }))
                return row.map { $0 / Float(norm) }
            }
        }
        let cpuOut = cpu(input)
        let gpuOut = await acc.l2Normalize(input)

        // Compare with tolerance to allow minor FP differences
        let tol: Float = 1e-4
        XCTAssertEqual(gpuOut.count, cpuOut.count)
        for i in 0..<n {
            XCTAssertEqual(gpuOut[i].count, cpuOut[i].count)
            for j in 0..<d {
                let a = gpuOut[i][j], b = cpuOut[i][j]
                if abs(a - b) > tol {
                    XCTFail("Mismatch at [\(i),\(j)]: \(a) vs \(b)")
                    return
                }
            }
        }
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }
}

