// Tests for CoreMLBackend output flattening on non-contiguous MLMultiArray.
// Regression for audit 2.2: a padded/strided ANE/GPU output must be gathered
// through its strides rather than read linearly (which would ingest padding).
import Testing
import Foundation
@testable import EmbedKit

#if canImport(CoreML)
import CoreML

@Suite("CoreML Backend - strided output")
struct CoreMLBackendStrideTests {

    @Test("isDenselyContiguous detects a packed layout")
    func contiguityDetectionPacked() throws {
        let packed = try MLMultiArray(shape: [2, 3], dataType: .float32)
        #expect(CoreMLBackend.isDenselyContiguous(packed))
    }

    @Test("isDenselyContiguous rejects a padded layout")
    func contiguityDetectionPadded() throws {
        let rows = 2, cols = 3, rowStride = 4
        let backing = UnsafeMutablePointer<Float>.allocate(capacity: rows * rowStride)
        defer { backing.deallocate() }
        let array = try MLMultiArray(
            dataPointer: UnsafeMutableRawPointer(backing),
            shape: [NSNumber(value: rows), NSNumber(value: cols)],
            dataType: .float32,
            strides: [NSNumber(value: rowStride), NSNumber(value: 1)]
        )
        #expect(!CoreMLBackend.isDenselyContiguous(array))
    }

    @Test("gatherStrided skips row padding")
    func gatherSkipsPadding() throws {
        // Logical [2,3] with a padded row stride of 4 (one float of padding/row).
        let rows = 2, cols = 3, rowStride = 4
        let backing = UnsafeMutablePointer<Float>.allocate(capacity: rows * rowStride)
        defer { backing.deallocate() }
        for i in 0..<(rows * rowStride) { backing[i] = 99 } // sentinel padding
        backing[0] = 1; backing[1] = 2; backing[2] = 3
        backing[4] = 4; backing[5] = 5; backing[6] = 6

        let array = try MLMultiArray(
            dataPointer: UnsafeMutableRawPointer(backing),
            shape: [NSNumber(value: rows), NSNumber(value: cols)],
            dataType: .float32,
            strides: [NSNumber(value: rowStride), NSNumber(value: 1)]
        )

        #expect(array.count == rows * cols)
        let gathered = CoreMLBackend.gatherStrided(base: backing, array: array, count: array.count) { $0 }
        #expect(gathered == [1, 2, 3, 4, 5, 6])
    }
}
#endif
