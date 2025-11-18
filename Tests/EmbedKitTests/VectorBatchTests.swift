import XCTest
@testable import EmbedKit

/// Comprehensive tests for VectorBatch container
final class VectorBatchTests: XCTestCase {

    // MARK: - Initialization Tests

    func testInitFromFlatBuffer() throws {
        // Create batch from flat buffer
        let flatData: [Float] = [1, 2, 3, 4, 5, 6]  // 2 vectors × 3 dimensions
        let batch = try VectorBatch(data: flatData, count: 2, dimensions: 3)

        XCTAssertEqual(batch.count, 2)
        XCTAssertEqual(batch.dimensions, 3)
        XCTAssertEqual(batch.totalElements, 6)
        XCTAssertEqual(batch.data, flatData)
    }

    func testInitFromFlatBufferInvalidSize() {
        // Mismatched data size
        let flatData: [Float] = [1, 2, 3, 4, 5]  // 5 elements, not divisible by dimensions

        XCTAssertThrowsError(try VectorBatch(data: flatData, count: 2, dimensions: 3)) { error in
            guard case MetalError.invalidInput(let message) = error else {
                XCTFail("Expected MetalError.invalidInput")
                return
            }
            XCTAssertTrue(message.contains("doesn't match"))
        }
    }

    func testInitFromFlatBufferZeroDimensions() {
        let flatData: [Float] = [1, 2, 3]

        XCTAssertThrowsError(try VectorBatch(data: flatData, count: 3, dimensions: 0)) { error in
            guard case MetalError.invalidInput(let message) = error else {
                XCTFail("Expected MetalError.invalidInput")
                return
            }
            XCTAssertTrue(message.contains("must be positive"))
        }
    }

    func testInitFromFlatBufferNegativeCount() {
        let flatData: [Float] = [1, 2, 3]

        XCTAssertThrowsError(try VectorBatch(data: flatData, count: -1, dimensions: 3)) { error in
            guard case MetalError.invalidInput(let message) = error else {
                XCTFail("Expected MetalError.invalidInput")
                return
            }
            XCTAssertTrue(message.contains("non-negative"))
        }
    }

    func testInitFromVectors() throws {
        let vectors: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]

        let batch = try VectorBatch(vectors: vectors)

        XCTAssertEqual(batch.count, 3)
        XCTAssertEqual(batch.dimensions, 3)
        XCTAssertEqual(batch.totalElements, 9)
        XCTAssertEqual(batch.data, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    }

    func testInitFromEmptyVectors() {
        let empty: [[Float]] = []

        XCTAssertThrowsError(try VectorBatch(vectors: empty)) { error in
            guard case MetalError.invalidInput(let message) = error else {
                XCTFail("Expected MetalError.invalidInput")
                return
            }
            XCTAssertTrue(message.contains("empty array"))
        }
    }

    func testInitFromVectorsWithMismatchedDimensions() {
        let vectors: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0],        // Wrong dimension!
            [7.0, 8.0, 9.0]
        ]

        XCTAssertThrowsError(try VectorBatch(vectors: vectors)) { error in
            guard case MetalError.invalidInput(let message) = error else {
                XCTFail("Expected MetalError.invalidInput")
                return
            }
            XCTAssertTrue(message.contains("index 1"))
            XCTAssertTrue(message.contains("2 dimensions"))
            XCTAssertTrue(message.contains("expected 3"))
        }
    }

    func testInitFromVectorsWithEmptyVector() {
        let vectors: [[Float]] = [
            []  // Empty vector
        ]

        XCTAssertThrowsError(try VectorBatch(vectors: vectors)) { error in
            guard case MetalError.invalidInput = error else {
                XCTFail("Expected MetalError.invalidInput")
                return
            }
        }
    }

    func testEmptyBatch() throws {
        let empty = try VectorBatch.empty(dimensions: 768)

        XCTAssertEqual(empty.count, 0)
        XCTAssertEqual(empty.dimensions, 768)
        XCTAssertTrue(empty.isEmpty)
        XCTAssertEqual(empty.totalElements, 0)
        XCTAssertEqual(empty.data, [])
    }

    func testEmptyBatchInvalidDimensions() {
        XCTAssertThrowsError(try VectorBatch.empty(dimensions: 0))
        XCTAssertThrowsError(try VectorBatch.empty(dimensions: -5))
    }

    // MARK: - Access Tests

    func testSubscriptAccess() throws {
        let batch = try VectorBatch(vectors: [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])

        // Access first vector
        let first = batch[0]
        XCTAssertEqual(Array(first), [1.0, 2.0, 3.0])

        // Access middle vector
        let second = batch[1]
        XCTAssertEqual(Array(second), [4.0, 5.0, 6.0])

        // Access last vector
        let third = batch[2]
        XCTAssertEqual(Array(third), [7.0, 8.0, 9.0])
    }

    func testSubscriptIsSlice() throws {
        // Verify subscript returns a slice (no copy)
        let batch = try VectorBatch(vectors: [[1, 2, 3]])
        let slice = batch[0]

        XCTAssertTrue(type(of: slice) == ArraySlice<Float>.self)
    }

    func testRangeSubscript() throws {
        let batch = try VectorBatch(vectors: [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0]
        ])

        // Get middle two vectors
        let middle = batch[1..<3]

        XCTAssertEqual(middle.count, 2)
        XCTAssertEqual(middle.dimensions, 2)
        XCTAssertEqual(Array(middle[0]), [3.0, 4.0])
        XCTAssertEqual(Array(middle[1]), [5.0, 6.0])
    }

    func testRangeSubscriptEmpty() throws {
        let batch = try VectorBatch(vectors: [[1, 2], [3, 4]])
        let empty = batch[1..<1]

        XCTAssertEqual(empty.count, 0)
        XCTAssertTrue(empty.isEmpty)
    }

    func testRangeSubscriptFull() throws {
        let batch = try VectorBatch(vectors: [[1, 2], [3, 4], [5, 6]])
        let full = batch[0..<3]

        XCTAssertEqual(full.count, 3)
        XCTAssertEqual(full, batch)
    }

    // MARK: - Conversion Tests

    func testToArraysRoundTrip() throws {
        let original: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]

        let batch = try VectorBatch(vectors: original)
        let converted = batch.toArrays()

        XCTAssertEqual(converted.count, original.count)
        for (idx, vector) in converted.enumerated() {
            XCTAssertEqual(vector, original[idx])
        }
    }

    func testToArraysEmptyBatch() throws {
        let empty = try VectorBatch.empty(dimensions: 5)
        let arrays = empty.toArrays()

        XCTAssertTrue(arrays.isEmpty)
    }

    func testWithUnsafeBufferPointer() throws {
        let batch = try VectorBatch(vectors: [[1, 2, 3], [4, 5, 6]])

        batch.withUnsafeBufferPointer { ptr in
            XCTAssertEqual(ptr.count, 6)
            XCTAssertEqual(ptr[0], 1.0)
            XCTAssertEqual(ptr[3], 4.0)
            XCTAssertEqual(ptr[5], 6.0)
        }
    }

    func testWithUnsafeMutableBufferPointer() throws {
        var batch = try VectorBatch(vectors: [[1, 2], [3, 4]])

        batch.withUnsafeMutableBufferPointer { ptr in
            ptr[0] = 10.0
            ptr[3] = 40.0
        }

        XCTAssertEqual(Array(batch[0]), [10.0, 2.0])
        XCTAssertEqual(Array(batch[1]), [3.0, 40.0])
    }

    // MARK: - Mutation Tests

    func testAppendVector() throws {
        var batch = try VectorBatch(vectors: [[1, 2, 3], [4, 5, 6]])

        try batch.append([7, 8, 9])

        XCTAssertEqual(batch.count, 3)
        XCTAssertEqual(Array(batch[2]), [7.0, 8.0, 9.0])
    }

    func testAppendVectorDimensionMismatch() throws {
        var batch = try VectorBatch(vectors: [[1, 2, 3]])

        XCTAssertThrowsError(try batch.append([4, 5])) { error in
            guard case MetalError.invalidInput(let message) = error else {
                XCTFail("Expected MetalError.invalidInput")
                return
            }
            XCTAssertTrue(message.contains("2 dimensions"))
            XCTAssertTrue(message.contains("expected 3"))
        }
    }

    func testAppendBatch() throws {
        var batch1 = try VectorBatch(vectors: [[1, 2], [3, 4]])
        let batch2 = try VectorBatch(vectors: [[5, 6], [7, 8]])

        try batch1.append(contentsOf: batch2)

        XCTAssertEqual(batch1.count, 4)
        XCTAssertEqual(Array(batch1[0]), [1, 2])
        XCTAssertEqual(Array(batch1[3]), [7, 8])
    }

    func testAppendBatchDimensionMismatch() throws {
        var batch1 = try VectorBatch(vectors: [[1, 2, 3]])
        let batch2 = try VectorBatch(vectors: [[4, 5]])

        XCTAssertThrowsError(try batch1.append(contentsOf: batch2)) { error in
            guard case MetalError.invalidInput(let message) = error else {
                XCTFail("Expected MetalError.invalidInput")
                return
            }
            XCTAssertTrue(message.contains("dimension"))
        }
    }

    func testAppendToEmptyBatch() throws {
        var batch = try VectorBatch.empty(dimensions: 3)
        try batch.append([1, 2, 3])

        XCTAssertEqual(batch.count, 1)
        XCTAssertEqual(Array(batch[0]), [1, 2, 3])
    }

    // MARK: - Computed Properties

    func testSizeInBytes() throws {
        let batch = try VectorBatch(vectors: [[1, 2, 3], [4, 5, 6]])

        // 6 floats × 4 bytes = 24 bytes
        XCTAssertEqual(batch.sizeInBytes, 24)
    }

    func testIsEmpty() throws {
        let empty = try VectorBatch.empty(dimensions: 5)
        XCTAssertTrue(empty.isEmpty)

        let nonEmpty = try VectorBatch(vectors: [[1, 2]])
        XCTAssertFalse(nonEmpty.isEmpty)
    }

    // MARK: - Equality Tests

    func testEquality() throws {
        let batch1 = try VectorBatch(vectors: [[1, 2], [3, 4]])
        let batch2 = try VectorBatch(vectors: [[1, 2], [3, 4]])
        let batch3 = try VectorBatch(vectors: [[1, 2], [3, 5]])

        XCTAssertEqual(batch1, batch2)
        XCTAssertNotEqual(batch1, batch3)
    }

    func testEqualityDifferentDimensions() throws {
        let batch1 = try VectorBatch(vectors: [[1, 2, 3]])
        let batch2 = try VectorBatch(vectors: [[1, 2]])

        XCTAssertNotEqual(batch1, batch2)
    }

    func testEqualityDifferentCount() throws {
        let batch1 = try VectorBatch(vectors: [[1, 2], [3, 4]])
        let batch2 = try VectorBatch(vectors: [[1, 2]])

        XCTAssertNotEqual(batch1, batch2)
    }

    // MARK: - String Representations

    func testDescription() throws {
        let batch = try VectorBatch(vectors: [[1, 2], [3, 4]])
        let desc = batch.description

        XCTAssertTrue(desc.contains("VectorBatch"))
        XCTAssertTrue(desc.contains("count: 2"))
        XCTAssertTrue(desc.contains("dimensions: 2"))
        XCTAssertTrue(desc.contains("bytes: 16"))
    }

    func testDebugDescription() throws {
        let batch = try VectorBatch(vectors: [[1, 2, 3], [4, 5, 6]])
        let debug = batch.debugDescription

        XCTAssertTrue(debug.contains("VectorBatch"))
        XCTAssertTrue(debug.contains("[0]:"))
        XCTAssertTrue(debug.contains("[1]:"))
    }

    func testDebugDescriptionLargeBatch() throws {
        // Create batch with many vectors (should truncate in debug output)
        var vectors: [[Float]] = []
        for i in 0..<10 {
            vectors.append([Float(i), Float(i + 1)])
        }

        let batch = try VectorBatch(vectors: vectors)
        let debug = batch.debugDescription

        XCTAssertTrue(debug.contains("..."))  // Truncation indicator
    }

    // MARK: - Map Transform Tests

    func testMap() throws {
        let batch = try VectorBatch(vectors: [[1, 2], [3, 4]])

        // Double all values
        let doubled = batch.map { vector in
            vector.map { $0 * 2.0 }
        }

        XCTAssertEqual(Array(doubled[0]), [2.0, 4.0])
        XCTAssertEqual(Array(doubled[1]), [6.0, 8.0])
    }

    func testMapPreservesStructure() throws {
        let batch = try VectorBatch(vectors: [[1, 2, 3], [4, 5, 6]])

        let transformed = batch.map { vector in
            // No-op transform
            Array(vector)
        }

        XCTAssertEqual(transformed.count, batch.count)
        XCTAssertEqual(transformed.dimensions, batch.dimensions)
        XCTAssertEqual(transformed, batch)
    }

    // MARK: - Edge Cases

    func testSingleVector() throws {
        let batch = try VectorBatch(vectors: [[1, 2, 3]])

        XCTAssertEqual(batch.count, 1)
        XCTAssertEqual(batch.dimensions, 3)
        XCTAssertEqual(Array(batch[0]), [1, 2, 3])
    }

    func testSingleDimension() throws {
        let batch = try VectorBatch(vectors: [[1], [2], [3]])

        XCTAssertEqual(batch.count, 3)
        XCTAssertEqual(batch.dimensions, 1)
        XCTAssertEqual(Array(batch[0]), [1])
        XCTAssertEqual(Array(batch[1]), [2])
        XCTAssertEqual(Array(batch[2]), [3])
    }

    func testLargeBatch() throws {
        // Create BERT-sized batch (100 vectors × 768 dimensions)
        var vectors: [[Float]] = []
        for i in 0..<100 {
            var vector: [Float] = []
            for j in 0..<768 {
                vector.append(Float(i * 768 + j))
            }
            vectors.append(vector)
        }

        let batch = try VectorBatch(vectors: vectors)

        XCTAssertEqual(batch.count, 100)
        XCTAssertEqual(batch.dimensions, 768)
        XCTAssertEqual(batch.totalElements, 76800)

        // Verify first and last vectors
        let firstVector = batch[0]
        let lastVector = batch[99]

        XCTAssertEqual(Array(firstVector)[0], 0.0)
        XCTAssertEqual(Array(lastVector)[767], Float(99 * 768 + 767))
    }

    func testHighDimensionalVectors() throws {
        // Test with very high dimensions (e.g., 4096D)
        let vector = [Float](repeating: 1.0, count: 4096)
        let batch = try VectorBatch(vectors: [vector, vector])

        XCTAssertEqual(batch.count, 2)
        XCTAssertEqual(batch.dimensions, 4096)
        XCTAssertEqual(batch.totalElements, 8192)
    }

    // MARK: - Performance-Critical Behavior

    func testNoCopyOnSubscript() throws {
        // Verify subscript doesn't copy data
        let batch = try VectorBatch(vectors: [[1, 2, 3]])

        // Multiple subscript accesses should return slices to same buffer
        let slice1 = batch[0]
        let slice2 = batch[0]

        // Slices should reference same underlying storage
        XCTAssertEqual(slice1.startIndex, slice2.startIndex)
        XCTAssertEqual(slice1.endIndex, slice2.endIndex)
    }

    func testFlatBufferIntegrity() throws {
        // Verify flat buffer layout is correct for GPU transfer
        let batch = try VectorBatch(vectors: [
            [1, 2, 3],
            [4, 5, 6]
        ])

        // Buffer should be: [1, 2, 3, 4, 5, 6]
        batch.withUnsafeBufferPointer { ptr in
            for i in 0..<6 {
                XCTAssertEqual(ptr[i], Float(i + 1))
            }
        }
    }
}
