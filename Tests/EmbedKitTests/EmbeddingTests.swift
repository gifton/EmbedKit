//
//  EmbeddingTests.swift
//  EmbedKitTests
//
//  Comprehensive tests for dimension-specific embedding types
//

import XCTest
import VectorCore
@testable import EmbedKit

// MARK: - Embedding Type Tests

final class EmbeddingTests: XCTestCase {

    // MARK: - Initialization Tests

    func testZeroInitialization() {
        let emb384 = Embedding384.zeros()
        XCTAssertEqual(emb384.dimensions, 384)
        XCTAssertEqual(emb384.magnitude, 0.0, accuracy: 1e-6)
        XCTAssertTrue(emb384.isZero)

        let emb768 = Embedding768.zeros()
        XCTAssertEqual(emb768.dimensions, 768)
        XCTAssertEqual(emb768.magnitude, 0.0, accuracy: 1e-6)
        XCTAssertTrue(emb768.isZero)

        let emb1536 = Embedding1536.zeros()
        XCTAssertEqual(emb1536.dimensions, 1536)
        XCTAssertEqual(emb1536.magnitude, 0.0, accuracy: 1e-6)
        XCTAssertTrue(emb1536.isZero)
    }

    func testOnesInitialization() {
        let emb384 = Embedding384.ones()
        XCTAssertEqual(emb384.dimensions, 384)
        XCTAssertFalse(emb384.isZero)

        let emb768 = Embedding768.ones()
        XCTAssertEqual(emb768.dimensions, 768)
        XCTAssertFalse(emb768.isZero)

        let emb1536 = Embedding1536.ones()
        XCTAssertEqual(emb1536.dimensions, 1536)
        XCTAssertFalse(emb1536.isZero)
    }

    func testArrayInitialization() throws {
        // Valid initialization
        let values384 = [Float](repeating: 0.5, count: 384)
        let emb384 = try Embedding384(values384)
        XCTAssertEqual(emb384.dimensions, 384)
        XCTAssertEqual(emb384.toArray(), values384)

        let values768 = [Float](repeating: 0.5, count: 768)
        let emb768 = try Embedding768(values768)
        XCTAssertEqual(emb768.dimensions, 768)
        XCTAssertEqual(emb768.toArray(), values768)

        let values1536 = [Float](repeating: 0.5, count: 1536)
        let emb1536 = try Embedding1536(values1536)
        XCTAssertEqual(emb1536.dimensions, 1536)
        XCTAssertEqual(emb1536.toArray(), values1536)

        // Invalid initialization (wrong dimension)
        let wrongSize = [Float](repeating: 0.5, count: 100)
        XCTAssertThrowsError(try Embedding384(wrongSize)) { error in
            // Verify it's a dimension mismatch error
            XCTAssertTrue(String(describing: error).contains("dimension"), "Expected dimensionMismatch error")
        }
    }

    func testGeneratorInitialization() {
        // Linear increasing values
        let emb384 = Embedding384 { Float($0) }
        XCTAssertEqual(emb384[0], 0.0)
        XCTAssertEqual(emb384[1], 1.0)
        XCTAssertEqual(emb384[383], 383.0)

        // Constant values
        let emb768 = Embedding768 { _ in 2.5 }
        XCTAssertTrue(emb768.toArray().allSatisfy { $0 == 2.5 })

        // Index-dependent formula
        let emb1536 = Embedding1536 { (index: Int) -> Float in Float(index) / 1536.0 }
        XCTAssertEqual(emb1536[0], 0.0, accuracy: 1e-6)
        XCTAssertEqual(emb1536[1535], Float(1535.0) / Float(1536.0), accuracy: 1e-6)
    }

    func testRandomInitialization() {
        let random384 = Embedding384.random(in: -1...1)
        XCTAssertEqual(random384.dimensions, 384)
        XCTAssertTrue(random384.toArray().allSatisfy { $0 >= -1 && $0 <= 1 })

        let random768 = Embedding768.random(in: 0...1)
        XCTAssertEqual(random768.dimensions, 768)
        XCTAssertTrue(random768.toArray().allSatisfy { $0 >= 0 && $0 <= 1 })

        let random1536 = Embedding1536.random()
        XCTAssertEqual(random1536.dimensions, 1536)
        XCTAssertTrue(random1536.toArray().allSatisfy { $0 >= 0 && $0 <= 1 })
    }

    func testRandomUnitInitialization() {
        let unit384 = Embedding384.randomUnit()
        XCTAssertEqual(unit384.dimensions, 384)
        XCTAssertEqual(unit384.magnitude, 1.0, accuracy: 1e-5)

        let unit768 = Embedding768.randomUnit()
        XCTAssertEqual(unit768.dimensions, 768)
        XCTAssertEqual(unit768.magnitude, 1.0, accuracy: 1e-5)

        let unit1536 = Embedding1536.randomUnit()
        XCTAssertEqual(unit1536.dimensions, 1536)
        XCTAssertEqual(unit1536.magnitude, 1.0, accuracy: 1e-5)
    }

    // MARK: - Normalization Tests

    func testNormalization() throws {
        // Non-zero vector normalization
        let values384 = (0..<384).map { Float($0 + 1) }
        let emb384 = try Embedding384(values384)
        let normalized384 = try emb384.normalized().get()

        XCTAssertEqual(normalized384.magnitude, 1.0, accuracy: 1e-5)
        XCTAssertEqual(normalized384.dimensions, 384)

        // Verify direction preserved (proportional components)
        let originalMag = emb384.magnitude
        for i in 0..<384 {
            XCTAssertEqual(normalized384[i], emb384[i] / originalMag, accuracy: 1e-5)
        }

        // Zero vector normalization should fail
        let zero768 = Embedding768.zeros()
        XCTAssertThrowsError(try zero768.normalized().get()) { error in
            // Verify it's an invalid operation error (zero vector normalization)
            XCTAssertTrue(String(describing: error).contains("invalid") || String(describing: error).contains("zero"), "Expected invalidOperation error")
        }
    }

    func testNormalizedIsIdempotent() throws {
        let emb = try Embedding768([Float](repeating: 2.0, count: 768))
        let normalized1 = try emb.normalized().get()
        let normalized2 = try normalized1.normalized().get()

        XCTAssertEqual(normalized1.magnitude, 1.0, accuracy: 1e-6)
        XCTAssertEqual(normalized2.magnitude, 1.0, accuracy: 1e-6)

        // Should be essentially the same after normalization
        for i in 0..<768 {
            XCTAssertEqual(normalized1[i], normalized2[i], accuracy: 1e-5)
        }
    }

    // MARK: - Similarity and Distance Tests

    func testCosineSimilarity() throws {
        // Identical vectors
        let emb1 = try Embedding384([Float](repeating: 1.0, count: 384))
        let emb2 = try Embedding384([Float](repeating: 1.0, count: 384))
        XCTAssertEqual(emb1.cosineSimilarity(to: emb2), 1.0, accuracy: 1e-6)

        // Opposite vectors
        let emb3 = try Embedding384([Float](repeating: -1.0, count: 384))
        XCTAssertEqual(emb1.cosineSimilarity(to: emb3), -1.0, accuracy: 1e-6)

        // Orthogonal vectors (first half vs second half non-zero)
        var values1 = [Float](repeating: 0.0, count: 384)
        var values2 = [Float](repeating: 0.0, count: 384)
        for i in 0..<192 {
            values1[i] = 1.0
            values2[i + 192] = 1.0
        }
        let ortho1 = try Embedding384(values1)
        let ortho2 = try Embedding384(values2)
        XCTAssertEqual(ortho1.cosineSimilarity(to: ortho2), 0.0, accuracy: 1e-6)

        // Normalized embeddings via dot product
        let norm1 = try emb1.normalized().get()
        let norm2 = try emb2.normalized().get()
        let dotSimilarity = norm1.dotProduct(norm2)
        let cosineSimilarity = emb1.cosineSimilarity(to: emb2)
        XCTAssertEqual(dotSimilarity, cosineSimilarity, accuracy: 1e-5)
    }

    func testCosineDistance() throws {
        let emb1 = try Embedding768([Float](repeating: 1.0, count: 768))
        let emb2 = try Embedding768([Float](repeating: 1.0, count: 768))

        // cosine_distance = 1 - cosine_similarity
        let similarity = emb1.cosineSimilarity(to: emb2)
        let distance = emb1.cosineDistance(to: emb2)
        XCTAssertEqual(distance, 1.0 - similarity, accuracy: 1e-6)

        // Identical vectors have distance 0
        XCTAssertEqual(distance, 0.0, accuracy: 1e-6)

        // Opposite vectors have distance 2
        let emb3 = try Embedding768([Float](repeating: -1.0, count: 768))
        XCTAssertEqual(emb1.cosineDistance(to: emb3), 2.0, accuracy: 1e-6)
    }

    func testEuclideanDistance() throws {
        let emb1 = try Embedding1536([Float](repeating: 0.0, count: 1536))
        let emb2 = try Embedding1536([Float](repeating: 1.0, count: 1536))

        // Distance between [0,0,...] and [1,1,...] is sqrt(n)
        let expectedDistance = sqrt(Float(1536))
        XCTAssertEqual(emb1.euclideanDistance(to: emb2), expectedDistance, accuracy: 1e-4)

        // Distance to self is 0
        XCTAssertEqual(emb1.euclideanDistance(to: emb1), 0.0, accuracy: 1e-6)

        // Verify symmetry
        XCTAssertEqual(
            emb1.euclideanDistance(to: emb2),
            emb2.euclideanDistance(to: emb1),
            accuracy: 1e-6
        )
    }

    func testEuclideanDistanceSquared() throws {
        let emb1 = try Embedding384([Float](repeating: 0.0, count: 384))
        let emb2 = try Embedding384([Float](repeating: 1.0, count: 384))

        let distSquared = emb1.euclideanDistanceSquared(to: emb2)
        let dist = emb1.euclideanDistance(to: emb2)

        XCTAssertEqual(distSquared, dist * dist, accuracy: 1e-4)
        XCTAssertEqual(distSquared, Float(384), accuracy: 1e-4)
    }

    func testDotProduct() throws {
        // Standard dot product
        let emb1 = try Embedding768([Float](repeating: 2.0, count: 768))
        let emb2 = try Embedding768([Float](repeating: 3.0, count: 768))

        // 2 * 3 * 768 = 4608
        XCTAssertEqual(emb1.dotProduct(emb2), 4608.0, accuracy: 1e-4)

        // Orthogonality test
        var values1 = [Float](repeating: 0.0, count: 768)
        var values2 = [Float](repeating: 0.0, count: 768)
        for i in 0..<384 {
            values1[i] = 1.0
            values2[i + 384] = 1.0
        }
        let ortho1 = try Embedding768(values1)
        let ortho2 = try Embedding768(values2)
        XCTAssertEqual(ortho1.dotProduct(ortho2), 0.0, accuracy: 1e-6)
    }

    // MARK: - Collection Conformance Tests

    func testCollectionSubscript() throws {
        let values = (0..<384).map { Float($0) }
        let emb = try Embedding384(values)

        // Test subscript access
        XCTAssertEqual(emb[0], 0.0)
        XCTAssertEqual(emb[100], 100.0)
        XCTAssertEqual(emb[383], 383.0)

        // Test mutation
        var mutableEmb = emb
        mutableEmb[0] = 999.0
        XCTAssertEqual(mutableEmb[0], 999.0)
        XCTAssertEqual(emb[0], 0.0)  // Original unchanged
    }

    func testCollectionIteration() throws {
        let values = (0..<768).map { Float($0) }
        let emb = try Embedding768(values)

        var sum: Float = 0
        for value in emb {
            sum += value
        }

        let expectedSum = values.reduce(0, +)
        XCTAssertEqual(sum, expectedSum, accuracy: 1e-3)
    }

    func testCollectionIndices() {
        let emb = Embedding1536.zeros()

        XCTAssertEqual(emb.startIndex, 0)
        XCTAssertEqual(emb.endIndex, 1536)
        XCTAssertEqual(emb.count, 1536)
    }

    // MARK: - Type Safety Tests

    func testCompileTimeDimensionSafety() {
        // This test verifies that the type system prevents dimension mismatches
        let emb384 = Embedding384.zeros()
        let emb768 = Embedding768.zeros()

        // These operations compile because dimensions match
        _ = emb384.cosineSimilarity(to: emb384)
        _ = emb768.cosineSimilarity(to: emb768)

        // Uncommenting the following would cause compile-time errors:
        // let invalid = emb384.cosineSimilarity(to: emb768)  // ❌ Type mismatch
        // let alsoInvalid = emb768.cosineSimilarity(to: emb384)  // ❌ Type mismatch

        // This test passes if it compiles
        XCTAssertEqual(emb384.dimensions, 384)
        XCTAssertEqual(emb768.dimensions, 768)
    }

    // MARK: - Equatable & Hashable Tests

    func testEquality() throws {
        let emb1 = try Embedding384([Float](repeating: 1.0, count: 384))
        let emb2 = try Embedding384([Float](repeating: 1.0, count: 384))
        let emb3 = try Embedding384([Float](repeating: 2.0, count: 384))

        XCTAssertEqual(emb1, emb2)
        XCTAssertNotEqual(emb1, emb3)
        XCTAssertNotEqual(emb2, emb3)
    }

    func testHashing() throws {
        let emb1 = try Embedding768([Float](repeating: 1.0, count: 768))
        let emb2 = try Embedding768([Float](repeating: 1.0, count: 768))
        let emb3 = try Embedding768([Float](repeating: 2.0, count: 768))

        // Equal objects must have equal hashes
        XCTAssertEqual(emb1.hashValue, emb2.hashValue)

        // Use in Set
        var set = Set<Embedding768>()
        set.insert(emb1)
        set.insert(emb2)
        set.insert(emb3)

        XCTAssertEqual(set.count, 2)  // emb1 and emb2 are equal
        XCTAssertTrue(set.contains(emb1))
        XCTAssertTrue(set.contains(emb3))
    }

    // MARK: - Codable Tests

    func testCodableRoundTrip() throws {
        let original = try Embedding384((0..<384).map { Float($0) / 384.0 })

        // Encode
        let encoder = JSONEncoder()
        let data = try encoder.encode(original)

        // Decode
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(Embedding384.self, from: data)

        // Verify equality
        XCTAssertEqual(original, decoded)
        XCTAssertEqual(original.dimensions, decoded.dimensions)
        XCTAssertEqual(original.toArray(), decoded.toArray())
    }

    func testCodableWithDifferentDimensions() throws {
        let emb768 = Embedding768.random(in: -1...1)
        let emb1536 = Embedding1536.random(in: -1...1)

        // Encode both
        let encoder = JSONEncoder()
        let data768 = try encoder.encode(emb768)
        let data1536 = try encoder.encode(emb1536)

        // Decode
        let decoder = JSONDecoder()
        let decoded768 = try decoder.decode(Embedding768.self, from: data768)
        let decoded1536 = try decoder.decode(Embedding1536.self, from: data1536)

        XCTAssertEqual(emb768, decoded768)
        XCTAssertEqual(emb1536, decoded1536)
    }

    // MARK: - Properties Tests

    func testMagnitude() throws {
        // Unit vector in first dimension
        var values = [Float](repeating: 0.0, count: 384)
        values[0] = 1.0
        let emb = try Embedding384(values)
        XCTAssertEqual(emb.magnitude, 1.0, accuracy: 1e-6)

        // All ones
        let ones = Embedding768.ones()
        XCTAssertEqual(ones.magnitude, sqrt(Float(768)), accuracy: 1e-4)

        // Zero vector
        let zeros = Embedding1536.zeros()
        XCTAssertEqual(zeros.magnitude, 0.0, accuracy: 1e-6)
    }

    func testMagnitudeSquared() throws {
        let values = [Float](repeating: 2.0, count: 384)
        let emb = try Embedding384(values)

        let magSquared = emb.magnitudeSquared
        let mag = emb.magnitude

        // For Float precision near sqrt(1536), the minimal achievable
        // difference between (mag * mag) and magSquared is ~3e-4 due to
        // ULP spacing. Use a slightly looser tolerance.
        XCTAssertEqual(magSquared, mag * mag, accuracy: 5e-4)
        XCTAssertEqual(magSquared, Float(384 * 4), accuracy: 1e-4)  // 384 * 2^2
    }

    func testIsFinite() throws {
        let finite = Embedding384.random(in: -1...1)
        XCTAssertTrue(finite.isFinite)

        // Create embedding with infinity
        var values = [Float](repeating: 1.0, count: 384)
        values[0] = Float.infinity
        let infinite = try Embedding384(values)
        XCTAssertFalse(infinite.isFinite)

        // Create embedding with NaN
        values[0] = Float.nan
        let nan = try Embedding384(values)
        XCTAssertFalse(nan.isFinite)
    }

    func testIsZero() {
        let zeros = Embedding768.zeros()
        XCTAssertTrue(zeros.isZero)

        let nonZeros = Embedding768.ones()
        XCTAssertFalse(nonZeros.isZero)

        var partialZero = Embedding1536.zeros()
        partialZero[0] = 0.001
        XCTAssertFalse(partialZero.isZero)
    }

    // MARK: - Performance Tests

    func testNormalizationPerformance() throws {
        let embedding = Embedding768.random(in: -1...1)

        measure {
            for _ in 0..<10000 {
                _ = try? embedding.normalized()
            }
        }
    }

    func testCosineSimilarityPerformance() throws {
        let emb1 = Embedding1536.random(in: -1...1)
        let emb2 = Embedding1536.random(in: -1...1)

        measure {
            for _ in 0..<10000 {
                _ = emb1.cosineSimilarity(to: emb2)
            }
        }
    }

    func testDotProductPerformance() throws {
        let emb1 = Embedding768.random(in: -1...1)
        let emb2 = Embedding768.random(in: -1...1)

        measure {
            for _ in 0..<10000 {
                _ = emb1.dotProduct(emb2)
            }
        }
    }
}

// MARK: - Dynamic Embedding Tests

final class DynamicEmbeddingTests: XCTestCase {

    // MARK: - Initialization Tests

    func testZeroInitialization() throws {
        let emb384 = try DynamicEmbedding.zeros(dimension: 384)
        XCTAssertEqual(emb384.dimensions, 384)
        XCTAssertEqual(emb384.magnitude, 0.0, accuracy: 1e-6)
        XCTAssertTrue(emb384.isZero)

        let emb768 = try DynamicEmbedding.zeros(dimension: 768)
        XCTAssertEqual(emb768.dimensions, 768)

        let emb1536 = try DynamicEmbedding.zeros(dimension: 1536)
        XCTAssertEqual(emb1536.dimensions, 1536)

        // Unsupported dimension
        XCTAssertThrowsError(try DynamicEmbedding.zeros(dimension: 512)) { error in
            guard case EmbeddingError.unsupportedDimension = error else {
                XCTFail("Expected unsupportedDimension error")
                return
            }
        }
    }

    func testArrayInitialization() throws {
        let values384 = [Float](repeating: 0.5, count: 384)
        let emb384 = try DynamicEmbedding(values: values384)
        XCTAssertEqual(emb384.dimensions, 384)
        XCTAssertEqual(emb384.toArray(), values384)

        let values768 = [Float](repeating: 0.5, count: 768)
        let emb768 = try DynamicEmbedding(values: values768)
        XCTAssertEqual(emb768.dimensions, 768)

        // Invalid dimension
        let invalid = [Float](repeating: 0.5, count: 512)
        XCTAssertThrowsError(try DynamicEmbedding(values: invalid)) { error in
            guard case EmbeddingError.unsupportedDimension = error else {
                XCTFail("Expected unsupportedDimension error")
                return
            }
        }
    }

    func testRandomInitialization() throws {
        let random384 = try DynamicEmbedding.random(dimension: 384, in: -1...1)
        XCTAssertEqual(random384.dimensions, 384)
        XCTAssertTrue(random384.toArray().allSatisfy { $0 >= -1 && $0 <= 1 })

        let random768 = try DynamicEmbedding.random(dimension: 768)
        XCTAssertEqual(random768.dimensions, 768)
        XCTAssertTrue(random768.toArray().allSatisfy { $0 >= 0 && $0 <= 1 })
    }

    // MARK: - Dimension Tests

    func testSupportedDimensions() {
        let supported = DynamicEmbedding.supportedDimensions
        XCTAssertEqual(supported, [384, 768, 1536])
    }

    func testHasDimension() throws {
        let emb768 = try DynamicEmbedding.zeros(dimension: 768)
        XCTAssertTrue(emb768.hasDimension(768))
        XCTAssertFalse(emb768.hasDimension(384))
        XCTAssertFalse(emb768.hasDimension(1536))
    }

    // MARK: - Typed Accessors Tests

    func testTypedAccessors() throws {
        let emb384 = try DynamicEmbedding.random(dimension: 384)
        XCTAssertNotNil(emb384.as384)
        XCTAssertNil(emb384.as768)
        XCTAssertNil(emb384.as1536)

        let emb768 = try DynamicEmbedding.random(dimension: 768)
        XCTAssertNil(emb768.as384)
        XCTAssertNotNil(emb768.as768)
        XCTAssertNil(emb768.as1536)

        let emb1536 = try DynamicEmbedding.random(dimension: 1536)
        XCTAssertNil(emb1536.as384)
        XCTAssertNil(emb1536.as768)
        XCTAssertNotNil(emb1536.as1536)
    }

    func testTypedConversion() throws {
        let dynamic = try DynamicEmbedding.random(dimension: 768)

        guard let typed = dynamic.as768 else {
            XCTFail("Failed to convert to Embedding768")
            return
        }

        XCTAssertEqual(typed.dimensions, 768)
        XCTAssertEqual(typed.toArray(), dynamic.toArray())
    }

    // MARK: - Normalization Tests

    func testNormalization() throws {
        let values = (0..<384).map { Float($0 + 1) }
        let emb = try DynamicEmbedding(values: values)
        let normalized = try emb.normalized()

        XCTAssertEqual(normalized.magnitude, 1.0, accuracy: 1e-5)
        XCTAssertEqual(normalized.dimensions, 384)

        // Zero vector normalization should fail
        let zero = try DynamicEmbedding.zeros(dimension: 768)
        XCTAssertThrowsError(try zero.normalized())
    }

    // MARK: - Similarity Tests

    func testCosineSimilarity() throws {
        let emb1 = try DynamicEmbedding(values: [Float](repeating: 1.0, count: 768))
        let emb2 = try DynamicEmbedding(values: [Float](repeating: 1.0, count: 768))

        let similarity = try emb1.cosineSimilarity(to: emb2)
        XCTAssertEqual(similarity, 1.0, accuracy: 1e-6)

        // Dimension mismatch
        let emb3 = try DynamicEmbedding.zeros(dimension: 384)
        XCTAssertThrowsError(try emb1.cosineSimilarity(to: emb3)) { error in
            guard case EmbeddingError.dimensionMismatch = error else {
                XCTFail("Expected dimensionMismatch error")
                return
            }
        }
    }

    func testCosineDistance() throws {
        let emb1 = try DynamicEmbedding(values: [Float](repeating: 1.0, count: 768))
        let emb2 = try DynamicEmbedding(values: [Float](repeating: 1.0, count: 768))

        let distance = try emb1.cosineDistance(to: emb2)
        XCTAssertEqual(distance, 0.0, accuracy: 1e-6)

        let emb3 = try DynamicEmbedding(values: [Float](repeating: -1.0, count: 768))
        let oppositeDistance = try emb1.cosineDistance(to: emb3)
        XCTAssertEqual(oppositeDistance, 2.0, accuracy: 1e-6)
    }

    func testEuclideanDistance() throws {
        let emb1 = try DynamicEmbedding.zeros(dimension: 1536)
        let emb2 = try DynamicEmbedding.ones(dimension: 1536)

        let distance = try emb1.euclideanDistance(to: emb2)
        XCTAssertEqual(distance, sqrt(Float(1536)), accuracy: 1e-4)

        // Dimension mismatch
        let emb3 = try DynamicEmbedding.zeros(dimension: 384)
        XCTAssertThrowsError(try emb1.euclideanDistance(to: emb3))
    }

    func testDotProduct() throws {
        let emb1 = try DynamicEmbedding(values: [Float](repeating: 2.0, count: 384))
        let emb2 = try DynamicEmbedding(values: [Float](repeating: 3.0, count: 384))

        let dot = try emb1.dotProduct(emb2)
        XCTAssertEqual(dot, 2.0 * 3.0 * 384.0, accuracy: 1e-4)

        // Dimension mismatch
        let emb3 = try DynamicEmbedding.zeros(dimension: 768)
        XCTAssertThrowsError(try emb1.dotProduct(emb3))
    }

    // MARK: - Equatable & Hashable Tests

    func testEquality() throws {
        let emb1 = try DynamicEmbedding(values: [Float](repeating: 1.0, count: 384))
        let emb2 = try DynamicEmbedding(values: [Float](repeating: 1.0, count: 384))
        let emb3 = try DynamicEmbedding(values: [Float](repeating: 2.0, count: 384))
        let emb4 = try DynamicEmbedding(values: [Float](repeating: 1.0, count: 768))

        XCTAssertEqual(emb1, emb2)
        XCTAssertNotEqual(emb1, emb3)
        XCTAssertNotEqual(emb1, emb4)  // Different dimensions
    }

    func testHashing() throws {
        let emb1 = try DynamicEmbedding.ones(dimension: 768)
        let emb2 = try DynamicEmbedding.ones(dimension: 768)

        XCTAssertEqual(emb1.hashValue, emb2.hashValue)

        var set = Set<DynamicEmbedding>()
        set.insert(emb1)
        set.insert(emb2)
        XCTAssertEqual(set.count, 1)
    }

    // MARK: - Codable Tests

    func testCodableRoundTrip() throws {
        let original = try DynamicEmbedding.random(dimension: 768)

        let encoder = JSONEncoder()
        let data = try encoder.encode(original)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(DynamicEmbedding.self, from: data)

        XCTAssertEqual(original, decoded)
        XCTAssertEqual(original.dimensions, decoded.dimensions)
    }

    func testCodablePreservesDimension() throws {
        let emb384 = try DynamicEmbedding.random(dimension: 384)
        let emb768 = try DynamicEmbedding.random(dimension: 768)
        let emb1536 = try DynamicEmbedding.random(dimension: 1536)

        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        let data384 = try encoder.encode(emb384)
        let decoded384 = try decoder.decode(DynamicEmbedding.self, from: data384)
        XCTAssertEqual(decoded384.dimensions, 384)

        let data768 = try encoder.encode(emb768)
        let decoded768 = try decoder.decode(DynamicEmbedding.self, from: data768)
        XCTAssertEqual(decoded768.dimensions, 768)

        let data1536 = try encoder.encode(emb1536)
        let decoded1536 = try decoder.decode(DynamicEmbedding.self, from: data1536)
        XCTAssertEqual(decoded1536.dimensions, 1536)
    }

    // MARK: - Heterogeneous Collection Tests

    func testHeterogeneousCollection() throws {
        let embeddings: [DynamicEmbedding] = [
            try DynamicEmbedding.random(dimension: 384),
            try DynamicEmbedding.random(dimension: 768),
            try DynamicEmbedding.random(dimension: 1536),
            try DynamicEmbedding.random(dimension: 384)
        ]

        XCTAssertEqual(embeddings[0].dimensions, 384)
        XCTAssertEqual(embeddings[1].dimensions, 768)
        XCTAssertEqual(embeddings[2].dimensions, 1536)
        XCTAssertEqual(embeddings[3].dimensions, 384)

        // Can store different dimensions
        let dims = embeddings.map(\.dimensions)
        XCTAssertEqual(dims, [384, 768, 1536, 384])
    }
}

// MARK: - Dimension Tests

final class EmbeddingDimensionTests: XCTestCase {

    func testDimensionValues() {
        // Test dimension values match expected constants
        XCTAssertEqual(EmbedKit.Dim384.value as Int, 384)
        XCTAssertEqual(EmbedKit.Dim768.value as Int, 768)
        XCTAssertEqual(EmbedKit.Dim1536.value as Int, 1536)
    }

    func testDimensionUtilities() {
        XCTAssertEqual(Dim384.dimensionName, "Dim384")
        XCTAssertEqual(Dim768.dimensionName, "Dim768")
        XCTAssertEqual(Dim1536.dimensionName, "Dim1536")

        XCTAssertEqual(Dim384.bytesPerEmbedding, 384 * 4)  // 4 bytes per Float
        XCTAssertEqual(Dim768.bytesPerEmbedding, 768 * 4)
        XCTAssertEqual(Dim1536.bytesPerEmbedding, 1536 * 4)
    }

    func testMemoryCalculations() {
        // 1M embeddings of Dim768 = 768 * 4 bytes * 1M = 3,072 MB
        let megabytesPerMillion768 = Dim768.megabytesPerMillion
        let expected768 = Double(768 * 4 * 1_000_000) / (1024 * 1024)
        XCTAssertEqual(megabytesPerMillion768, expected768, accuracy: 0.1)
    }

    func testValidateEmbeddingDimension() {
        XCTAssertEqual(validateEmbeddingDimension(Dim384.self), 384)
        XCTAssertEqual(validateEmbeddingDimension(Dim768.self), 768)
        XCTAssertEqual(validateEmbeddingDimension(Dim1536.self), 1536)
    }
}
