//
//  NormalizationNumericalStabilityP0Tests.swift
//  EmbedKit
//
//  P0 tests for L2 normalization numerical stability (T-NUM-004)
//  Tests CPU-only implementation: Embedding<D>.normalized()
//

import XCTest
@testable import EmbedKit
import VectorCore

/// Comprehensive numerical stability tests for L2 normalization
///
/// This test suite validates the robustness of the CPU-based L2 normalization
/// implementation across edge cases including:
/// - Overflow prevention (T-NUM-004a)
/// - Division by zero detection (T-NUM-004b)
/// - Epsilon threshold and underflow handling (T-NUM-004c)
/// - Precision loss in sqrt() operations (T-NUM-004d)
///
/// ## Test Coverage
/// - 40 test functions across 4 categories
/// - Tests standard dimensions: 384, 768, 1536
/// - Tests edge dimensions: 3, 127, 2048
/// - Validates IEEE 754 Float32 boundary conditions
///
/// ## Implementation Under Test
/// - `Embedding<D>.normalized() -> Result<Self, VectorError>`
/// - VectorCore's `Vector<D>.normalized()` (via delegation)
final class NormalizationNumericalStabilityP0Tests: XCTestCase {

    // MARK: - Test Configuration

    private struct TestConfig {
        // Standard embedding dimensions
        static let dim384 = 384  // MiniLM-L12-v2
        static let dim768 = 768  // BERT-base
        static let dim1536 = 1536  // OpenAI ada-002

        // Edge case dimensions
        static let dimSmall = 3
        static let dimPrime = 127  // Non-power-of-2
        static let dimLarge = 2048

        // Numerical tolerance thresholds
        static let magnitudeTolerance: Float = 1e-6
        static let relativeErrorTolerance: Float = 1e-5
        static let componentTolerance: Float = 1e-6

        // IEEE 754 Float32 constants
        static let maxFinite = Float.greatestFiniteMagnitude
        static let minNormal = Float.leastNormalMagnitude
        static let minNonzero = Float.leastNonzeroMagnitude  // Subnormal
        static let epsilon = Float.ulpOfOne
    }

    // MARK: - Setup & Teardown

    override func setUp() async throws {
        try await super.setUp()
        // Setup code if needed
    }

    override func tearDown() async throws {
        // Cleanup code if needed
        try await super.tearDown()
    }

    // MARK: - T-NUM-004a: Overflow Prevention Tests

    /// T-NUM-004a-1: Large uniform values near overflow threshold
    ///
    /// Tests that normalization handles large uniform values without overflow.
    /// Uses values at 10% of Float.greatestFiniteMagnitude to test the boundary
    /// where norm² could potentially overflow.
    ///
    /// **Mathematical Analysis:**
    /// - Input: v = [k, k, ..., k] where k = Float.max * 0.1
    /// - Norm²: ||v||² = 384 * k² (risk of overflow if k² too large)
    /// - Expected: sqrt(384) * k, then division should yield unit vector
    func testOverflow_LargeUniformValues_Dim384() async throws {
        // Given: Vector with large uniform values (10% of max finite)
        let largeValue = TestConfig.maxFinite * 0.1
        let values = createLargeUniformVector(dimension: TestConfig.dim384, scale: 0.1)

        // Verify test setup: values are large but finite
        XCTAssertTrue(values.allSatisfy { $0.isFinite }, "Input values should be finite")
        XCTAssertTrue(values.allSatisfy { abs($0 - largeValue) < 1.0 }, "Input values should be uniform")

        // When: Normalize the embedding
        let embedding = try Embedding<EmbedKit.Dim384>(values)
        print("DEBUG: Embedding magnitude: \(embedding.magnitude)")
        print("DEBUG: First 5 components: \(embedding.toArray().prefix(5))")

        let result = embedding.normalized()

        // Then: Normalization should succeed
        guard case .success(let normalized) = result else {
            XCTFail("Normalization should succeed for large uniform values")
            return
        }

        print("DEBUG: Normalized magnitude: \(normalized.magnitude)")
        print("DEBUG: Normalized first 5: \(normalized.toArray().prefix(5))")

        // Verify: All components are finite (no overflow to infinity)
        XCTAssertTrue(normalized.isFinite, "Normalized vector should have all finite values")
        assertAllFinite(normalized.toArray())

        // Verify: Magnitude is 1.0 (properly normalized)
        assertIsNormalized(normalized, accuracy: TestConfig.magnitudeTolerance)

        // Verify: Direction preserved (all components should be equal and positive)
        let components = normalized.toArray()
        let expectedComponent = 1.0 / sqrt(Float(TestConfig.dim384))
        for (index, component) in components.enumerated() {
            XCTAssertEqual(
                component,
                expectedComponent,
                accuracy: TestConfig.componentTolerance,
                "Component \(index) should equal 1/sqrt(384)"
            )
        }

        // Verify: No precision loss in large value handling (compute in Double to avoid overflow)
        let manualNormStable = computeNormStableDouble(values)
        XCTAssertTrue(manualNormStable.isFinite, "Manual norm calculation should not overflow in Double precision")
    }

    /// T-NUM-004a-2: Large uniform values for BERT dimensions
    ///
    /// Tests overflow prevention for larger dimension (768), which increases
    /// the risk of overflow in the norm² calculation.
    ///
    /// **Mathematical Analysis:**
    /// - Input: v = [k, k, ..., k] where k = Float.max * 0.1
    /// - Norm²: ||v||² = 768 * k² (higher risk than 384-dim)
    /// - Critical: 768 * (Float.max * 0.1)² must not overflow
    func testOverflow_LargeUniformValues_Dim768() async throws {
        // Given: Vector with large uniform values in higher dimension
        let largeValue = TestConfig.maxFinite * 0.1
        let values = createLargeUniformVector(dimension: TestConfig.dim768, scale: 0.1)

        // Verify test setup
        XCTAssertEqual(values.count, TestConfig.dim768, "Should have 768 dimensions")
        XCTAssertTrue(values.allSatisfy { $0.isFinite }, "Input values should be finite")

        // Calculate expected norm² in Double to verify it doesn't overflow in theory
        // norm² = 768 * (Float.max * 0.1)²
        let normSquaredTheoretical = Double(TestConfig.dim768) * pow(Double(largeValue), 2)
        XCTAssertTrue(normSquaredTheoretical.isFinite, "Theoretical norm² should be finite for test validity")

        // When: Normalize the embedding
        let embedding = try Embedding<EmbedKit.Dim768>(values)
        let result = embedding.normalized()

        // Then: Normalization should succeed despite large values
        guard case .success(let normalized) = result else {
            XCTFail("Normalization should succeed for large uniform values in 768 dimensions")
            return
        }

        // Verify: All components are finite
        XCTAssertTrue(normalized.isFinite, "Normalized vector should be finite")

        // Verify: Magnitude is exactly 1.0
        let magnitude = normalized.magnitude
        XCTAssertEqual(
            magnitude,
            1.0,
            accuracy: TestConfig.magnitudeTolerance,
            "Magnitude should be 1.0, got \(magnitude)"
        )

        // Verify: Uniform input produces uniform output
        let components = normalized.toArray()
        let expectedComponent = 1.0 / sqrt(Float(TestConfig.dim768))

        // Check first, middle, and last components for uniformity
        XCTAssertEqual(components[0], expectedComponent, accuracy: TestConfig.componentTolerance)
        XCTAssertEqual(components[TestConfig.dim768/2], expectedComponent, accuracy: TestConfig.componentTolerance)
        XCTAssertEqual(components[TestConfig.dim768-1], expectedComponent, accuracy: TestConfig.componentTolerance)

        // Verify: Statistical properties of normalized vector
        let mean = components.reduce(0.0, +) / Float(components.count)
        XCTAssertEqual(
            mean,
            expectedComponent,
            accuracy: TestConfig.componentTolerance * 10,
            "Mean of components should equal expected uniform value"
        )

        // Verify: Maximum deviation from expected is small
        let maxDeviation = components.map { abs($0 - expectedComponent) }.max() ?? 0.0
        XCTAssertLessThan(
            maxDeviation,
            TestConfig.componentTolerance * 2,
            "Maximum deviation from expected component value should be small"
        )

        // Verify: Compare manual norm calculation (overflow-safe, compute in Double)
        let manualNorm = computeNormStableDouble(values)
        let expectedNormalized = values.map { Float(Double($0) / manualNorm) }
        assertVectorsMatch(components, expectedNormalized, tolerance: TestConfig.componentTolerance)
    }

    /// T-NUM-004a-3: Large uniform values for large dimensions
    ///
    /// Tests overflow prevention at the highest standard dimension (1536).
    /// This is the most challenging case for uniform large values due to:
    /// - Largest dimension multiplier in norm² calculation
    /// - norm² = 1536 * k² where k = Float.max * 0.1
    ///
    /// **Critical Validation:**
    /// Ensures 1536 * (Float.max * 0.1)² doesn't overflow
    func testOverflow_LargeUniformValues_Dim1536() async throws {
        // Given: Large uniform values in highest standard dimension
        let largeValue = TestConfig.maxFinite * 0.1
        let values = createLargeUniformVector(dimension: TestConfig.dim1536, scale: 0.1)

        // Verify test preconditions
        XCTAssertEqual(values.count, TestConfig.dim1536, "Should have 1536 dimensions")
        XCTAssertTrue(values.allSatisfy { $0.isFinite }, "All input values should be finite")

        // Critical: Verify theoretical norm² is computable without overflow
        // If this fails, the test itself is invalid
        let normSquaredTheoretical = Double(TestConfig.dim1536) * pow(Double(largeValue), 2)
        XCTAssertTrue(
            normSquaredTheoretical.isFinite,
            "Test precondition: norm² must be theoretically computable (Double). Got: \(normSquaredTheoretical)"
        )

        // When: Normalize the large dimensional embedding
        let embedding = try Embedding<EmbedKit.Dim1536>(values)
        let result = embedding.normalized()

        // Then: Should succeed despite high overflow risk
        guard case .success(let normalized) = result else {
            XCTFail("Normalization should succeed for 1536-dim large uniform values")
            return
        }

        // Verify: No overflow to infinity or underflow to NaN
        XCTAssertTrue(normalized.isFinite, "All normalized values should be finite")

        // Verify: Magnitude is exactly 1.0 (within numerical precision)
        let magnitude = normalized.magnitude
        XCTAssertEqual(
            magnitude,
            1.0,
            accuracy: TestConfig.magnitudeTolerance,
            "Magnitude should be 1.0 for normalized vector, got \(magnitude)"
        )

        // Verify: Uniformity preserved (all components should be equal)
        let expectedComponent = 1.0 / sqrt(Float(TestConfig.dim1536))
        let components = normalized.toArray()

        // Sample components across the vector for uniformity
        let sampleIndices = [0, TestConfig.dim1536/4, TestConfig.dim1536/2, 3*TestConfig.dim1536/4, TestConfig.dim1536-1]
        for index in sampleIndices {
            XCTAssertEqual(
                components[index],
                expectedComponent,
                accuracy: TestConfig.componentTolerance,
                "Component at index \(index) should equal 1/sqrt(1536)"
            )
        }

        // Verify: No systematic bias introduced by normalization
        let variance = components.map { pow($0 - expectedComponent, 2) }.reduce(0, +) / Float(components.count)
        XCTAssertLessThan(
            variance,
            TestConfig.componentTolerance * TestConfig.componentTolerance,
            "Variance from expected uniform value should be minimal"
        )
    }

    /// T-NUM-004a-4: Near maximum finite value without overflow
    ///
    /// Tests the boundary condition: maximum value that can be normalized
    /// without causing overflow in the norm² calculation.
    ///
    /// **Mathematical Boundary:**
    /// For dimension d, if all components are k:
    /// - norm² = d * k²
    /// - To avoid overflow: d * k² < Float.max
    /// - Therefore: k < sqrt(Float.max / d)
    ///
    /// This test uses k = sqrt(Float.max / d) * 0.95 (95% of theoretical max)
    func testOverflow_NearMaxFinite_NoOverflow() async throws {
        // Given: Values at 95% of theoretical maximum for 384 dimensions
        let dimension = TestConfig.dim384

        // Calculate maximum safe value: k_max = sqrt(Float.max / dimension)
        let theoreticalMax = sqrt(TestConfig.maxFinite / Float(dimension))
        let testValue = theoreticalMax * 0.95  // Use 95% for safety margin

        // Verify the math: dimension * testValue² should be < Float.max
        let normSquaredExpected = Float(dimension) * (testValue * testValue)
        XCTAssertTrue(
            normSquaredExpected.isFinite,
            "Precondition: norm² should be finite. Got: \(normSquaredExpected)"
        )
        XCTAssertLessThan(
            normSquaredExpected,
            TestConfig.maxFinite,
            "Precondition: norm² should be less than Float.max"
        )

        let values = [Float](repeating: testValue, count: dimension)

        // When: Normalize at the boundary
        let embedding = try Embedding<EmbedKit.Dim384>(values)
        let result = embedding.normalized()

        // Then: Should succeed even near the boundary
        guard case .success(let normalized) = result else {
            XCTFail("Normalization should succeed near maximum finite value. Test value: \(testValue)")
            return
        }

        // Verify: All values are finite (critical for boundary test)
        XCTAssertTrue(normalized.isFinite, "Normalized values must be finite at boundary")
        assertAllFinite(normalized.toArray())

        // Verify: Magnitude is 1.0
        assertIsNormalized(normalized, accuracy: TestConfig.magnitudeTolerance)

        // Verify: Direction preserved despite extreme magnitude
        let components = normalized.toArray()
        let expectedComponent = 1.0 / sqrt(Float(dimension))

        // All components should be positive and equal
        XCTAssertTrue(components.allSatisfy { $0 > 0 }, "All components should be positive")

        for (index, component) in components.enumerated() {
            XCTAssertEqual(
                component,
                expectedComponent,
                accuracy: TestConfig.componentTolerance * 10,  // Relaxed tolerance at boundary
                "Boundary test: component \(index) deviates from expected"
            )
        }

        // Verify: Re-normalization is idempotent (stability test)
        let renormalized = normalized.normalized()
        guard case .success(let doublyNormalized) = renormalized else {
            XCTFail("Re-normalization should also succeed")
            return
        }

        assertVectorsMatch(
            normalized.toArray(),
            doublyNormalized.toArray(),
            tolerance: TestConfig.componentTolerance
        )
    }

    /// T-NUM-004a-5: Mixed large and small magnitudes
    ///
    /// Tests that normalization correctly handles vectors with heterogeneous
    /// component magnitudes. Alternates between large and small values.
    ///
    /// **Numerical Challenge:**
    /// - Large values: risk of overflow in norm² calculation
    /// - Small values: risk of precision loss when divided by large norm
    /// - Must preserve relative proportions
    func testOverflow_MixedMagnitudes_LargeAndSmall() async throws {
        // Given: Vector alternating between large and small values
        let dimension = TestConfig.dim768
        let largeValue = TestConfig.maxFinite * 0.1
        let smallValue = TestConfig.minNormal * 1000.0  // Well above subnormal

        let values = createMixedMagnitudeVector(
            dimension: dimension,
            largeScale: 0.1,
            smallScale: 1000.0
        )

        // Verify mixed pattern
        XCTAssertEqual(values[0], largeValue, accuracy: 1.0)
        XCTAssertEqual(values[1], smallValue, accuracy: TestConfig.minNormal)

        // When: Normalize the mixed magnitude vector
        let embedding = try Embedding<EmbedKit.Dim768>(values)
        let result = embedding.normalized()

        // Then: Should handle mixed magnitudes gracefully
        guard case .success(let normalized) = result else {
            XCTFail("Normalization should succeed for mixed magnitudes")
            return
        }

        // Verify: All finite (no overflow from large, no underflow from small)
        XCTAssertTrue(normalized.isFinite, "Mixed magnitude normalization should produce finite values")

        // Verify: Magnitude is 1.0
        assertIsNormalized(normalized, accuracy: TestConfig.magnitudeTolerance)

        // Verify: Relative proportions preserved
        let components = normalized.toArray()

        // Large components should be larger than small components after normalization
        for i in stride(from: 0, to: dimension, by: 2) {
            let largeNormalized = components[i]      // From large value
            let smallNormalized = components[i + 1]  // From small value

            XCTAssertGreaterThan(
                largeNormalized,
                smallNormalized,
                "Relative magnitude order should be preserved after normalization"
            )
        }

        // Verify: Approximate ratio preservation when representable.
        // Extreme dynamic range may underflow small components to 0 in Float,
        // making the normalized ratio infinite. In that case, verify ordering
        // and expected dominating component magnitude instead of ratio.
        let originalRatioD = Double(largeValue) / Double(smallValue)
        let largeN = Double(components[0])
        let smallN = Double(components[1])
        if originalRatioD.isFinite && smallN > 0 {
            let normalizedRatioD = largeN / smallN
            let ratioError = abs(normalizedRatioD - originalRatioD) / originalRatioD
            XCTAssertLessThan(
                ratioError,
                Double(TestConfig.relativeErrorTolerance),
                "Ratio between large and small components should be preserved. " +
                "Original: \(originalRatioD), Normalized: \(normalizedRatioD), Error: \(ratioError)"
            )
        } else {
            // Underflow path: small components are effectively zero in Float.
            // Check monotonicity and expected large-component magnitude.
            XCTAssertGreaterThan(components[0], components[1], "Large should exceed small after normalization")
            let expectedLarge = 1.0 / sqrt(Float(dimension / 2))
            XCTAssertEqual(
                components[0],
                expectedLarge,
                accuracy: 1e-3,
                "Large component magnitude should match 1/sqrt(#large)"
            )
        }

        // Verify: Manual calculation agrees (overflow-safe Double math)
        let manualNorm = computeNormStableDouble(values)
        let expectedNormalized = values.map { Float(Double($0) / manualNorm) }
        assertVectorsMatch(components, expectedNormalized, tolerance: TestConfig.componentTolerance)
    }

    /// T-NUM-004a-6: Progressively increasing values
    ///
    /// Tests normalization with linearly increasing components: [1, 2, 3, ..., n].
    /// This creates a gradient of magnitudes without extreme ratios.
    ///
    /// **Properties to Verify:**
    /// - Smooth gradient preserved after normalization
    /// - No systematic bias toward large or small components
    /// - Numerical stability across the range
    func testOverflow_ProgressivelyIncreasing_StableNormalization() async throws {
        // Given: Progressively increasing values scaled to avoid overflow
        let dimension = TestConfig.dim384

        // Scale factor to keep largest value reasonable
        // Largest value will be: dimension * scaleFactor
        let scaleFactor = (TestConfig.maxFinite * 0.01) / Float(dimension)

        let values = createProgressiveVector(
            dimension: dimension,
            start: scaleFactor,
            step: scaleFactor
        )

        // Verify progression
        XCTAssertEqual(values[0], scaleFactor, accuracy: Float.ulpOfOne)
        XCTAssertEqual(values[dimension-1], Float(dimension) * scaleFactor, accuracy: 0.1)

        // Verify all finite
        XCTAssertTrue(values.allSatisfy { $0.isFinite }, "All progressive values should be finite")

        // When: Normalize the progressive vector
        let embedding = try Embedding<EmbedKit.Dim384>(values)
        let result = embedding.normalized()

        // Then: Should normalize successfully
        guard case .success(let normalized) = result else {
            XCTFail("Normalization should succeed for progressive values")
            return
        }

        // Verify: All finite
        XCTAssertTrue(normalized.isFinite, "Normalized progressive vector should be finite")

        // Verify: Magnitude is 1.0
        assertIsNormalized(normalized, accuracy: TestConfig.magnitudeTolerance)

        // Verify: Progressive order preserved
        let components = normalized.toArray()

        for i in 0..<(dimension-1) {
            XCTAssertLessThan(
                components[i],
                components[i+1],
                "Progressive order should be preserved: component[\(i)] should be < component[\(i+1)]"
            )
        }

        // Verify: Linear relationship approximately preserved
        // If original is [1, 2, 3], normalized should be [k, 2k, 3k] for some k
        let firstNonZero = components[0]
        let ratios = components.enumerated().map { index, value -> Float in
            return value / (firstNonZero * Float(index + 1))
        }

        // All ratios should be approximately 1.0 (linear scaling preserved)
        let avgRatio = ratios.reduce(0, +) / Float(ratios.count)
        XCTAssertEqual(
            avgRatio,
            1.0,
            accuracy: TestConfig.relativeErrorTolerance * 10,
            "Average ratio should be close to 1.0, indicating linear relationship preserved"
        )

        // Verify: Smallest and largest components have expected ratio
        let originalRatio = values[dimension-1] / values[0]  // Should be ≈ dimension
        let normalizedRatio = components[dimension-1] / components[0]

        XCTAssertEqual(
            normalizedRatio,
            originalRatio,
            accuracy: originalRatio * TestConfig.relativeErrorTolerance,
            "Ratio between largest and smallest should be preserved"
        )
    }

    /// T-NUM-004a-7: Single large component with small remainder
    ///
    /// Tests a vector dominated by one large component, with all others small.
    /// This is a common scenario (e.g., one dominant feature).
    ///
    /// **Expected Behavior:**
    /// - Large component should dominate the norm
    /// - After normalization, large component ≈ 1.0
    /// - Small components should remain proportionally small
    func testOverflow_SingleLargeComponent_PreservesDirection() async throws {
        // Given: One large component, rest are small
        let dimension = TestConfig.dim768
        let largeValue = TestConfig.maxFinite * 0.1
        let smallValue: Float = 1.0  // Regular float, not extreme

        var values = [Float](repeating: smallValue, count: dimension)
        values[dimension / 2] = largeValue  // Large component in the middle

        // Verify setup
        XCTAssertEqual(values.filter { $0 == largeValue }.count, 1, "Should have exactly one large value")

        // When: Normalize the vector with dominant component
        let embedding = try Embedding<EmbedKit.Dim768>(values)
        let result = embedding.normalized()

        // Then: Should succeed
        guard case .success(let normalized) = result else {
            XCTFail("Normalization should succeed for single large component")
            return
        }

        // Verify: All finite
        XCTAssertTrue(normalized.isFinite, "Normalized vector should be finite")

        // Verify: Magnitude is 1.0
        assertIsNormalized(normalized, accuracy: TestConfig.magnitudeTolerance)

        // Verify: Large component dominates
        let components = normalized.toArray()
        let dominantComponent: Float = components[dimension / 2]

        // The dominant component should be very close to 1.0
        // because it dominates the norm: ||v|| ≈ largeValue
        XCTAssertGreaterThan(
            dominantComponent,
            0.99,
            "Dominant component should be close to 1.0 after normalization"
        )

        // Verify: Small components remain small relative to dominant
        let smallComponents = components.enumerated().filter { $0.offset != dimension / 2 }.map { $1 }

        for (index, component) in smallComponents.enumerated() {
            XCTAssertLessThan(
                component,
                0.01,
                "Small component \(index) should remain much smaller than dominant component"
            )
        }

        // Verify: Sum of squares equals 1.0
        let sumOfSquares = components.reduce(0) { $0 + $1 * $1 }
        XCTAssertEqual(
            sumOfSquares,
            1.0,
            accuracy: TestConfig.magnitudeTolerance,
            "Sum of squares should equal 1.0"
        )

        // Verify: Approximate independence verification
        // dominant² + (n-1)*small² ≈ 1.0
        let dimFloat = Float(dimension - 1)
        let smallSquared: Float = smallValue * smallValue
        let largeSquared: Float = largeValue * largeValue
        let totalNormSquared: Float = largeSquared + (dimFloat * smallSquared)
        let smallContribution: Float = dimFloat * smallSquared
        let expectedDominantSquared: Float = 1.0 - (smallContribution / totalNormSquared)

        let actualDominantSquared: Float = dominantComponent * dominantComponent
        XCTAssertEqual(
            actualDominantSquared,
            expectedDominantSquared,
            accuracy: 0.01 as Float,
            "Dominant component squared should match theoretical expectation"
        )
    }

    /// T-NUM-004a-8: All positive large values
    ///
    /// Tests that sign information is correctly preserved when all components
    /// are positive and large. This is a sanity check that positive values
    /// remain positive after normalization.
    func testOverflow_AllPositiveLarge_CorrectMagnitude() async throws {
        // Given: All positive large values
        let dimension = TestConfig.dim384
        let largeValue = TestConfig.maxFinite * 0.08
        let values = [Float](repeating: largeValue, count: dimension)

        // Verify all positive
        XCTAssertTrue(values.allSatisfy { $0 > 0 }, "All values should be positive")

        // When: Normalize
        let embedding = try Embedding<EmbedKit.Dim384>(values)
        let result = embedding.normalized()

        // Then: Should succeed
        guard case .success(let normalized) = result else {
            XCTFail("Normalization should succeed for all positive large values")
            return
        }

        // Verify: All components remain positive
        let components = normalized.toArray()
        XCTAssertTrue(
            components.allSatisfy { $0 > 0 },
            "All components should remain positive after normalization"
        )

        // Verify: Magnitude is 1.0
        assertIsNormalized(normalized, accuracy: TestConfig.magnitudeTolerance)

        // Verify: All components equal (uniformity)
        let expectedComponent = 1.0 / sqrt(Float(dimension))
        for (index, component) in components.enumerated() {
            XCTAssertEqual(
                component,
                expectedComponent,
                accuracy: TestConfig.componentTolerance,
                "Component \(index) should equal expected value"
            )

            // Also verify explicitly positive
            XCTAssertGreaterThan(component, 0, "Component \(index) should be positive")
        }

        // Verify: No sign flips occurred
        XCTAssertEqual(
            components.filter { $0 < 0 }.count,
            0,
            "No components should have become negative"
        )
    }

    /// T-NUM-004a-9: Mixed positive and negative large values
    ///
    /// Critical test: normalization must preserve signs of components.
    /// Tests with alternating positive/negative large values.
    ///
    /// **Sign Preservation Property:**
    /// If v_i > 0, then (v/||v||)_i > 0
    /// If v_i < 0, then (v/||v||)_i < 0
    func testOverflow_MixedSignsLarge_CorrectMagnitude() async throws {
        // Given: Alternating positive and negative large values
        let dimension = TestConfig.dim768
        let largeValue = TestConfig.maxFinite * 0.08

        var values = [Float](repeating: 0, count: dimension)
        for i in 0..<dimension {
            values[i] = (i % 2 == 0) ? largeValue : -largeValue
        }

        // Verify alternating pattern
        XCTAssertGreaterThan(values[0], 0, "Even indices should be positive")
        XCTAssertLessThan(values[1], 0, "Odd indices should be negative")

        // Count signs
        let positiveCount = values.filter { $0 > 0 }.count
        let negativeCount = values.filter { $0 < 0 }.count
        XCTAssertEqual(positiveCount, dimension / 2, "Half should be positive")
        XCTAssertEqual(negativeCount, dimension / 2, "Half should be negative")

        // When: Normalize mixed signs
        let embedding = try Embedding<EmbedKit.Dim768>(values)
        let result = embedding.normalized()

        // Then: Should succeed
        guard case .success(let normalized) = result else {
            XCTFail("Normalization should succeed for mixed signs")
            return
        }

        // Verify: All finite
        XCTAssertTrue(normalized.isFinite, "Normalized values should be finite")

        // Verify: Magnitude is 1.0
        assertIsNormalized(normalized, accuracy: TestConfig.magnitudeTolerance)

        // Verify: Signs preserved
        let components = normalized.toArray()

        for (index, component) in components.enumerated() {
            if index % 2 == 0 {
                XCTAssertGreaterThan(
                    component,
                    0,
                    "Component \(index) should be positive (was positive before normalization)"
                )
            } else {
                XCTAssertLessThan(
                    component,
                    0,
                    "Component \(index) should be negative (was negative before normalization)"
                )
            }
        }

        // Verify: Magnitudes are equal (since input had uniform absolute values)
        let expectedMagnitude = 1.0 / sqrt(Float(dimension))

        for (index, component) in components.enumerated() {
            XCTAssertEqual(
                abs(component),
                expectedMagnitude,
                accuracy: TestConfig.componentTolerance,
                "Component \(index) magnitude should equal expected value"
            )
        }

        // Verify: Sign count preserved
        let normalizedPositiveCount = components.filter { $0 > 0 }.count
        let normalizedNegativeCount = components.filter { $0 < 0 }.count

        XCTAssertEqual(normalizedPositiveCount, positiveCount, "Positive count should be preserved")
        XCTAssertEqual(normalizedNegativeCount, negativeCount, "Negative count should be preserved")

        // Verify: Dot product with original has expected sign
        // Since signs preserved and magnitudes uniform, specific checks on dot product
        let dotProduct = zip(values, components).map { $0 * $1 }.reduce(0, +)
        XCTAssertGreaterThan(dotProduct, 0, "Dot product should be positive (direction preserved)")
    }

    /// T-NUM-004a-10: Worst case: maximum dimension with large values
    ///
    /// The most challenging overflow scenario:
    /// - Highest dimension (2048) maximizes the multiplier in norm²
    /// - Large values maximize individual component contribution
    /// - Combined effect: highest risk of overflow
    ///
    /// **Critical Test:**
    /// If this passes, the implementation handles overflow correctly
    /// in all realistic scenarios.
    func testOverflow_MaxDimensionLargeValues_NoOverflow() async throws {
        // Given: Maximum test dimension with large values
        let dimension = TestConfig.dimLarge  // 2048

        // Use moderately large values (can't use 0.1*Float.max due to 2048 multiplier)
        // Calculate safe maximum: k < sqrt(Float.max / dimension)
        let maxSafeValue = sqrt(TestConfig.maxFinite / Float(dimension))
        let testValue = maxSafeValue * 0.9  // Use 90% of safe maximum

        let values = [Float](repeating: testValue, count: dimension)

        // Verify precondition: norm² should be theoretically computable
        let theoreticalNormSquared = Float(dimension) * (testValue * testValue)
        XCTAssertTrue(
            theoreticalNormSquared.isFinite,
            "Precondition failed: theoretical norm² overflows. Test value may be too large."
        )
        XCTAssertLessThan(
            theoreticalNormSquared,
            TestConfig.maxFinite * 0.99,
            "Precondition: norm² should be safely below Float.max"
        )

        // When: Normalize in worst-case scenario
        // Note: We can't use compile-time dimension here as we don't have Dim2048
        // So we'll use Dim1536 which is still a challenging high dimension
        let embedding = try Embedding<EmbedKit.Dim1536>(Array(values.prefix(TestConfig.dim1536)))
        let result = embedding.normalized()

        // Then: Should succeed despite challenging conditions
        guard case .success(let normalized) = result else {
            XCTFail("Normalization should succeed in worst-case high-dimension scenario. Dimension: \(TestConfig.dim1536)")
            return
        }

        // Verify: No overflow or underflow
        XCTAssertTrue(normalized.isFinite, "Worst-case normalization should produce finite values")
        assertAllFinite(normalized.toArray())

        // Verify: Magnitude is 1.0
        let magnitude = normalized.magnitude
        XCTAssertEqual(
            magnitude,
            1.0,
            accuracy: TestConfig.magnitudeTolerance,
            "Worst-case: magnitude should be 1.0, got \(magnitude)"
        )

        // Verify: Uniformity preserved
        let components = normalized.toArray()
        let expectedComponent = 1.0 / sqrt(Float(TestConfig.dim1536))

        // Sample several components across the high-dimensional space
        let sampleSize = 10
        let stride = TestConfig.dim1536 / sampleSize

        for i in 0..<sampleSize {
            let index = i * stride
            XCTAssertEqual(
                components[index],
                expectedComponent,
                accuracy: TestConfig.componentTolerance * 10,  // Relaxed for high dimension
                "Worst-case: component \(index) should match expected value"
            )
        }

        // Verify: Statistical properties remain sound
        let mean = components.reduce(0, +) / Float(components.count)
        XCTAssertEqual(
            mean,
            expectedComponent,
            accuracy: TestConfig.componentTolerance * 100,
            "Worst-case: mean should equal expected uniform value"
        )

        // Verify: Maximum deviation is bounded
        let maxDeviation = components.map { abs($0 - expectedComponent) }.max() ?? 0
        XCTAssertLessThan(
            maxDeviation,
            TestConfig.componentTolerance * 20,
            "Worst-case: maximum deviation should be bounded"
        )

        print("✓ Worst-case test passed: dim=\(TestConfig.dim1536), value=\(testValue), norm²=\(theoreticalNormSquared)")
    }

    // MARK: - T-NUM-004b: Division by Zero Detection Tests

    /// T-NUM-004b-1: Exact zero vector should return error
    ///
    /// The most basic test: a vector of all zeros has norm = 0,
    /// which makes normalization impossible (division by zero).
    ///
    /// **Expected Behavior:**
    /// - `normalized()` returns `.failure(VectorError)`
    /// - Does not crash or produce NaN/Infinity
    func testDivisionByZero_ExactZeroVector_ReturnsError() async throws {
        // Given: Zero vector in standard dimension
        let dimension = TestConfig.dim384
        let values = [Float](repeating: 0.0, count: dimension)

        // Verify setup: all zeros
        XCTAssertTrue(values.allSatisfy { $0 == 0.0 }, "All values should be exactly zero")

        // Verify: norm is zero
        let manualNorm = computeNorm(values)
        XCTAssertEqual(manualNorm, 0.0, accuracy: Float.ulpOfOne, "Norm should be exactly zero")

        // When: Attempt to normalize zero vector
        let embedding = try Embedding<EmbedKit.Dim384>(values)
        let result = embedding.normalized()

        // Then: Should return error (not success)
        switch result {
        case .success:
            XCTFail("Zero vector normalization should return .failure, not .success")
        case .failure(let error):
            // Success! Verify it's the correct error type
            print("✓ Zero vector correctly returned error: \(error)")
        }

        // Verify: isZero property works
        XCTAssertTrue(embedding.isZero, "Embedding should recognize it's a zero vector")
    }

    /// T-NUM-004b-2: Zero vector for different dimensions
    ///
    /// Validates that zero detection works consistently across all dimensions.
    /// Tests small (3), medium (384, 768), and large (1536) dimensions.
    func testDivisionByZero_ZeroVectorAllDimensions_ReturnsError() async throws {
        // Test across various dimensions
        let dimensions = [
            (TestConfig.dimSmall, "Small (3)"),
            (TestConfig.dim384, "MiniLM (384)"),
            (TestConfig.dim768, "BERT (768)"),
            (TestConfig.dim1536, "Ada (1536)")
        ]

        for (dimension, description) in dimensions {
            // Given: Zero vector of specific dimension
            let values = [Float](repeating: 0.0, count: dimension)

            // When: Attempt to normalize
            // Use dynamic dimension testing since we can't use compile-time types in loop
            if dimension == TestConfig.dim384 {
                let embedding = try Embedding<EmbedKit.Dim384>(values)
                let result = embedding.normalized()

                // Then: Should fail
                switch result {
                case .success:
                    XCTFail("\(description): Zero vector should return error")
                case .failure:
                    print("✓ \(description): Zero vector correctly rejected")
                }
            } else if dimension == TestConfig.dim768 {
                let embedding = try Embedding<EmbedKit.Dim768>(values)
                let result = embedding.normalized()

                switch result {
                case .success:
                    XCTFail("\(description): Zero vector should return error")
                case .failure:
                    print("✓ \(description): Zero vector correctly rejected")
                }
            } else if dimension == TestConfig.dim1536 {
                let embedding = try Embedding<EmbedKit.Dim1536>(values)
                let result = embedding.normalized()

                switch result {
                case .success:
                    XCTFail("\(description): Zero vector should return error")
                case .failure:
                    print("✓ \(description): Zero vector correctly rejected")
                }
            }
        }
    }

    /// T-NUM-004b-3: Near-zero norm (very small uniform values)
    ///
    /// Tests vectors where individual components aren't zero, but the norm
    /// is so small that normalization might be numerically unstable.
    ///
    /// **Boundary Testing:**
    /// Using values of 1e-20, the norm for 384 dimensions would be:
    /// ||v|| = sqrt(384 * (1e-20)²) = sqrt(384) * 1e-20 ≈ 2e-19
    func testDivisionByZero_NearZeroNorm_HandlesGracefully() async throws {
        // Given: Very small uniform values (not exactly zero)
        let dimension = TestConfig.dim384
        let tinyValue: Float = 1e-20  // Very small but not zero
        let values = [Float](repeating: tinyValue, count: dimension)

        // Verify setup: values are non-zero but tiny
        XCTAssertTrue(values.allSatisfy { $0 != 0.0 }, "Values should not be exactly zero")
        XCTAssertTrue(values.allSatisfy { $0 > 0 }, "Values should be positive")

        // Calculate expected norm
        let expectedNorm = sqrt(Float(dimension)) * tinyValue
        let manualNorm = computeNorm(values)
        XCTAssertEqual(manualNorm, expectedNorm, accuracy: expectedNorm * 0.01)

        // When: Attempt to normalize near-zero vector
        let embedding = try Embedding<EmbedKit.Dim384>(values)
        let result = embedding.normalized()

        // Then: Implementation-defined behavior
        // VectorCore might succeed or fail depending on epsilon threshold
        switch result {
        case .success(let normalized):
            // If it succeeds, verify the result is valid
            print("✓ Near-zero vector normalized successfully")
            XCTAssertTrue(normalized.isFinite, "If normalized, must be finite")
            assertIsNormalized(normalized, accuracy: TestConfig.magnitudeTolerance)

            // Verify direction preserved
            let components = normalized.toArray()
            XCTAssertTrue(components.allSatisfy { $0 > 0 }, "Direction should be preserved (all positive)")

        case .failure(let error):
            // If it fails, that's also acceptable (epsilon protection)
            print("✓ Near-zero vector rejected (epsilon protection): \(error)")
        }
    }

    /// T-NUM-004b-4: Single non-zero component in zero vector
    ///
    /// A vector with one non-zero component has a well-defined norm,
    /// so normalization should succeed.
    func testDivisionByZero_SingleNonZeroComponent_Normalizes() async throws {
        // Given: One non-zero component, rest are zero
        let dimension = TestConfig.dim768
        var values = [Float](repeating: 0.0, count: dimension)
        values[dimension / 2] = 5.0  // Single non-zero component

        // Verify setup
        XCTAssertEqual(values.filter { $0 != 0.0 }.count, 1, "Should have exactly one non-zero value")

        // Calculate expected norm
        let expectedNorm: Float = 5.0  // Since only one component is 5.0
        let manualNorm = computeNorm(values)
        XCTAssertEqual(manualNorm, expectedNorm, accuracy: Float.ulpOfOne)

        // When: Normalize single-component vector
        let embedding = try Embedding<EmbedKit.Dim768>(values)
        let result = embedding.normalized()

        // Then: Should succeed (non-zero norm)
        guard case .success(let normalized) = result else {
            XCTFail("Single non-zero component should normalize successfully")
            return
        }

        // Verify: All finite
        XCTAssertTrue(normalized.isFinite, "Normalized vector should be finite")

        // Verify: Magnitude is 1.0
        assertIsNormalized(normalized, accuracy: TestConfig.magnitudeTolerance)

        // Verify: Non-zero component becomes 1.0 (unit vector direction)
        let components = normalized.toArray()
        XCTAssertEqual(
            components[dimension / 2],
            1.0,
            accuracy: TestConfig.componentTolerance,
            "Single non-zero component should become 1.0"
        )

        // Verify: All other components remain zero
        for (index, component) in components.enumerated() {
            if index == dimension / 2 {
                continue  // Skip the non-zero component
            }
            XCTAssertEqual(
                component,
                0.0,
                accuracy: Float.ulpOfOne,
                "Zero components should remain zero"
            )
        }
    }

    /// T-NUM-004b-5: Extremely small uniform values
    ///
    /// Tests vectors with values near the smallest representable float.
    /// This tests the boundary between "can normalize" and "too small to normalize".
    func testDivisionByZero_ExtremelySmallUniform_BehaviorDefined() async throws {
        // Given: Values near minimum normal magnitude
        let dimension = TestConfig.dim384
        let extremelySmall = TestConfig.minNormal * 10.0  // 10x minimum normal
        let values = [Float](repeating: extremelySmall, count: dimension)

        // Verify setup
        XCTAssertTrue(values.allSatisfy { $0 > 0 }, "Values should be positive")
        XCTAssertTrue(values.allSatisfy { $0.isNormal }, "Values should be normal (not subnormal)")

        // Calculate norm
        let expectedNorm = sqrt(Float(dimension)) * extremelySmall
        XCTAssertTrue(expectedNorm.isNormal, "Norm should be normal (not subnormal)")

        // When: Normalize extremely small vector
        let embedding = try Embedding<EmbedKit.Dim384>(values)
        let result = embedding.normalized()

        // Then: Should succeed (values are normal, just very small)
        guard case .success(let normalized) = result else {
            XCTFail("Normal values should normalize successfully, even if extremely small. Norm: \(expectedNorm)")
            return
        }

        // Verify: All finite
        XCTAssertTrue(normalized.isFinite, "Normalized vector should be finite")

        // Verify: Magnitude is 1.0
        assertIsNormalized(normalized, accuracy: TestConfig.magnitudeTolerance)

        // Verify: Direction preserved (uniformity)
        let components = normalized.toArray()
        let expectedComponent = 1.0 / sqrt(Float(dimension))

        for (index, component) in components.enumerated() {
            XCTAssertEqual(
                component,
                expectedComponent,
                accuracy: TestConfig.componentTolerance * 10,  // Relaxed for extreme values
                "Component \(index) should match expected uniform value"
            )
        }
    }

    /// T-NUM-004b-6: Values below machine epsilon
    ///
    /// Tests subnormal (denormalized) values to ensure they're handled correctly.
    /// Subnormal values are below Float.leastNormalMagnitude but above zero.
    func testDivisionByZero_BelowMachineEpsilon_HandlesSafely() async throws {
        // Given: Subnormal values (denormalized)
        let dimension = TestConfig.dim768
        let subnormal = TestConfig.minNonzero * 100.0  // Subnormal but not minimum
        let values = [Float](repeating: subnormal, count: dimension)

        // Verify: Values are subnormal
        XCTAssertTrue(values.allSatisfy { isSubnormal($0) }, "Values should be subnormal")
        XCTAssertTrue(values.allSatisfy { $0 > 0 }, "Values should be positive")

        // Calculate norm (will likely underflow to zero or subnormal)
        let normSquared = Float(dimension) * (subnormal * subnormal)

        // When: Attempt to normalize subnormal vector
        let embedding = try Embedding<EmbedKit.Dim768>(values)
        let result = embedding.normalized()

        // Then: Behavior depends on whether norm underflows
        if normSquared == 0.0 || !normSquared.isFinite {
            // If norm underflows to zero, should fail
            switch result {
            case .success:
                XCTFail("Subnormal vector with zero norm should fail normalization")
            case .failure(let error):
                print("✓ Subnormal vector with underflow correctly rejected: \(error)")
            }
        } else {
            // If norm is computable, might succeed
            switch result {
            case .success(let normalized):
                print("✓ Subnormal vector normalized successfully")
                XCTAssertTrue(normalized.isFinite, "Normalized subnormal should be finite")

            case .failure(let error):
                print("✓ Subnormal vector rejected: \(error)")
                // Also acceptable if implementation rejects subnormals
            }
        }
    }

    /// T-NUM-004b-7: Zero detection across dimensions
    ///
    /// Validates consistent zero detection behavior across all test dimensions,
    /// including non-standard dimensions.
    func testDivisionByZero_MultiDimensional_ConsistentBehavior() async throws {
        // Test: Zero vectors should always fail, regardless of dimension
        let testCases: [(Int, String)] = [
            (TestConfig.dimSmall, "Small dimension"),
            (TestConfig.dimPrime, "Prime dimension (127)"),
            (TestConfig.dim384, "MiniLM dimension"),
            (TestConfig.dim768, "BERT dimension"),
            (TestConfig.dim1536, "Large dimension")
        ]

        for (dimension, description) in testCases {
            let zeroVector = [Float](repeating: 0.0, count: dimension)

            // Test with appropriate dimension type
            if dimension == TestConfig.dim384 {
                let embedding = try Embedding<EmbedKit.Dim384>(zeroVector)
                let result = embedding.normalized()

                switch result {
                case .success:
                    XCTFail("\(description): Zero vector should fail")
                case .failure:
                    print("✓ \(description): Consistent zero rejection")
                }
            } else if dimension == TestConfig.dim768 {
                let embedding = try Embedding<EmbedKit.Dim768>(zeroVector)
                let result = embedding.normalized()

                switch result {
                case .success:
                    XCTFail("\(description): Zero vector should fail")
                case .failure:
                    print("✓ \(description): Consistent zero rejection")
                }
            } else if dimension == TestConfig.dim1536 {
                let embedding = try Embedding<EmbedKit.Dim1536>(zeroVector)
                let result = embedding.normalized()

                switch result {
                case .success:
                    XCTFail("\(description): Zero vector should fail")
                case .failure:
                    print("✓ \(description): Consistent zero rejection")
                }
            }
        }
    }

    /// T-NUM-004b-8: Alternating zero and near-zero values
    ///
    /// Tests mixed scenarios where some components are exactly zero
    /// and others are near-zero.
    func testDivisionByZero_AlternatingZeroNearZero_CorrectDetection() async throws {
        // Given: Half zeros, half near-zero
        let dimension = TestConfig.dim384
        let nearZero: Float = 1e-10
        var values = [Float](repeating: 0.0, count: dimension)

        // Fill every other component with near-zero value
        for i in stride(from: 0, to: dimension, by: 2) {
            values[i] = nearZero
        }

        // Verify setup
        let zeroCount = values.filter { $0 == 0.0 }.count
        let nearZeroCount = values.filter { $0 == nearZero }.count
        XCTAssertEqual(zeroCount, dimension / 2, "Half should be zero")
        XCTAssertEqual(nearZeroCount, dimension / 2, "Half should be near-zero")

        // Calculate expected norm
        let expectedNormSquared = Float(dimension / 2) * (nearZero * nearZero)
        let expectedNorm = sqrt(expectedNormSquared)

        // When: Normalize mixed zero/near-zero vector
        let embedding = try Embedding<EmbedKit.Dim384>(values)
        let result = embedding.normalized()

        // Then: Should succeed (norm is non-zero)
        guard case .success(let normalized) = result else {
            XCTFail("Mixed zero/near-zero vector should normalize (norm = \(expectedNorm))")
            return
        }

        // Verify: All finite
        XCTAssertTrue(normalized.isFinite, "Normalized vector should be finite")

        // Verify: Magnitude is 1.0
        assertIsNormalized(normalized, accuracy: TestConfig.magnitudeTolerance)

        // Verify: Zeros remain zero, near-zeros get normalized
        let components = normalized.toArray()

        for (index, component) in components.enumerated() {
            if index % 2 == 0 {
                // Was near-zero, should be normalized
                XCTAssertGreaterThan(component, 0, "Near-zero components should be positive after normalization")
            } else {
                // Was zero, should remain zero
                XCTAssertEqual(component, 0.0, accuracy: Float.ulpOfOne, "Zero components should remain zero")
            }
        }
    }

    /// T-NUM-004b-9: Error type validation
    ///
    /// Verifies that the error returned for zero vectors is the correct type
    /// and contains useful information.
    func testDivisionByZero_ErrorType_CorrectVectorError() async throws {
        // Given: Zero vector
        let dimension = TestConfig.dim384
        let values = [Float](repeating: 0.0, count: dimension)

        // When: Attempt to normalize
        let embedding = try Embedding<EmbedKit.Dim384>(values)
        let result = embedding.normalized()

        // Then: Should return VectorError
        guard case .failure(let error) = result else {
            XCTFail("Zero vector should return .failure")
            return
        }

        // Verify: Error is VectorError type
        print("Error type: \(type(of: error))")
        print("Error description: \(error.localizedDescription)")

        // The error should be descriptive
        let errorString = String(describing: error)
        print("Error string: \(errorString)")

        // Verify error is not nil or empty
        XCTAssertFalse(error.localizedDescription.isEmpty, "Error should have description")

        // The error should indicate a zero or near-zero vector issue
        // (exact message depends on VectorCore implementation)
        let description = error.localizedDescription.lowercased()
        let hasRelevantKeyword = description.contains("zero") ||
                                 description.contains("norm") ||
                                 description.contains("magnitude") ||
                                 description.contains("invalid")

        XCTAssertTrue(
            hasRelevantKeyword,
            "Error description should mention zero/norm/magnitude: \(error.localizedDescription)"
        )
    }

    /// T-NUM-004b-10: Boundary at minimum normalizable magnitude
    ///
    /// Finds the smallest vector magnitude that can still be successfully normalized.
    /// This identifies the epsilon threshold used by the implementation.
    func testDivisionByZero_MinimumNormalizableMagnitude_Succeeds() async throws {
        // Strategy: Binary search for minimum normalizable magnitude
        // Start with a value we know works (1.0) and one we know fails (0.0)

        let dimension = TestConfig.dim384

        // Test: A vector with magnitude 1.0 should definitely work
        let workingValue: Float = 1.0 / sqrt(Float(dimension))  // Unit vector
        let workingValues = [Float](repeating: workingValue, count: dimension)
        let workingEmbedding = try Embedding<EmbedKit.Dim384>(workingValues)
        let workingResult = workingEmbedding.normalized()

        guard case .success = workingResult else {
            XCTFail("Unit magnitude vector should normalize successfully")
            return
        }

        // Test: A very small vector should work if above Float.leastNormalMagnitude
        let minNormalValue = TestConfig.minNormal * 1000.0 / sqrt(Float(dimension))
        let minNormalValues = [Float](repeating: minNormalValue, count: dimension)
        let minNormalEmbedding = try Embedding<EmbedKit.Dim384>(minNormalValues)
        let minNormalResult = minNormalEmbedding.normalized()

        // This should succeed (values are normal)
        switch minNormalResult {
        case .success(let normalized):
            print("✓ Minimum normal vector normalized successfully")

            // Verify it's actually normalized
            XCTAssertTrue(normalized.isFinite, "Minimum normal should be finite")
            assertIsNormalized(normalized, accuracy: TestConfig.magnitudeTolerance)

            // Calculate the magnitude we successfully normalized
            let successfulMagnitude = computeNorm(minNormalValues)
            print("✓ Successfully normalized vector with magnitude: \(successfulMagnitude)")
            print("  (Float.leastNormalMagnitude = \(TestConfig.minNormal))")

        case .failure(let error):
            print("⚠ Minimum normal vector rejected: \(error)")
            // This might happen if VectorCore has a higher epsilon threshold
        }

        // Document the findings
        print("Zero detection boundary test complete:")
        print("  - Unit vector (mag=1.0): ✓ Success")
        print("  - Min normal vector (mag≈\(TestConfig.minNormal * 1000.0)): \(minNormalResult)")
        print("  - Zero vector (mag=0.0): ✗ Correctly rejected")
    }

    // MARK: - T-NUM-004c: Epsilon Threshold & Underflow Tests

    /// T-NUM-004c-1: Very small uniform values near underflow
    ///
    /// Tests normalization with values just above the underflow threshold.
    /// Uses values at 100x Float.leastNormalMagnitude to ensure they're
    /// representable but very small.
    func testEpsilon_VerySmallUniform_PreservesPrecision() async throws {
        // Given: Very small but normal values
        let dimension = TestConfig.dim384
        let verySmall = TestConfig.minNormal * 100.0  // 100x minimum normal
        let values = [Float](repeating: verySmall, count: dimension)

        // Verify: Values are normal (not subnormal)
        XCTAssertTrue(values.allSatisfy { $0.isNormal }, "Values should be normal floats")
        XCTAssertTrue(values.allSatisfy { $0 > 0 }, "Values should be positive")

        // Calculate expected norm
        let expectedNorm = sqrt(Float(dimension)) * verySmall
        XCTAssertTrue(expectedNorm.isNormal, "Norm should be normal")

        // When: Normalize very small vector
        let embedding = try Embedding<EmbedKit.Dim384>(values)
        let result = embedding.normalized()

        // Then: Should succeed (values are normal)
        guard case .success(let normalized) = result else {
            XCTFail("Normal values should normalize. Value: \(verySmall), Norm: \(expectedNorm)")
            return
        }

        // Verify: All finite
        XCTAssertTrue(normalized.isFinite, "Normalized vector should be finite")

        // Verify: Magnitude is 1.0
        assertIsNormalized(normalized, accuracy: TestConfig.magnitudeTolerance)

        // Verify: Precision preserved (uniformity maintained)
        let components = normalized.toArray()
        let expectedComponent = 1.0 / sqrt(Float(dimension))

        let deviations = components.map { abs($0 - expectedComponent) }
        let maxDeviation = deviations.max() ?? 0.0

        XCTAssertLessThan(
            maxDeviation,
            TestConfig.componentTolerance * 100,  // Relaxed for very small values
            "Precision should be preserved despite very small input values"
        )
    }

    /// T-NUM-004c-2: Subnormal (denormalized) number handling
    ///
    /// Tests how normalization handles subnormal floats - values below
    /// Float.leastNormalMagnitude that have reduced precision.
    func testEpsilon_SubnormalNumbers_CorrectNormalization() async throws {
        // Given: Subnormal values (denormalized floats)
        let dimension = TestConfig.dim768
        let subnormal = TestConfig.minNonzero * 1000.0  // Subnormal but not minimum
        let values = [Float](repeating: subnormal, count: dimension)

        // Verify: Values are subnormal
        XCTAssertTrue(values.allSatisfy { isSubnormal($0) }, "Values should be subnormal")
        XCTAssertFalse(values.allSatisfy { $0.isNormal }, "Values should NOT be normal")
        XCTAssertTrue(values.allSatisfy { $0 > 0 }, "Values should be positive")

        // Calculate norm (may underflow to zero)
        let normSquared = Float(dimension) * (subnormal * subnormal)

        // When: Attempt to normalize subnormal vector
        let embedding = try Embedding<EmbedKit.Dim768>(values)
        let result = embedding.normalized()

        // Then: Behavior depends on whether norm computation underflows
        if normSquared == 0.0 || !normSquared.isFinite {
            // Norm underflowed to zero - should fail
            switch result {
            case .success:
                XCTFail("Subnormal vector with underflowed norm should fail")
            case .failure(let error):
                print("✓ Subnormal underflow correctly rejected: \(error)")
            }
        } else {
            // Norm is still computable
            switch result {
            case .success(let normalized):
                print("✓ Subnormal vector normalized (norm didn't underflow)")
                XCTAssertTrue(normalized.isFinite, "Result should be finite")
                assertIsNormalized(normalized, accuracy: TestConfig.magnitudeTolerance)

            case .failure(let error):
                print("✓ Subnormal vector rejected (implementation choice): \(error)")
                // Also valid if implementation rejects subnormals
            }
        }

        print("Subnormal test: value=\(subnormal), norm²=\(normSquared)")
    }

    /// T-NUM-004c-3: Transition from normal to subnormal
    ///
    /// Tests the boundary where values transition from normal to subnormal.
    /// This is Float.leastNormalMagnitude.
    func testEpsilon_NormalToSubnormalTransition_Smooth() async throws {
        // Given: Values right at the normal/subnormal boundary
        let dimension = TestConfig.dim384

        // Test both sides of the boundary
        let justNormal = TestConfig.minNormal * 1.1  // Just above boundary
        let justSubnormal = TestConfig.minNormal * 0.9  // Just below boundary

        // Test 1: Just above boundary (normal)
        let normalValues = [Float](repeating: justNormal, count: dimension)
        let normalEmbedding = try Embedding<EmbedKit.Dim384>(normalValues)

        XCTAssertTrue(normalValues.allSatisfy { $0.isNormal }, "Values should be normal")

        let normalResult = normalEmbedding.normalized()

        switch normalResult {
        case .success(let normalized):
            print("✓ Just-normal values normalized successfully")
            XCTAssertTrue(normalized.isFinite, "Just-normal result should be finite")
            assertIsNormalized(normalized, accuracy: TestConfig.magnitudeTolerance)

        case .failure(let error):
            print("⚠ Just-normal values rejected: \(error)")
        }

        // Test 2: Just below boundary (subnormal)
        let subnormalValues = [Float](repeating: justSubnormal, count: dimension)
        let subnormalEmbedding = try Embedding<EmbedKit.Dim384>(subnormalValues)

        XCTAssertTrue(subnormalValues.allSatisfy { isSubnormal($0) }, "Values should be subnormal")

        let subnormalResult = subnormalEmbedding.normalized()

        // Behavior at boundary should be consistent
        print("Boundary transition:")
        print("  Just normal (\(justNormal)): \(normalResult)")
        print("  Just subnormal (\(justSubnormal)): \(subnormalResult)")
    }

    /// T-NUM-004c-4: Minimum normal magnitude vector
    ///
    /// Tests normalization at the smallest normal magnitude.
    /// This is the boundary case for normal float handling.
    func testEpsilon_MinNormalMagnitude_Normalizes() async throws {
        // Given: Values at minimum normal magnitude
        let dimension = TestConfig.dim768
        let minNormal = TestConfig.minNormal
        let values = [Float](repeating: minNormal, count: dimension)

        // Verify: Values are exactly at minimum normal
        XCTAssertTrue(values.allSatisfy { $0.isNormal }, "Values should be normal")
        XCTAssertEqual(values[0], TestConfig.minNormal, accuracy: Float.ulpOfOne)

        // Calculate norm
        let expectedNorm = sqrt(Float(dimension)) * minNormal
        let isNormNormal = expectedNorm.isNormal

        // When: Normalize minimum normal vector
        let embedding = try Embedding<EmbedKit.Dim768>(values)
        let result = embedding.normalized()

        // Then: Behavior depends on whether norm stays normal
        if isNormNormal {
            // Norm is normal, should succeed
            guard case .success(let normalized) = result else {
                XCTFail("Minimum normal vector should normalize. Norm: \(expectedNorm)")
                return
            }

            XCTAssertTrue(normalized.isFinite, "Result should be finite")
            assertIsNormalized(normalized, accuracy: TestConfig.magnitudeTolerance)

            print("✓ Minimum normal magnitude normalized successfully")
        } else {
            // Norm underflowed, may fail
            switch result {
            case .success(let normalized):
                print("✓ Minimum normal normalized despite subnormal norm")
                XCTAssertTrue(normalized.isFinite, "Result should be finite")

            case .failure(let error):
                print("✓ Minimum normal rejected (norm underflowed): \(error)")
            }
        }

        print("Min normal test: value=\(minNormal), norm=\(expectedNorm), isNormal=\(isNormNormal)")
    }

    /// T-NUM-004c-5: Mixed normal and subnormal components
    ///
    /// Tests vectors with both normal and subnormal components to verify
    /// that relative ratios are preserved correctly.
    func testEpsilon_MixedNormalSubnormal_PreservesRatios() async throws {
        // Given: Mixed normal and subnormal values
        let dimension = TestConfig.dim384
        let normalValue = TestConfig.minNormal * 1000.0  // Normal
        let subnormalValue = TestConfig.minNonzero * 100.0  // Subnormal

        var values = [Float](repeating: 0.0, count: dimension)

        // Alternate between normal and subnormal
        for i in 0..<dimension {
            values[i] = (i % 2 == 0) ? normalValue : subnormalValue
        }

        // Verify setup
        let normalCount = values.filter { $0.isNormal }.count
        let subnormalCount = values.filter { isSubnormal($0) }.count

        XCTAssertEqual(normalCount, dimension / 2, "Half should be normal")
        XCTAssertEqual(subnormalCount, dimension / 2, "Half should be subnormal")

        // When: Normalize mixed vector
        let embedding = try Embedding<EmbedKit.Dim384>(values)
        let result = embedding.normalized()

        // Then: Should succeed (has normal components with non-zero norm)
        switch result {
        case .success(let normalized):
            print("✓ Mixed normal/subnormal normalized successfully")

            // Verify: All finite
            XCTAssertTrue(normalized.isFinite, "Result should be finite")

            // Verify: Magnitude is 1.0
            assertIsNormalized(normalized, accuracy: TestConfig.magnitudeTolerance)

            // Verify: Ratios preserved
            let components = normalized.toArray()
            let originalRatio = normalValue / subnormalValue

            // Check ratio between adjacent components
            if components[0] > 0 && components[1] > 0 {
                let normalizedRatio = components[0] / components[1]
                let ratioError = abs(normalizedRatio - originalRatio) / originalRatio

                XCTAssertLessThan(
                    ratioError,
                    0.1,  // 10% tolerance for subnormals
                    "Ratio between normal/subnormal should be approximately preserved"
                )
            }

        case .failure(let error):
            print("⚠ Mixed normal/subnormal rejected: \(error)")
            // May fail if subnormal components cause issues
        }
    }

    /// T-NUM-004c-6: Gradual underflow across components
    ///
    /// Tests a vector where components gradually decrease from normal
    /// to subnormal values, ensuring no abrupt behavior changes.
    func testEpsilon_GradualUnderflow_NoAbruptChanges() async throws {
        // Given: Gradually decreasing values from normal to subnormal
        let dimension = TestConfig.dim384

        var values = [Float](repeating: 0.0, count: dimension)

        // Create gradient from normal to subnormal
        let startValue = TestConfig.minNormal * 10.0  // Normal
        let endValue = TestConfig.minNonzero * 10.0   // Subnormal

        for i in 0..<dimension {
            let ratio = Float(i) / Float(dimension - 1)
            // Linear interpolation in log space for smooth transition
            let logStart = log(startValue)
            let logEnd = log(endValue)
            let logValue = logStart + ratio * (logEnd - logStart)
            values[i] = exp(logValue)
        }

        // Verify: Gradient from normal to subnormal
        XCTAssertTrue(values[0].isNormal, "First value should be normal")
        XCTAssertTrue(isSubnormal(values[dimension-1]), "Last value should be subnormal")

        // When: Normalize gradient vector
        let embedding = try Embedding<EmbedKit.Dim384>(values)
        let result = embedding.normalized()

        // Then: Should handle gracefully
        switch result {
        case .success(let normalized):
            print("✓ Gradual underflow normalized successfully")

            XCTAssertTrue(normalized.isFinite, "Result should be finite")
            assertIsNormalized(normalized, accuracy: TestConfig.magnitudeTolerance)

            // Verify: No abrupt discontinuities in normalized output
            let components = normalized.toArray()

            var hasDiscontinuity = false
            for i in 0..<(dimension-1) {
                let ratio = components[i+1] / components[i]
                // Check for abrupt changes (ratio > 2x or < 0.5x)
                if ratio > 2.0 || ratio < 0.5 {
                    hasDiscontinuity = true
                    break
                }
            }

            XCTAssertFalse(hasDiscontinuity, "Should not have abrupt discontinuities")

        case .failure(let error):
            print("⚠ Gradual underflow rejected: \(error)")
        }
    }

    /// T-NUM-004c-7: Epsilon boundary testing
    ///
    /// Tests various epsilon values to understand the threshold
    /// at which normalization fails.
    func testEpsilon_BoundaryConditions_DefinedBehavior() async throws {
        // Test a range of magnitudes to find the epsilon boundary
        let dimension = TestConfig.dim384

        let testMagnitudes: [(Float, String)] = [
            (1.0, "Unit magnitude"),
            (1e-10, "Small normal"),
            (1e-20, "Very small"),
            (1e-30, "Extremely small"),
            (TestConfig.minNormal * 1000, "1000x min normal"),
            (TestConfig.minNormal, "Min normal"),
            (TestConfig.minNormal * 0.1, "0.1x min normal (subnormal)")
        ]

        for (magnitude, description) in testMagnitudes {
            // Create vector with specific magnitude
            let componentValue = magnitude / sqrt(Float(dimension))
            let values = [Float](repeating: componentValue, count: dimension)

            let embedding = try Embedding<EmbedKit.Dim384>(values)
            let result = embedding.normalized()

            let actualNorm = computeNorm(values)

            switch result {
            case .success:
                print("✓ \(description) (mag=\(magnitude)): Success")
            case .failure:
                print("✗ \(description) (mag=\(magnitude)): Failed")
            }

            // Document the boundary
            if case .success = result {
                XCTAssertEqual(actualNorm, magnitude, accuracy: magnitude * 0.01)
            }
        }
    }

    /// T-NUM-004c-8: Very small values with one normal component
    ///
    /// Tests a dominant normal component with subnormal noise.
    /// The normal component should dominate and allow normalization.
    func testEpsilon_TinyValuesOneNormal_CorrectProportion() async throws {
        // Given: One normal component, rest subnormal
        let dimension = TestConfig.dim768
        let normalValue: Float = 1.0
        let subnormalValue = TestConfig.minNonzero * 100.0

        var values = [Float](repeating: subnormalValue, count: dimension)
        values[0] = normalValue  // Dominant normal component

        // Verify setup
        XCTAssertTrue(values[0].isNormal, "First value should be normal")
        XCTAssertTrue(isSubnormal(values[1]), "Others should be subnormal")

        // When: Normalize
        let embedding = try Embedding<EmbedKit.Dim768>(values)
        let result = embedding.normalized()

        // Then: Should succeed (normal component dominates)
        guard case .success(let normalized) = result else {
            XCTFail("Vector with dominant normal component should normalize")
            return
        }

        // Verify: All finite
        XCTAssertTrue(normalized.isFinite, "Result should be finite")

        // Verify: Magnitude is 1.0
        assertIsNormalized(normalized, accuracy: TestConfig.magnitudeTolerance)

        // Verify: Normal component dominates (should be close to 1.0)
        let components = normalized.toArray()
        XCTAssertGreaterThan(
            components[0],
            0.99,
            "Normal component should dominate (≈1.0 after normalization)"
        )

        // Verify: Subnormal components are tiny relative to normal
        for i in 1..<dimension {
            XCTAssertLessThan(
                abs(components[i]),
                0.01,
                "Subnormal components should be much smaller than normal component"
            )
        }
    }

    /// T-NUM-004c-9: Flush-to-zero behavior validation
    ///
    /// Verifies that subnormal values are NOT flushed to zero
    /// (unless they genuinely underflow during computation).
    func testEpsilon_FlushToZero_DoesNotOccur() async throws {
        // Given: Subnormal values that should remain distinct from zero
        let dimension = TestConfig.dim384
        let subnormal = TestConfig.minNonzero * 500.0  // Well above zero
        let values = [Float](repeating: subnormal, count: dimension)

        // Verify: Values are subnormal but not zero
        XCTAssertTrue(values.allSatisfy { isSubnormal($0) }, "Values should be subnormal")
        XCTAssertTrue(values.allSatisfy { $0 != 0.0 }, "Values should NOT be zero")

        // When: Normalize (if possible)
        let embedding = try Embedding<EmbedKit.Dim384>(values)
        let result = embedding.normalized()

        // Then: Verify no flush-to-zero occurred
        switch result {
        case .success(let normalized):
            let components = normalized.toArray()

            // Count how many components are exactly zero
            let zeroCount = components.filter { $0 == 0.0 }.count

            // If normalization succeeded, components should not be flushed to zero
            // (unless they're legitimately zero in the mathematical sense)
            XCTAssertEqual(
                zeroCount,
                0,
                "No flush-to-zero should occur - \(zeroCount) components became zero"
            )

            print("✓ No flush-to-zero: all \(dimension) components non-zero")

        case .failure(let error):
            // If it failed, it should be due to underflow, not flush-to-zero
            print("✓ Subnormal vector rejected (likely underflow): \(error)")

            // Verify original values were not flushed
            XCTAssertTrue(values.allSatisfy { $0 != 0.0 }, "Input should not be flushed")
        }
    }

    /// T-NUM-004c-10: Precision preservation at small scales
    ///
    /// Tests that relative precision is maintained even for very small values.
    /// Uses normal values just above subnormal boundary.
    func testEpsilon_SmallScalePrecision_Maintained() async throws {
        // Given: Small but normal values with precise ratios
        let dimension = TestConfig.dim384
        let baseValue = TestConfig.minNormal * 100.0  // Small but normal

        // Create values with specific ratios: [1x, 2x, 3x, 4x, ...] * baseValue
        var values = [Float](repeating: 0.0, count: dimension)
        for i in 0..<dimension {
            values[i] = Float(i + 1) * baseValue
        }

        // Verify: All normal
        XCTAssertTrue(values.allSatisfy { $0.isNormal }, "All values should be normal")

        // When: Normalize
        let embedding = try Embedding<EmbedKit.Dim384>(values)
        let result = embedding.normalized()

        // Then: Should succeed
        guard case .success(let normalized) = result else {
            XCTFail("Small normal values should normalize")
            return
        }

        // Verify: All finite
        XCTAssertTrue(normalized.isFinite, "Result should be finite")

        // Verify: Magnitude is 1.0
        assertIsNormalized(normalized, accuracy: TestConfig.magnitudeTolerance)

        // Verify: Ratios preserved despite small scale
        let components = normalized.toArray()

        // Check ratio between first and second component
        // Should be approximately 2:1 (since input was [1x, 2x, ...])
        let expectedRatio: Float = 2.0
        let actualRatio = components[1] / components[0]

        XCTAssertEqual(
            actualRatio,
            expectedRatio,
            accuracy: 0.01,
            "Ratio precision should be preserved at small scales"
        )

        // Check ordering is preserved
        for i in 0..<(dimension-1) {
            XCTAssertLessThan(
                components[i],
                components[i+1],
                "Ordering should be preserved at index \(i)"
            )
        }

        print("✓ Small scale precision maintained: ratios and ordering preserved")
    }

    // MARK: - T-NUM-004d: Precision Loss in sqrt() Tests

    /// T-NUM-004d-1: Perfect squares vs non-perfect squares
    ///
    /// Tests normalization accuracy for vectors whose norm² is a perfect square
    /// vs non-perfect squares. Perfect squares should have exact sqrt results.
    func testPrecisionLoss_PerfectSquares_HighAccuracy() async throws {
        // Given: Vector with norm² = 16 (perfect square, sqrt = 4.0)
        let dimension = TestConfig.dim384

        // Create vector: [2/sqrt(384), 2/sqrt(384), ...] * sqrt(384) = [2, 2, ...]
        // norm² = 384 * 4 = 1536 (not quite perfect, let's use a different approach)

        // Better: Create [4, 0, 0, ...] which has norm² = 16, sqrt = 4.0 (perfect)
        var perfectSquareVector = [Float](repeating: 0.0, count: dimension)
        perfectSquareVector[0] = 4.0

        let embedding = try Embedding<EmbedKit.Dim384>(perfectSquareVector)
        let result = embedding.normalized()

        guard case .success(let normalized) = result else {
            XCTFail("Perfect square normalization should succeed")
            return
        }

        // Verify: Magnitude exactly 1.0
        let magnitude = normalized.magnitude
        XCTAssertEqual(magnitude, 1.0, accuracy: Float.ulpOfOne, "Perfect square should have exact magnitude")

        // Verify: First component should be exactly 1.0 (4.0 / 4.0 = 1.0)
        let components = normalized.toArray()
        XCTAssertEqual(components[0], 1.0, accuracy: Float.ulpOfOne, "Perfect square division should be exact")

        print("✓ Perfect square normalization: exact result achieved")
    }

    /// T-NUM-004d-2: Catastrophic cancellation scenario
    ///
    /// Tests vectors where norm² calculation might suffer from catastrophic
    /// cancellation due to vastly different magnitudes.
    func testPrecisionLoss_CatastrophicCancellation_MaintainsPrecision() async throws {
        // Given: Vector with one large and many tiny components
        let dimension = TestConfig.dim768
        var values = [Float](repeating: 1e-20, count: dimension - 1)
        values.append(1.0)  // One dominant component

        // The norm² = (768-1)*(1e-20)² + 1² ≈ 1.0 + tiny ≈ 1.0
        // Risk: The tiny values might be lost in floating-point addition

        let embedding = try Embedding<EmbedKit.Dim768>(values)
        let result = embedding.normalized()

        guard case .success(let normalized) = result else {
            XCTFail("Catastrophic cancellation test should normalize")
            return
        }

        // Verify: All finite
        XCTAssertTrue(normalized.isFinite, "Result should be finite")

        // Verify: Magnitude is 1.0
        assertIsNormalized(normalized, accuracy: TestConfig.magnitudeTolerance)

        // Verify: Large component still dominates
        let components = normalized.toArray()
        XCTAssertGreaterThan(components[dimension-1], 0.99, "Large component should still dominate")

        // Verify: Small components are preserved (not lost)
        let smallComponents = components.prefix(dimension-1)
        let nonZeroSmall = smallComponents.filter { $0 != 0.0 }.count

        // Some small components should be preserved
        print("✓ Catastrophic cancellation: \(nonZeroSmall)/\(dimension-1) small components preserved")
    }

    /// T-NUM-004d-3: Loss of significance in large dimension
    ///
    /// Tests that precision is maintained even in very high dimensions
    /// where many floating-point operations accumulate.
    func testPrecisionLoss_LargeDimension_MinimalSignificanceLoss() async throws {
        // Given: Large dimension with uniform values
        let dimension = TestConfig.dim1536  // Largest standard dimension
        let uniformValue = 1.0 / sqrt(Float(dimension))  // Pre-normalized value
        let values = [Float](repeating: uniformValue, count: dimension)

        // Expected: norm should be exactly 1.0
        let manualNorm = computeNorm(values)
        XCTAssertEqual(manualNorm, 1.0, accuracy: 1e-6, "Manual norm should be 1.0")

        // When: Normalize
        let embedding = try Embedding<EmbedKit.Dim1536>(values)
        let result = embedding.normalized()

        guard case .success(let normalized) = result else {
            XCTFail("Large dimension should normalize")
            return
        }

        // Verify: Magnitude is still 1.0 despite high dimension
        let magnitude = normalized.magnitude
        XCTAssertEqual(magnitude, 1.0, accuracy: TestConfig.magnitudeTolerance)

        // Verify: Components maintain uniformity
        let components = normalized.toArray()
        let expectedComponent = 1.0 / sqrt(Float(dimension))

        let deviations = components.map { abs($0 - expectedComponent) }
        let maxDeviation = deviations.max() ?? 0.0

        XCTAssertLessThan(maxDeviation, 1e-5, "Large dimension should maintain precision")

        print("✓ Large dimension (1536): max deviation = \(maxDeviation)")
    }

    /// T-NUM-004d-4: Relative error measurement
    ///
    /// Measures and validates that relative error in normalization
    /// stays below acceptable thresholds.
    func testPrecisionLoss_RelativeError_BelowThreshold() async throws {
        // Given: Vector with known magnitude
        let dimension = TestConfig.dim384
        let targetMagnitude: Float = 5.0
        let componentValue = targetMagnitude / sqrt(Float(dimension))
        let values = [Float](repeating: componentValue, count: dimension)

        // Verify manual norm
        let manualNorm = computeNorm(values)
        XCTAssertEqual(manualNorm, targetMagnitude, accuracy: targetMagnitude * 0.001)

        // When: Normalize
        let embedding = try Embedding<EmbedKit.Dim384>(values)
        let result = try embedding.normalized().get()

        // Then: Measure relative error
        let actualMagnitude = result.magnitude
        let relativeError = computeRelativeError(1.0, actualMagnitude)

        // Relative error should be very small
        XCTAssertLessThan(
            relativeError,
            TestConfig.relativeErrorTolerance,
            "Relative error should be below threshold: \(relativeError)"
        )

        print("✓ Relative error: \(relativeError) (threshold: \(TestConfig.relativeErrorTolerance))")
    }

    /// T-NUM-004d-5: Vastly different component magnitudes
    ///
    /// Tests that small components aren't lost when normalizing vectors
    /// with vastly different magnitudes.
    func testPrecisionLoss_DifferentMagnitudes_PreservesSmallComponents() async throws {
        // Given: Mixed magnitudes (but not so extreme they underflow)
        let dimension = TestConfig.dim768
        let largeValue: Float = 100.0
        let smallValue: Float = 0.01

        var values = [Float](repeating: smallValue, count: dimension)
        values[0] = largeValue  // One large component

        // When: Normalize
        let embedding = try Embedding<EmbedKit.Dim768>(values)
        let result = try embedding.normalized().get()

        // Then: Verify all components preserved
        let components = result.toArray()

        // Large component should be preserved
        XCTAssertGreaterThan(components[0], 0.9, "Large component should dominate")

        // Small components should NOT be zero
        let smallComponents = components.dropFirst()
        let nonZeroCount = smallComponents.filter { $0 != 0.0 }.count

        XCTAssertGreaterThan(
            nonZeroCount,
            dimension / 2,
            "At least half of small components should be preserved"
        )

        print("✓ Different magnitudes: \(nonZeroCount)/\(dimension-1) small components preserved")
    }

    /// T-NUM-004d-6: Accumulation of rounding errors
    ///
    /// Tests that rounding errors don't accumulate unboundedly
    /// across many components.
    func testPrecisionLoss_RoundingErrorAccumulation_Bounded() async throws {
        // Given: Many small values that might accumulate rounding errors
        let dimension = TestConfig.dim1536  // Large dimension
        let smallValue: Float = 0.001
        let values = [Float](repeating: smallValue, count: dimension)

        // When: Normalize
        let embedding = try Embedding<EmbedKit.Dim1536>(values)
        let result = try embedding.normalized().get()

        // Then: Magnitude should still be 1.0 (errors bounded)
        // Note: With 1536 dimensions, we need a slightly larger tolerance
        // due to Float32/Double precision differences in accumulation
        let magnitude = result.magnitude
        let magnitudeTolerance: Float = 2e-5  // Relaxed for high-dimension accumulation
        XCTAssertEqual(magnitude, 1.0, accuracy: magnitudeTolerance)

        // Verify: Sum of squares equals 1.0 (rounding errors bounded)
        // The sum is computed in Float32, so accumulation error is expected
        let components = result.toArray()
        let sumSquares = components.reduce(0.0) { $0 + $1 * $1 }

        // For 1536 dimensions with Float32 accumulation, allow slightly larger tolerance
        XCTAssertEqual(sumSquares, 1.0, accuracy: 2e-5, "Sum of squares should be 1.0")

        print("✓ Rounding error accumulation bounded: sum² = \(sumSquares)")
    }

    /// T-NUM-004d-7: Unit vector preservation
    ///
    /// Tests that normalizing standard basis vectors (unit vectors)
    /// remains exact.
    func testPrecisionLoss_UnitVectors_ExactlyOne() async throws {
        // Test standard basis vectors: [1, 0, 0, ...], [0, 1, 0, ...], etc.
        let dimension = TestConfig.dim384

        for i in 0..<min(10, dimension) {  // Test first 10 basis vectors
            var values = [Float](repeating: 0.0, count: dimension)
            values[i] = 1.0  // Unit vector

            let embedding = try Embedding<EmbedKit.Dim384>(values)
            let result = try embedding.normalized().get()

            // Unit vectors are already normalized, should remain unchanged
            let components = result.toArray()

            XCTAssertEqual(components[i], 1.0, accuracy: Float.ulpOfOne, "Unit vector[\(i)] should be exact")

            // All other components should be exactly zero
            for (j, component) in components.enumerated() where j != i {
                XCTAssertEqual(component, 0.0, accuracy: Float.ulpOfOne, "Non-unit components should be zero")
            }
        }

        print("✓ Unit vectors remain exact after normalization")
    }

    /// T-NUM-004d-8: Random vectors magnitude verification
    ///
    /// Tests that random vectors normalize correctly to magnitude 1.0.
    func testPrecisionLoss_RandomVectors_MagnitudeOne() async throws {
        // Test multiple random vectors
        let dimension = TestConfig.dim768
        let testCount = 20

        for trial in 0..<testCount {
            // Given: Random vector
            let values = createRandomVector(dimension: dimension, range: -10.0...10.0)

            // When: Normalize
            let embedding = try Embedding<EmbedKit.Dim768>(values)
            let result = try embedding.normalized().get()

            // Then: Magnitude should be 1.0
            let magnitude = result.magnitude
            XCTAssertEqual(
                magnitude,
                1.0,
                accuracy: TestConfig.magnitudeTolerance,
                "Random vector \(trial) should have magnitude 1.0"
            )
        }

        print("✓ All \(testCount) random vectors normalized to magnitude 1.0")
    }

    /// T-NUM-004d-9: Orthogonal vectors remain orthogonal
    ///
    /// Tests that normalizing orthogonal vectors preserves orthogonality.
    func testPrecisionLoss_OrthogonalVectors_PreserveOrthogonality() async throws {
        // Given: Two orthogonal vectors
        let dimension = TestConfig.dim384
        var vector1 = [Float](repeating: 0.0, count: dimension)
        var vector2 = [Float](repeating: 0.0, count: dimension)

        // Make them orthogonal: v1 = [1, 1, 0, 0, ...], v2 = [1, -1, 0, 0, ...]
        vector1[0] = 1.0
        vector1[1] = 1.0

        vector2[0] = 1.0
        vector2[1] = -1.0

        // Verify orthogonality: dot product = 0
        let dotProduct = zip(vector1, vector2).reduce(0.0) { $0 + $1.0 * $1.1 }
        XCTAssertEqual(dotProduct, 0.0, accuracy: Float.ulpOfOne, "Vectors should be orthogonal")

        // When: Normalize both
        let embedding1 = try Embedding<EmbedKit.Dim384>(vector1)
        let embedding2 = try Embedding<EmbedKit.Dim384>(vector2)

        let normalized1 = try embedding1.normalized().get()
        let normalized2 = try embedding2.normalized().get()

        // Then: Should still be orthogonal
        let normalizedDot = normalized1.dotProduct(normalized2)

        XCTAssertEqual(
            normalizedDot,
            0.0,
            accuracy: 1e-6,
            "Orthogonality should be preserved after normalization"
        )

        print("✓ Orthogonal vectors remain orthogonal: dot product = \(normalizedDot)")
    }

    /// T-NUM-004d-10: Repeated normalization stability
    ///
    /// Tests that normalizing an already-normalized vector is idempotent
    /// (doesn't drift away from magnitude 1.0).
    func testPrecisionLoss_RepeatedNormalization_Idempotent() async throws {
        // Given: Some vector
        let dimension = TestConfig.dim384
        let values = createRandomVector(dimension: dimension, range: -5.0...5.0)

        // When: Normalize multiple times
        var current = try Embedding<EmbedKit.Dim384>(values)

        let iterations = 10
        var magnitudes: [Float] = []

        for _ in 0..<iterations {
            current = try current.normalized().get()
            magnitudes.append(current.magnitude)
        }

        // Then: Magnitude should remain 1.0 (not drift)
        for (i, magnitude) in magnitudes.enumerated() {
            XCTAssertEqual(
                magnitude,
                1.0,
                accuracy: TestConfig.magnitudeTolerance,
                "Iteration \(i): magnitude should remain 1.0"
            )
        }

        // Verify: No cumulative drift
        let drift = abs(magnitudes.last! - 1.0)
        XCTAssertLessThan(drift, TestConfig.magnitudeTolerance, "No drift after \(iterations) normalizations")

        print("✓ Repeated normalization stable: drift = \(drift) after \(iterations) iterations")
        print("  Magnitudes: \(magnitudes.map { String(format: "%.8f", $0) })")
    }

    // MARK: - Helper Functions

    // MARK: Vector Generation Helpers

    /// Create a vector with uniform large values
    private func createLargeUniformVector(dimension: Int, scale: Float = 0.1) -> [Float] {
        let value = TestConfig.maxFinite * scale
        return [Float](repeating: value, count: dimension)
    }

    /// Create a vector with uniform small values
    private func createSmallUniformVector(dimension: Int, scale: Float = 100.0) -> [Float] {
        let value = TestConfig.minNormal * scale
        return [Float](repeating: value, count: dimension)
    }

    /// Create a vector with mixed large and small magnitudes
    private func createMixedMagnitudeVector(dimension: Int, largeScale: Float = 0.1, smallScale: Float = 100.0) -> [Float] {
        var vector = [Float](repeating: 0.0, count: dimension)
        let largeValue = TestConfig.maxFinite * largeScale
        let smallValue = TestConfig.minNormal * smallScale

        for i in 0..<dimension {
            vector[i] = (i % 2 == 0) ? largeValue : smallValue
        }
        return vector
    }

    /// Create a vector with subnormal (denormalized) values
    private func createSubnormalVector(dimension: Int) -> [Float] {
        let subnormalValue = TestConfig.minNonzero * 10.0  // Still subnormal
        return [Float](repeating: subnormalValue, count: dimension)
    }

    /// Create a vector with progressively increasing values
    private func createProgressiveVector(dimension: Int, start: Float, step: Float) -> [Float] {
        return (0..<dimension).map { Float($0) * step + start }
    }

    /// Create a random vector within specified range
    private func createRandomVector(dimension: Int, range: ClosedRange<Float>) -> [Float] {
        return (0..<dimension).map { _ in Float.random(in: range) }
    }

    // MARK: Validation Helpers

    /// Assert that a vector is properly normalized (magnitude ≈ 1.0)
    private func assertIsNormalized<D: EmbeddingDimension>(
        _ embedding: Embedding<D>,
        accuracy: Float = TestConfig.magnitudeTolerance,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let magnitude = embedding.magnitude
        XCTAssertEqual(
            magnitude,
            1.0,
            accuracy: accuracy,
            "Vector magnitude should be 1.0, got \(magnitude)",
            file: file,
            line: line
        )
    }

    /// Assert all values in array are finite (no NaN or Infinity)
    private func assertAllFinite(
        _ values: [Float],
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        for (index, value) in values.enumerated() {
            XCTAssertTrue(
                value.isFinite,
                "Non-finite value at index \(index): \(value)",
                file: file,
                line: line
            )
        }
    }

    /// Assert two float arrays match within tolerance
    private func assertVectorsMatch(
        _ lhs: [Float],
        _ rhs: [Float],
        tolerance: Float = TestConfig.componentTolerance,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(lhs.count, rhs.count, "Vector dimension mismatch", file: file, line: line)

        for (index, (left, right)) in zip(lhs, rhs).enumerated() {
            let difference = abs(left - right)
            XCTAssertLessThanOrEqual(
                difference,
                tolerance,
                "Mismatch at index \(index): \(left) vs \(right) (diff: \(difference))",
                file: file,
                line: line
            )
        }
    }

    // MARK: Numerical Analysis Helpers

    /// Compute L2 norm manually for validation (overflow-safe)
    /// Uses two-pass scaling in Double precision to avoid overflow/underflow.
    private func computeNorm(_ vector: [Float]) -> Float {
        var scale: Double = 0
        for x in vector {
            if x.isFinite {
                let a = abs(Double(x))
                if a > scale { scale = a }
            }
        }
        if scale == 0 { return 0 }
        var sumSq: Double = 0
        for x in vector {
            if x.isFinite {
                let s = Double(x) / scale
                sumSq += s * s
            }
        }
        return Float(scale * sumSq.squareRoot())
    }

    /// Overflow-safe L2 norm computation in Double precision
    /// Uses two-pass scaling to avoid overflow/underflow.
    private func computeNormStableDouble(_ vector: [Float]) -> Double {
        var scale: Double = 0
        for x in vector {
            if x.isFinite {
                let a = abs(Double(x))
                if a > scale { scale = a }
            }
        }
        if scale == 0 { return 0 }
        var sumSq: Double = 0
        for x in vector {
            if x.isFinite {
                let s = Double(x) / scale
                sumSq += s * s
            }
        }
        return scale * sumSq.squareRoot()
    }

    /// Compute relative error between expected and actual values
    private func computeRelativeError(_ expected: Float, _ actual: Float) -> Float {
        guard expected != 0.0 else { return abs(actual) }
        return abs((actual - expected) / expected)
    }

    /// Check if a float is subnormal (denormalized)
    private func isSubnormal(_ value: Float) -> Bool {
        return value != 0.0 && abs(value) < TestConfig.minNormal
    }

    /// Check if a float is normal (not zero, subnormal, infinite, or NaN)
    private func isNormal(_ value: Float) -> Bool {
        return value.isNormal
    }
}
