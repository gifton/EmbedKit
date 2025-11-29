// EmbedKit - CoreML Backend Tests

import Testing
import Foundation
@testable import EmbedKit

@Suite("CoreML Backend")
struct CoreMLBackendTests {

    // MARK: - DimensionConstraint Tests

    @Test("DimensionConstraint.fixed validates exact values")
    func fixedConstraintValidation() {
        let constraint = DimensionConstraint.fixed(128)

        #expect(constraint.satisfies(128))
        #expect(!constraint.satisfies(127))
        #expect(!constraint.satisfies(129))
        #expect(!constraint.satisfies(0))
    }

    @Test("DimensionConstraint.flexible validates range")
    func flexibleConstraintValidation() {
        // Unbounded flexible
        let unbounded = DimensionConstraint.flexible(min: nil, max: nil)
        #expect(unbounded.satisfies(1))
        #expect(unbounded.satisfies(1000000))
        #expect(unbounded.satisfies(0))

        // Min only
        let minOnly = DimensionConstraint.flexible(min: 10, max: nil)
        #expect(!minOnly.satisfies(5))
        #expect(minOnly.satisfies(10))
        #expect(minOnly.satisfies(1000))

        // Max only
        let maxOnly = DimensionConstraint.flexible(min: nil, max: 512)
        #expect(maxOnly.satisfies(1))
        #expect(maxOnly.satisfies(512))
        #expect(!maxOnly.satisfies(513))

        // Both bounds
        let bounded = DimensionConstraint.flexible(min: 1, max: 512)
        #expect(!bounded.satisfies(0))
        #expect(bounded.satisfies(1))
        #expect(bounded.satisfies(256))
        #expect(bounded.satisfies(512))
        #expect(!bounded.satisfies(513))
    }

    @Test("DimensionConstraint.enumerated validates allowed values")
    func enumeratedConstraintValidation() {
        let constraint = DimensionConstraint.enumerated([64, 128, 256, 512])

        #expect(constraint.satisfies(64))
        #expect(constraint.satisfies(128))
        #expect(constraint.satisfies(256))
        #expect(constraint.satisfies(512))
        #expect(!constraint.satisfies(65))
        #expect(!constraint.satisfies(100))
        #expect(!constraint.satisfies(1024))
    }

    @Test("DimensionConstraint description formatting")
    func constraintDescription() {
        #expect(DimensionConstraint.fixed(128).description == "128")
        #expect(DimensionConstraint.flexible(min: nil, max: nil).description == "[*..*]")
        #expect(DimensionConstraint.flexible(min: 1, max: nil).description == "[1..*]")
        #expect(DimensionConstraint.flexible(min: nil, max: 512).description == "[*..512]")
        #expect(DimensionConstraint.flexible(min: 1, max: 512).description == "[1..512]")
        #expect(DimensionConstraint.enumerated([64, 128]).description == "{64,128}")
    }

    // MARK: - ShapeConstraints Tests

    @Test("ShapeConstraints validates matching shapes")
    func shapeConstraintsValidation() throws {
        let constraints = ShapeConstraints(dimensions: [
            .fixed(1),
            .flexible(min: 1, max: 512),
            .fixed(384)
        ])

        // Valid shapes
        try constraints.validate([1, 128, 384])
        try constraints.validate([1, 512, 384])
        try constraints.validate([1, 1, 384])

        // Invalid: wrong batch size
        #expect(throws: EmbedKitError.self) {
            try constraints.validate([2, 128, 384])
        }

        // Invalid: seq too long
        #expect(throws: EmbedKitError.self) {
            try constraints.validate([1, 1024, 384])
        }

        // Invalid: wrong hidden dim
        #expect(throws: EmbedKitError.self) {
            try constraints.validate([1, 128, 768])
        }

        // Invalid: wrong rank
        #expect(throws: EmbedKitError.self) {
            try constraints.validate([1, 128])
        }
    }

    @Test("ShapeConstraints detects flexible dimensions")
    func shapeConstraintsFlexibility() {
        let allFixed = ShapeConstraints(dimensions: [
            .fixed(1), .fixed(128), .fixed(384)
        ])
        #expect(!allFixed.hasFlexibleDimensions)

        let someFlexible = ShapeConstraints(dimensions: [
            .fixed(1), .flexible(min: 1, max: 512), .fixed(384)
        ])
        #expect(someFlexible.hasFlexibleDimensions)

        let enumerated = ShapeConstraints(dimensions: [
            .fixed(1), .enumerated([64, 128, 256]), .fixed(384)
        ])
        #expect(enumerated.hasFlexibleDimensions)
    }

    @Test("ShapeConstraints description formatting")
    func shapeConstraintsDescription() {
        let constraints = ShapeConstraints(dimensions: [
            .fixed(1),
            .flexible(min: 1, max: 512),
            .fixed(384)
        ])

        #expect(constraints.description == "[1, [1..512], 384]")
    }

    // MARK: - CoreMLBackend Initialization Tests

    @Test("Backend initializes with default configuration")
    func backendDefaultInit() async {
        let backend = CoreMLBackend(modelURL: nil)

        let useStateful = await backend.useStatefulPredictions
        let maxSeqLen = await backend.maxSequenceLengthHint
        let hiddenDim = await backend.hiddenDimensionHint

        #expect(useStateful == true)
        #expect(maxSeqLen == 512)
        #expect(hiddenDim == 384)
    }

    @Test("Backend initializes with custom configuration")
    func backendCustomInit() async {
        let backend = CoreMLBackend(
            modelURL: nil,
            device: .gpu,
            useStatefulPredictions: false,
            maxSequenceLengthHint: 256,
            hiddenDimensionHint: 768
        )

        let useStateful = await backend.useStatefulPredictions
        let maxSeqLen = await backend.maxSequenceLengthHint
        let hiddenDim = await backend.hiddenDimensionHint

        #expect(useStateful == false)
        #expect(maxSeqLen == 256)
        #expect(hiddenDim == 768)
    }

    @Test("Backend reports not loaded initially")
    func backendNotLoadedInitially() async {
        let backend = CoreMLBackend(modelURL: nil)
        let loaded = await backend.isLoaded
        #expect(!loaded)
    }

    // MARK: - Shape Constraints API Tests

    @Test("Backend returns empty constraints when not loaded")
    func emptyConstraintsWhenNotLoaded() async {
        let backend = CoreMLBackend(modelURL: nil)

        let inputConstraints = await backend.allInputShapeConstraints
        let outputConstraints = await backend.outputShapeConstraints
        let hasFlexible = await backend.hasFlexibleInputs

        #expect(inputConstraints.isEmpty)
        #expect(outputConstraints == nil)
        #expect(!hasFlexible)
    }

    @Test("Backend allows validation when no constraints cached")
    func validationWithNoConstraints() async throws {
        let backend = CoreMLBackend(modelURL: nil)

        // Should not throw when no constraints are cached
        try await backend.validateInputShape([1, 128, 384])
        try await backend.validateInputShape([32, 512])
    }

    // MARK: - ComputeClass Tests

    @Test("ComputeClass selection based on workload")
    func computeClassSelection() {
        // Small workloads -> CPU
        #expect(CoreMLBackend.chooseComputeClass(paddedLength: 32, batchSize: 1) == .cpu)
        #expect(CoreMLBackend.chooseComputeClass(paddedLength: 64, batchSize: 2) == .cpu)

        // Medium workloads -> Hybrid
        #expect(CoreMLBackend.chooseComputeClass(paddedLength: 128, batchSize: 4) == .hybrid)
        #expect(CoreMLBackend.chooseComputeClass(paddedLength: 256, batchSize: 8) == .hybrid)

        // Large workloads -> All
        #expect(CoreMLBackend.chooseComputeClass(paddedLength: 512, batchSize: 16) == .all)
        #expect(CoreMLBackend.chooseComputeClass(paddedLength: 256, batchSize: 32) == .all)
    }

    // MARK: - Shape Validation Helper Tests

    @Test("Static shape validation works correctly")
    func staticShapeValidation() throws {
        // Matching shapes with all fixed
        try CoreMLBackend.validateShape(desired: [1, 128, 384], against: [1, 128, 384])

        // Matching with flexible (nil) dimensions
        try CoreMLBackend.validateShape(desired: [1, 128, 384], against: [1, nil, 384])
        try CoreMLBackend.validateShape(desired: [1, 256, 384], against: [1, nil, 384])

        // All flexible
        try CoreMLBackend.validateShape(desired: [8, 512, 768], against: [nil, nil, nil])

        // Empty constraint allows anything
        try CoreMLBackend.validateShape(desired: [1, 2, 3], against: [])
    }

    @Test("Static shape validation rejects mismatches")
    func staticShapeValidationRejects() {
        // Fixed dimension mismatch
        #expect(throws: EmbedKitError.self) {
            try CoreMLBackend.validateShape(desired: [2, 128, 384], against: [1, 128, 384])
        }

        // Rank mismatch
        #expect(throws: EmbedKitError.self) {
            try CoreMLBackend.validateShape(desired: [1, 128], against: [1, 128, 384])
        }

        // Fixed vs actual mismatch at non-flexible position
        #expect(throws: EmbedKitError.self) {
            try CoreMLBackend.validateShape(desired: [1, 128, 768], against: [1, nil, 384])
        }
    }

    // MARK: - CoreMLInput/Output Tests

    @Test("CoreMLInput initializes correctly")
    func coreMLInputInit() {
        let input = CoreMLInput(
            tokenIDs: [101, 2003, 2023, 102],
            attentionMask: [1, 1, 1, 1]
        )

        #expect(input.tokenIDs == [101, 2003, 2023, 102])
        #expect(input.attentionMask == [1, 1, 1, 1])
    }

    @Test("CoreMLOutput initializes correctly")
    func coreMLOutputInit() {
        let values: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        let output = CoreMLOutput(values: values, shape: [2, 3])

        #expect(output.values == values)
        #expect(output.shape == [2, 3])
    }
}
