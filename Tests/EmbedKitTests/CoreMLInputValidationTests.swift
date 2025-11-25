import Testing
@testable import EmbedKit

@Suite("CoreML Input Validation (Shape)")
struct CoreMLInputValidationTests {
    @Test
    func validate_rankMismatch_throws() async {
        do {
            try CoreMLBackend.validateShape(desired: [8], against: [1, nil])
            #expect(Bool(false), "Expected rank mismatch to throw")
        } catch {
            guard let ek = error as? EmbedKitError else { return #expect(Bool(false), "Unexpected error: \(error)") }
            switch ek {
            case .invalidConfiguration:
                #expect(true)
            default:
                #expect(Bool(false), "Unexpected error: \(ek)")
            }
        }
    }

    @Test
    func validate_fixedBatch_one_passes() async throws {
        // Expect [1, seq] where batch is fixed 1 and seq flexible
        try CoreMLBackend.validateShape(desired: [1, 12], against: [1, nil])
    }

    @Test
    func validate_fixedBatch_wrong_throws() async {
        do {
            try CoreMLBackend.validateShape(desired: [2, 12], against: [1, nil])
            #expect(Bool(false), "Expected fixed batch mismatch to throw")
        } catch {
            guard let ek = error as? EmbedKitError else { return #expect(Bool(false), "Unexpected error: \(error)") }
            switch ek {
            case .invalidConfiguration:
                #expect(true)
            default:
                #expect(Bool(false), "Unexpected error: \(ek)")
            }
        }
    }

    @Test
    func validate_allFlexible_passes() async throws {
        try CoreMLBackend.validateShape(desired: [24], against: [nil])
        try CoreMLBackend.validateShape(desired: [1, 24], against: [nil, nil])
    }
}

