import Testing
@testable import EmbedKit

@Suite("Auto Device Policy")
struct AutoDevicePolicyTests {
    @Test
    func choose_cpu_for_tiny_batches() async throws {
        let cls = CoreMLBackend.chooseComputeClass(paddedLength: 32, batchSize: 1)
        #expect(cls == .cpu)
    }

    @Test
    func choose_hybrid_for_medium() async throws {
        let cls = CoreMLBackend.chooseComputeClass(paddedLength: 128, batchSize: 4)
        #expect(cls == .hybrid)
    }

    @Test
    func choose_all_for_large() async throws {
        let cls = CoreMLBackend.chooseComputeClass(paddedLength: 384, batchSize: 16)
        #expect(cls == .all)
    }
}

