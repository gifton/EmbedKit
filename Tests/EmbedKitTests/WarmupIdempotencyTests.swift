// Test target: EmbedKitTests (migrated)
import Testing
@testable import EmbedKit

@Suite("Warmup Idempotency")
struct WarmupIdempotencyTests {
    @Test
    func warmupTwice_hasNoSideEffects() async throws {
        actor LoadCountingBackend: CoreMLProcessingBackend {
            private(set) var isLoaded: Bool = false
            var memoryUsage: Int64 { 0 }
            private(set) var loads: Int = 0
            func load() async throws { if !isLoaded { isLoaded = true; loads += 1 } }
            func unload() async throws { isLoaded = false }
            func process(_ input: CoreMLInput) async throws -> CoreMLOutput { CoreMLOutput(values: [], shape: [1,4]) }
            func processBatch(_ inputs: [CoreMLInput]) async throws -> [CoreMLOutput] { [] }
        }

        let backend = LoadCountingBackend()
        let tokenizer = SimpleTokenizer()
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: EmbeddingConfiguration(),
            id: ModelID(provider: "test", name: "warm", version: "1.0"),
            dimensions: 4,
            device: .cpu
        )

        try await model.warmup()
        try await model.warmup()
        #expect(await backend.isLoaded)
        #expect((await backend.loads) == 1)
    }
}
