import Testing
@testable import EmbedKit

@Suite("Memory Pressure Handling")
struct MemoryPressureHandlingTests {
    actor TrackBackend: CoreMLProcessingBackend {
        private(set) var isLoaded: Bool = false
        var memoryUsage: Int64 { isLoaded ? 42 : 0 }
        func load() async throws { isLoaded = true }
        func unload() async throws { isLoaded = false }
        func process(_ input: CoreMLInput) async throws -> CoreMLOutput {
            let dim = 4
            return CoreMLOutput(values: Array(repeating: 0, count: input.tokenIDs.count * dim), shape: [input.tokenIDs.count, dim])
        }
        func processBatch(_ inputs: [CoreMLInput]) async throws -> [CoreMLOutput] {
            let dim = 4
            return inputs.map { inp in
                CoreMLOutput(values: Array(repeating: 0, count: inp.tokenIDs.count * dim), shape: [inp.tokenIDs.count, dim])
            }
        }
    }

    @Test
    func simulate_critical_trimsAggressively_unloadsBackend() async throws {
        let backend = TrackBackend()
        let tokenizer = SimpleTokenizer()
        let model = AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: EmbeddingConfiguration(), id: ModelID(provider: "test", name: "mp", version: "1.0"), dimensions: 4, device: .cpu)
        try await backend.load()
        #expect(await backend.isLoaded)
        let manager = ModelManager()
        await manager.register(model)
        await manager.simulateMemoryPressure(.critical)
        #expect(!(await backend.isLoaded))
    }

    @Test
    func simulate_warning_resetsCache() async throws {
        let backend = TrackBackend()
        let tokenizer = SimpleTokenizer()
        var cfg = EmbeddingConfiguration()
        cfg.includeSpecialTokens = false
        let model = AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: cfg, id: ModelID(provider: "test", name: "mp2", version: "1.0"), dimensions: 4, device: .cpu)
        let manager = ModelManager()
        await manager.register(model)
        _ = try await model.embed("cache me")
        let before = await model.metrics
        #expect(before.totalRequests >= 1)
        await manager.simulateMemoryPressure(.warning)
        let after = await model.metrics
        #expect(after.cacheHitRate == 0)
    }
}
