import Testing
@testable import EmbedKit

@Suite("Memory Trim & Unload")
struct MemoryTrimTests {
    actor TrackBackend: CoreMLProcessingBackend {
        private(set) var isLoaded: Bool = false
        var memoryUsage: Int64 { isLoaded ? 123456 : 0 }
        private(set) var unloadCount: Int = 0
        func load() async throws { isLoaded = true }
        func unload() async throws { isLoaded = false; unloadCount &+= 1 }
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
    func trimMemory_resetsCache() async throws {
        let backend = TrackBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            maxTokens: 16,
            paddingStrategy: .none,
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: cfg, id: ModelID(provider: "test", name: "trim", version: "1.0"), dimensions: 4, device: .cpu)

        _ = try await model.embed("cache me") // populate cache
        let before = await model.metrics
        #expect(before.totalRequests >= 1)

        await model.trimMemory() // non-aggressive
        let after = await model.metrics
        // Hit rate resets after cache reset
        #expect(after.cacheHitRate == 0)
    }

    @Test
    func trimMemory_aggressive_unloadsBackend() async throws {
        let backend = TrackBackend()
        let tokenizer = SimpleTokenizer()
        let model = AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: EmbeddingConfiguration(), id: ModelID(provider: "test", name: "trim2", version: "1.0"), dimensions: 4, device: .cpu)
        try await backend.load()
        #expect(await backend.isLoaded)
        await model.trimMemory(aggressive: true)
        #expect(!(await backend.isLoaded))
    }
}

