import Testing
@testable import EmbedKit

@Suite("Token Cache Keys & Hits")
struct TokenCacheKeyTests {
    actor NoOpBackend: CoreMLProcessingBackend {
        private(set) var isLoaded: Bool = false
        var memoryUsage: Int64 { 0 }
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
    func cache_hits_on_repeated_embeddings() async throws {
        let backend = NoOpBackend()
        let tokenizer = SimpleTokenizer()
        var cfg = EmbeddingConfiguration()
        cfg.includeSpecialTokens = false
        cfg.maxTokens = 32
        cfg.paddingStrategy = .none
        let model = AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: cfg, id: ModelID(provider: "test", name: "cache", version: "1.0"), dimensions: 4, device: .cpu)

        _ = try await model.embed("repeat me") // warm fill (misses)
        let m1 = await model.metrics
        #expect(m1.cacheHitRate >= 0.0)

        _ = try await model.embed("repeat me") // hits expected for both pre/cfg entries
        let m2 = await model.metrics
        #expect(m2.cacheHitRate > m1.cacheHitRate)
        #expect(m2.cacheHitRate >= 0.3)
    }

    @Test
    func cache_distinct_texts_do_not_hit() async throws {
        let backend = NoOpBackend()
        let tokenizer = SimpleTokenizer()
        var cfg = EmbeddingConfiguration()
        cfg.includeSpecialTokens = false
        cfg.maxTokens = 32
        cfg.paddingStrategy = .none
        let model = AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: cfg, id: ModelID(provider: "test", name: "cache2", version: "1.0"), dimensions: 4, device: .cpu)

        _ = try await model.embed("first text")
        _ = try await model.embed("second text")
        let m = await model.metrics
        // With two different texts and no repeats, hit rate should remain low
        #expect(m.cacheHitRate <= 0.2)
    }
}

