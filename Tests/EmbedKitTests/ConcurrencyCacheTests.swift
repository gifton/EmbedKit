// Test target: EmbedKitTests (migrated)
import Testing
@testable import EmbedKit

@Suite("Token Cache Concurrency")
struct ConcurrencyCacheTests {
    @Test
    func concurrentEmbeds_shareCacheWithoutRaces() async throws {
        let backend = NoOpBackend()
        let tokenizer = SimpleTokenizer()
        var cfg = EmbeddingConfiguration()
        cfg.includeSpecialTokens = false

        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            id: ModelID(provider: "test", name: "conccache", version: "1.0"),
            dimensions: 4,
            device: .cpu
        )

        _ = try await model.embed("same text")
        let before = await model.metrics
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<10 {
                group.addTask {
                    _ = try? await model.embed("same text")
                }
            }
        }
        let after = await model.metrics
        #expect(after.cacheHitRate >= before.cacheHitRate)
    }
}
