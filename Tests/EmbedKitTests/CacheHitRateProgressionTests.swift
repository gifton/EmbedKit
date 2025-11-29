// Test target: EmbedKitTests (migrated)
import Testing
@testable import EmbedKit

@Suite("Cache Hit Rate Progression")
struct CacheHitRateProgressionTests {
    @Test
    func repeatedInputs_increaseHitRate() async throws {
        let backend = NoOpBackend()
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            id: ModelID(provider: "test", name: "hit", version: "1.0"),
            dimensions: 4,
            device: .cpu
        )

        _ = try await model.embed("repeat me")
        let m1 = await model.metrics
        _ = try await model.embed("repeat me")
        _ = try await model.embed("repeat me")
        let m2 = await model.metrics
        #expect(m2.cacheHitRate >= m1.cacheHitRate)
    }
}
