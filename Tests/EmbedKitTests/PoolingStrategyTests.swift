import Testing
@testable import EmbedKit

// Backend with increasing token values to exercise pooling strategies
actor StepBackend: CoreMLProcessingBackend {
    private(set) var isLoaded: Bool = false
    var memoryUsage: Int64 { 0 }

    func load() async throws { isLoaded = true }
    func unload() async throws { isLoaded = false }

    func process(_ input: CoreMLInput) async throws -> CoreMLOutput {
        let dim = 4
        var values: [Float] = []
        values.reserveCapacity(input.tokenIDs.count * dim)
        for i in 0..<input.tokenIDs.count {
            let b = Float(i + 1)
            values.append(contentsOf: [b, 2*b, 3*b, 4*b])
        }
        return CoreMLOutput(values: values, shape: [input.tokenIDs.count, dim])
    }

    func processBatch(_ inputs: [CoreMLInput]) async throws -> [CoreMLOutput] {
        var outs: [CoreMLOutput] = []
        outs.reserveCapacity(inputs.count)
        for inp in inputs { outs.append(try await process(inp)) }
        return outs
    }
}
@Suite("Pooling Strategies")
struct PoolingStrategyTests {
@Test
func pooling_clsAndMax() async throws {
        let backend = StepBackend()
        let tokenizer = SimpleTokenizer()
        var cfg = EmbeddingConfiguration()
        cfg.includeSpecialTokens = false
        cfg.normalizeOutput = false

        // CLS pooling (first token vector)
        cfg.poolingStrategy = .cls
        var model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            id: ModelID(provider: "test", name: "step", version: "1.0"),
            dimensions: 4,
            device: .cpu
        )
        var emb = try await model.embed("a b c")
        #expect(emb.vector == [1,2,3,4])

        // Max pooling (largest token vector = last)
        cfg.poolingStrategy = .max
        model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            id: ModelID(provider: "test", name: "step", version: "1.0"),
            dimensions: 4,
            device: .cpu
        )
        emb = try await model.embed("a b c")
        #expect(emb.vector == [3,6,9,12])
}
}
