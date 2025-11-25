// Test suite for AppleEmbeddingModel
import Testing
import Foundation
@testable import EmbedKit

// Deterministic backend for testing shape handling, pooling, and normalization (no CoreML required)
actor DeterministicBackend: CoreMLProcessingBackend {
    private(set) var isLoaded: Bool = false
    var memoryUsage: Int64 { 0 }

    func load() async throws { isLoaded = true }
    func unload() async throws { isLoaded = false }

    func process(_ input: CoreMLInput) async throws -> CoreMLOutput {
        let dim = 4
        // Build per-token vector [1,2,3,4] for each token, regardless of id
        let tokens = input.tokenIDs.count
        var values: [Float] = []
        values.reserveCapacity(tokens * dim)
        for _ in 0..<tokens { values.append(contentsOf: [1,2,3,4]) }
        return CoreMLOutput(values: values, shape: [tokens, dim])
    }

    func processBatch(_ inputs: [CoreMLInput]) async throws -> [CoreMLOutput] {
        var outs: [CoreMLOutput] = []
        outs.reserveCapacity(inputs.count)
        for inp in inputs { outs.append(try await process(inp)) }
        return outs
    }
}

@Suite("AppleEmbeddingModel")
struct AppleEmbeddingModelTests {
    @Test
    func appleModel_embedMeanPoolingAndNormalization() async throws {
        let backend = DeterministicBackend()
        let tokenizer = SimpleTokenizer()
        var cfg = EmbeddingConfiguration()
        cfg.maxTokens = 16
        cfg.includeSpecialTokens = false
        cfg.normalizeOutput = true

        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            id: ModelID(provider: "test", name: "det", version: "1.0"),
            dimensions: 4,
            device: .cpu
        )

        let emb = try await model.embed("one two three four")
        #expect(emb.dimensions == 4)
        // Mean of identical tokens [1,2,3,4] = [1,2,3,4], then normalized
        let base: Double = (1) + (4) + (9) + (16)  // 1*1 2*2 3*3 4*4
        let mag = sqrt(base) // sqrt(30)
        let expected: [Float] = [1,2,3,4].map { Float(Double($0) / mag) }
        for (a,b) in zip(emb.vector, expected) { #expect(abs(a - b) <= 1e-5) }

        // Token count equals non-padding tokens
        #expect(emb.metadata.tokenCount > 0)
        #expect(emb.metadata.normalized)
        #expect(emb.metadata.poolingStrategy == .mean)
    }
}
