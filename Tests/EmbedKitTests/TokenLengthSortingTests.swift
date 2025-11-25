// Test target: EmbedKitTests (migrated)
import Testing
@testable import EmbedKit

@Suite("Token Length Sorting")
struct TokenLengthSortingTests {
    actor AvgIDBackend: CoreMLProcessingBackend {
        private(set) var isLoaded: Bool = false
        var memoryUsage: Int64 { 0 }
        func load() async throws { isLoaded = true }
        func unload() async throws { isLoaded = false }
        func process(_ input: CoreMLInput) async throws -> CoreMLOutput {
            let dim = 4
            var values: [Float] = []
            values.reserveCapacity(input.tokenIDs.count * dim)
            for id in input.tokenIDs {
                let f = Float(id)
                values.append(contentsOf: [f, f * 2, 0, 0])
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

    private func makeWPTokenizer() -> WordPieceTokenizer {
        let vocab = Vocabulary(tokens: [
            "[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]",
            "a", "b", "c", "ccc"
        ])
        return WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
    }

    @Test
    func sortByTokenLength_preservesOutputOrder() async throws {
        let backend = AvgIDBackend()
        let tokenizer = makeWPTokenizer()
        var cfg = EmbeddingConfiguration()
        cfg.includeSpecialTokens = false
        cfg.maxTokens = 64
        cfg.paddingStrategy = .batch
        cfg.poolingStrategy = .mean
        cfg.normalizeOutput = false
        let model = AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: cfg, id: ModelID(provider: "test", name: "sort-len", version: "1.0"), dimensions: 4, device: .cpu)
        let texts = ["a b ccc", "a", "a b"] // token lengths: 3,1,2
        // Baseline: single embeds in original order
        var expected: [[Float]] = []
        for t in texts { expected.append(try await model.embed(t).vector) }

        // Batched with sortByLength=true must map back to original order
        var opts = BatchOptions()
        opts.sortByLength = true
        opts.bucketSize = 4
        let got = try await model.embedBatch(texts, options: opts)
        #expect(got.count == texts.count)
        for i in 0..<texts.count {
            let a = got[i].vector, b = expected[i]
            #expect(a.count == b.count)
            for (x, y) in zip(a, b) { #expect(abs(x - y) <= 1e-5) }
        }
    }
}
