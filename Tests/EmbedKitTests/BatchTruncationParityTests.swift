// Test target: EmbedKitTests (migrated)
import Testing
@testable import EmbedKit

@Suite("Batch Truncation Parity")
struct BatchTruncationParityTests {
    actor PassThroughBackend: CoreMLProcessingBackend {
        private(set) var isLoaded: Bool = false
        var memoryUsage: Int64 { 0 }
        func load() async throws { isLoaded = true }
        func unload() async throws { isLoaded = false }
        func process(_ input: CoreMLInput) async throws -> CoreMLOutput {
            // Dim 4, values derived from token index to be deterministic
            let dim = 4
            var values: [Float] = []
            values.reserveCapacity(input.tokenIDs.count * dim)
            for i in 0..<input.tokenIDs.count {
                let f = Float(i + 1)
                values.append(contentsOf: [f, 0, 0, 0])
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
            "a", "b", "c", "d", "e"
        ])
        return WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
    }

    @Test
    func parity_embedBatch_vs_single_startAndMiddle() async throws {
        let backend = PassThroughBackend()
        let tokenizer = makeWPTokenizer()
        var cfg = EmbeddingConfiguration()
        cfg.includeSpecialTokens = false
        cfg.maxTokens = 6
        cfg.paddingStrategy = .batch
        cfg.poolingStrategy = .mean
        cfg.normalizeOutput = false
        let texts = ["a b c d e", "a b", "a b c"]
        var opts = BatchOptions()
        opts.bucketSize = 4

        // Test for .start
        cfg.truncationStrategy = .start
        var model = AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: cfg, id: ModelID(provider: "test", name: "parity-start", version: "1.0"), dimensions: 4, device: .cpu)
        let batchStart = try await model.embedBatch(texts, options: opts)
        var singleStart: [Embedding] = []
        for t in texts { singleStart.append(try await model.embed(t)) }
        #expect(batchStart.count == singleStart.count)
        for i in 0..<texts.count {
            let a = batchStart[i].vector
            let b = singleStart[i].vector
            #expect(a.count == b.count)
            for (x, y) in zip(a, b) { #expect(abs(x - y) <= 1e-5) }
        }

        // Test for .middle
        cfg.truncationStrategy = .middle
        model = AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: cfg, id: ModelID(provider: "test", name: "parity-middle", version: "1.0"), dimensions: 4, device: .cpu)
        let batchMid = try await model.embedBatch(texts, options: opts)
        var singleMid: [Embedding] = []
        for t in texts { singleMid.append(try await model.embed(t)) }
        #expect(batchMid.count == singleMid.count)
        for i in 0..<texts.count {
            let a = batchMid[i].vector
            let b = singleMid[i].vector
            #expect(a.count == b.count)
            for (x, y) in zip(a, b) { #expect(abs(x - y) <= 1e-5) }
        }
    }
}
