// Test target: EmbedKitTests (migrated)
import Testing
@testable import EmbedKit

@Suite("Mean Pooling Fallback")
struct MeanPoolingFallbackTests {
    // Backend that returns ones per token vector (dim=4)
    actor OnesBackend: CoreMLProcessingBackend {
        private(set) var isLoaded: Bool = false
        var memoryUsage: Int64 { 0 }
        func load() async throws { isLoaded = true }
        func unload() async throws { isLoaded = false }
        func process(_ input: CoreMLInput) async throws -> CoreMLOutput {
            let dim = 4
            let token = [Float](repeating: 1, count: dim)
            var values: [Float] = []
            values.reserveCapacity(input.tokenIDs.count * dim)
            for _ in input.tokenIDs { values.append(contentsOf: token) }
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
            "a", "b"
        ])
        return WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
    }

    @Test
    func meanPooling_fallbackOnZeroMask_scaffold() async throws {
        // Scenario: one item empty (preLen=0) padded to bucket, mask all zeros; expect fallback to unmasked mean.
        // TODO: Validate pooled vector equals unmasked mean (ones)
        let backend = OnesBackend()
        let tokenizer = makeWPTokenizer()
        var cfg = EmbeddingConfiguration()
        cfg.includeSpecialTokens = false
        cfg.maxTokens = 8
        cfg.paddingStrategy = .batch
        cfg.poolingStrategy = .mean
        cfg.normalizeOutput = false
        let model = AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: cfg, id: ModelID(provider: "test", name: "mean-fallback", version: "1.0"), dimensions: 4, device: .cpu)
        var opts = BatchOptions()
        opts.bucketSize = 4
        let out = try await model.embedBatch(["", "a"], options: opts)
        #expect(out.count == 2)
        // For empty input padded to bucket with all-zero mask, fallback should compute unmasked mean.
        // With OnesBackend, expected vector is all ones.
        let v0 = out[0].vector
        #expect(v0.count == 4)
        for x in v0 { #expect(abs(x - 1.0) <= 1e-5) }
    }
}
