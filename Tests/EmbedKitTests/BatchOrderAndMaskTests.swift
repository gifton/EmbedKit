// Test target: EmbedKitTests (migrated)
import Testing
@testable import EmbedKit

@Suite("Batch Order & Mask Equivalence")
struct BatchOrderAndMaskTests {
    @Test
    func orderPreservation_withSortByLengthOnOff() async throws {
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

        let backend = AvgIDBackend()
        let cfg = EmbeddingConfiguration(
            includeSpecialTokens: false,
            poolingStrategy: .mean,
            normalizeOutput: false
        )
        let tokenizer = SimpleTokenizer()
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            id: ModelID(provider: "test", name: "order", version: "1.0"),
            dimensions: 4,
            device: .cpu
        )

        let texts = [
            "a b ccc", // longer
            "a",       // shortest
            "a b"      // mid
        ]

        // Baseline via single embeds
        var expected: [[Float]] = []
        for t in texts {
            let e = try await model.embed(t)
            expected.append(e.vector)
        }

        // sortByLength = true
        let opts = BatchOptions(sortByLength: true)
        let batchTrue = try await model.embedBatch(texts, options: opts)
        #expect(batchTrue.count == texts.count)
        for i in 0..<texts.count {
            let v = batchTrue[i].vector
            let exp = expected[i]
            #expect(v.count == exp.count)
            for (a,b) in zip(v, exp) { #expect(abs(a - b) <= 1e-5) }
        }

        // sortByLength = false
        let optsFalse = BatchOptions(sortByLength: false)
        let batchFalse = try await model.embedBatch(texts, options: optsFalse)
        #expect(batchFalse.count == texts.count)
        for i in 0..<texts.count {
            let v = batchFalse[i].vector
            let exp = expected[i]
            #expect(v.count == exp.count)
            for (a,b) in zip(v, exp) { #expect(abs(a - b) <= 1e-5) }
        }
    }

    @Test
    func meanPooling_maskedEqualsUnpadded() async throws {
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

        let backend = OnesBackend()
        // Use WordPiece tokenizer with PAD for batch padding
        let vocab = Vocabulary(tokens: [
            "[PAD]", "[CLS]", "[SEP]", "[UNK]",
            "a", "b", "c"
        ])
        let tokenizer = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)

        // Model A: no padding
        let cfgA = EmbeddingConfiguration(
            paddingStrategy: .none,
            includeSpecialTokens: false,
            poolingStrategy: .mean,
            normalizeOutput: false
        )
        let modelA = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfgA,
            id: ModelID(provider: "test", name: "nopad", version: "1.0"),
            dimensions: 4,
            device: .cpu
        )

        // Model B: batch padding
        let cfgB = EmbeddingConfiguration(
            paddingStrategy: .batch,
            includeSpecialTokens: false,
            poolingStrategy: .mean,
            normalizeOutput: false
        )
        let modelB = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfgB,
            id: ModelID(provider: "test", name: "batchpad", version: "1.0"),
            dimensions: 4,
            device: .cpu
        )

        let texts = ["a b c", "a"]

        // Baseline unpadded results
        var expected: [[Float]] = []
        for t in texts { expected.append(try await modelA.embed(t).vector) }

        // Batched with padding: should match for each item when mean masking is applied
        let got = try await modelB.embedBatch(texts, options: BatchOptions())
        #expect(got.count == texts.count)
        for i in 0..<texts.count {
            #expect(got[i].vector.count == expected[i].count)
            for (a,b) in zip(got[i].vector, expected[i]) { #expect(abs(a - b) <= 1e-5) }
        }
    }
}
