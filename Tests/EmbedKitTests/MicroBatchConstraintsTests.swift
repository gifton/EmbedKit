import Testing
@testable import EmbedKit

@Suite("Micro-batch Constraints Combo")
struct MicroBatchConstraintsTests {
    actor CountingBackend: CoreMLProcessingBackend {
        private(set) var isLoaded: Bool = false
        var memoryUsage: Int64 { 0 }
        private(set) var batchSizes: [Int] = []
        func load() async throws { isLoaded = true }
        func unload() async throws { isLoaded = false }
        func process(_ input: CoreMLInput) async throws -> CoreMLOutput {
            let dim = 4
            return CoreMLOutput(values: Array(repeating: 0, count: input.tokenIDs.count * dim), shape: [input.tokenIDs.count, dim])
        }
        func processBatch(_ inputs: [CoreMLInput]) async throws -> [CoreMLOutput] {
            batchSizes.append(inputs.count)
            let dim = 4
            return inputs.map { inp in
                CoreMLOutput(values: Array(repeating: 0, count: inp.tokenIDs.count * dim), shape: [inp.tokenIDs.count, dim])
            }
        }
    }

    @Test
    func respectsMinOfMaxBatchTokensAndMaxBatchSize() async throws {
        let backend = CountingBackend()
        let vocab = Vocabulary(tokens: ["[PAD]","a","b"]) // PAD for batch padding
        let tokenizer = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
        let cfg = EmbeddingConfiguration(
            maxTokens: 64,
            paddingStrategy: .batch,
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: cfg, id: ModelID(provider: "test", name: "mb-combo", version: "1.0"), dimensions: 4, device: .cpu)

        // Ten 2-token texts; bucket key = 16 with bucketSize 16
        let texts = Array(repeating: "a b", count: 10)
        let options = BatchOptions(
            maxBatchSize: 3,    // size cap would allow 3, but min(2,3) = 2
            bucketSize: 16,
            maxBatchTokens: 32  // token budget allows only 2 per batch (since 2*16 = 32)
        )
        _ = try await model.embedBatch(texts, options: options)
        let sizes1 = await backend.batchSizes
        #expect(!sizes1.isEmpty)
        #expect(sizes1.allSatisfy { $0 == 2 })

        // Now flip: allow token budget high but restrict maxBatchSize to 2
        let options2 = BatchOptions(
            maxBatchSize: 2,
            bucketSize: 16,
            maxBatchTokens: 10_000  // large
        )
        _ = try await model.embedBatch(texts, options: options2)
        let sizes2 = await backend.batchSizes
        let newSizes = Array(sizes2.dropFirst(sizes1.count))
        #expect(!newSizes.isEmpty)
        #expect(newSizes.allSatisfy { $0 == 2 })
    }
}

