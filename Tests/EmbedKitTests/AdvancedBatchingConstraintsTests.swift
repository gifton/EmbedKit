import Testing
@testable import EmbedKit

@Suite("Advanced Batching Constraints")
struct AdvancedBatchingConstraintsTests {
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
    func minBatchSize_respected_whenAvailable() async throws {
        let backend = CountingBackend()
        // Use WordPieceTokenizer with PAD token for batch padding
        let vocab = Vocabulary(tokens: ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "a", "b", "c"])
        let tokenizer = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
        let cfg = EmbeddingConfiguration(
            maxTokens: 64,
            paddingStrategy: .batch,
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: cfg, id: ModelID(provider: "test", name: "adv-min", version: "1.0"), dimensions: 4, device: .cpu)

        let texts = Array(repeating: "a b c", count: 9)
        let opts = BatchOptions(
            maxBatchSize: 3,
            bucketSize: 4,
            minBatchSize: 3
        )
        _ = try await model.embedBatch(texts, options: opts)
        let sizes = await backend.batchSizes
        #expect(!sizes.isEmpty)
        #expect(sizes.allSatisfy { $0 == 3 })
    }

    @Test
    func maxPaddingRatio_limitsInclusion() async throws {
        let backend = CountingBackend()
        let vocab = Vocabulary(tokens: ["[PAD]","[CLS]","[SEP]","[UNK]","a","b"]) 
        let tokenizer = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
        let cfg = EmbeddingConfiguration(
            maxTokens: 64,
            paddingStrategy: .batch,
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: cfg, id: ModelID(provider: "test", name: "adv-ratio", version: "1.0"), dimensions: 4, device: .cpu)

        let short = Array(repeating: "a b", count: 5)
        let long = Array(repeating: Array(repeating: "a", count: 13).joined(separator: " "), count: 5)
        let texts = long + short

        let opts = BatchOptions(
            sortByLength: true,
            bucketSize: 16,
            maxPaddingRatio: 0.25
        )
        _ = try await model.embedBatch(texts, options: opts)
        let sizes = await backend.batchSizes
        #expect(!sizes.isEmpty)
        #expect(sizes.first ?? 0 <= 5)
    }
}
