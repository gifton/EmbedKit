import Testing
@testable import EmbedKit

@Suite("ModelManager Batch Integration")
struct ModelManagerBatchIntegrationTests {
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
    func manager_embedBatch_usesModelBatch_andPopulatesCounts() async throws {
        let backend = CountingBackend()
        let vocab = Vocabulary(tokens: ["[PAD]","a","b","c"]) // PAD for batch padding
        let tokenizer = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
        let cfg = EmbeddingConfiguration(
            maxTokens: 64,
            paddingStrategy: .batch,
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: cfg, id: ModelID(provider: "test", name: "mm", version: "1.0"), dimensions: 4, device: .cpu)

        let manager = ModelManager()
        await manager.register(model)
        let id = model.id

        let texts = ["a b c", "a", "a b"]
        let opts = BatchOptions(
            bucketSize: 4
        )
        let result = try await manager.embedBatch(texts, using: id, options: opts)
        #expect(result.embeddings.count == texts.count)
        #expect(result.tokenCounts.count == texts.count)
        #expect(result.perItemTimes.count == texts.count)
        let sum = result.perItemTimes.reduce(0, +)
        #expect(abs(sum - result.totalTime) <= max(0.001, result.totalTime * 0.1))
        let sizes = await backend.batchSizes
        #expect(!sizes.isEmpty)
    }
}
