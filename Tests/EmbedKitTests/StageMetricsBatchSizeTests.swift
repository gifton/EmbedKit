import Testing
@testable import EmbedKit

@Suite("Stage Metrics - Average Batch Size")
struct StageMetricsBatchSizeTests {
    actor CountingBackend: CoreMLProcessingBackend {
        private(set) var isLoaded: Bool = false
        var memoryUsage: Int64 { 0 }
        func load() async throws { isLoaded = true }
        func unload() async throws { isLoaded = false }
        func process(_ input: CoreMLInput) async throws -> CoreMLOutput {
            let dim = 4
            return CoreMLOutput(values: Array(repeating: 0, count: input.tokenIDs.count * dim), shape: [input.tokenIDs.count, dim])
        }
        func processBatch(_ inputs: [CoreMLInput]) async throws -> [CoreMLOutput] {
            let dim = 4
            return inputs.map { inp in
                CoreMLOutput(values: Array(repeating: 0, count: inp.tokenIDs.count * dim), shape: [inp.tokenIDs.count, dim])
            }
        }
    }

    @Test
    func averageBatchSize_reportsExpected() async throws {
        let backend = CountingBackend()
        let vocab = Vocabulary(tokens: ["[PAD]","a","b"]) // PAD for batch padding
        let tokenizer = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
        var cfg = EmbeddingConfiguration()
        cfg.includeSpecialTokens = false
        cfg.maxTokens = 64
        cfg.paddingStrategy = .batch
        let model = AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: cfg, id: ModelID(provider: "test", name: "sm-batch", version: "1.0"), dimensions: 4, device: .cpu)

        // 10 items, bucketSize 16, each length ~2 -> key 16
        let texts = Array(repeating: "a b", count: 10)
        var opts = BatchOptions()
        opts.bucketSize = 16
        opts.maxBatchTokens = 32 // -> 2 per micro-batch
        _ = try await model.embedBatch(texts, options: opts)

        let sm = await model.stageMetricsSnapshot
        #expect(sm.averageBatchSize >= 1.9 && sm.averageBatchSize <= 2.1)
    }
}

