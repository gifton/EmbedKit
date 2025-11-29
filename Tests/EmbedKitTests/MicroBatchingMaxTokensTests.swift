import Testing
@testable import EmbedKit

actor CountingBackend: CoreMLProcessingBackend {
    private(set) var isLoaded: Bool = false
    var memoryUsage: Int64 { 0 }
    private(set) var batchSizes: [Int] = []

    func load() async throws { isLoaded = true }
    func unload() async throws { isLoaded = false }

    func process(_ input: CoreMLInput) async throws -> CoreMLOutput {
        // Not used in this test
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

@Suite("MicroBatching Max Tokens")
struct MicroBatchingMaxTokensTestsSuite {
@Test
func microBatching_respectsMaxBatchTokens() async throws {
    let backend = CountingBackend()
    let vocab = Vocabulary(tokens: ["[PAD]","a","b"]) // PAD for batch padding
    let tokenizer = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
    let cfg = EmbeddingConfiguration(
        maxTokens: 64,
        paddingStrategy: .batch,
        includeSpecialTokens: false
    )
    let model = AppleEmbeddingModel(
        backend: backend,
        tokenizer: tokenizer,
        configuration: cfg,
        id: ModelID(provider: "test", name: "mb", version: "1.0"),
        dimensions: 4,
        device: .cpu
    )

    // Ten 2-token texts; bucket target length will be 16 tokens (rounded by bucketSize)
    let texts = Array(repeating: "a b", count: 10)

    let options = BatchOptions(
        maxBatchSize: 99,   // Ensure token budget is the limiting factor
        bucketSize: 16,
        maxBatchTokens: 32  // With padded length 16, allows only 2 items per micro-batch
    )
    _ = try await model.embedBatch(texts, options: options)

    let sizes = await backend.batchSizes
    #expect(!sizes.isEmpty)
    #expect(sizes.allSatisfy { $0 == 2 })
    #expect(sizes.count == (texts.count + 1) / 2)
}
}
