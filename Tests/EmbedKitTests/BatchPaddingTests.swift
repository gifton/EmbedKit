import Testing
@testable import EmbedKit

// Backend that records batch input lengths and ensures batch padding produces equal lengths
actor RecordingBackend: CoreMLProcessingBackend {
    private(set) var isLoaded: Bool = false
    var memoryUsage: Int64 { 0 }
    private(set) var lastLengths: [Int] = []

    func load() async throws { isLoaded = true }
    func unload() async throws { isLoaded = false }

    func process(_ input: CoreMLInput) async throws -> CoreMLOutput {
        // Not used in this test
        let dim = 4
        return CoreMLOutput(values: Array(repeating: 0, count: input.tokenIDs.count * dim), shape: [input.tokenIDs.count, dim])
    }

    func processBatch(_ inputs: [CoreMLInput]) async throws -> [CoreMLOutput] {
        lastLengths = inputs.map { $0.tokenIDs.count }
        let dim = 4
        return inputs.map { inp in
            CoreMLOutput(values: Array(repeating: 0, count: inp.tokenIDs.count * dim), shape: [inp.tokenIDs.count, dim])
        }
    }
}

@Suite("Batch Padding")
struct BatchPaddingTestsSuite {
@Test
func batchPadding_producesEqualLengths() async throws {
    let backend = RecordingBackend()
    let vocab = Vocabulary(tokens: ["[PAD]","a","b","c","d","e","f"]) // PAD for batch padding
    let tokenizer = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
    let cfg = EmbeddingConfiguration(
        maxTokens: 16,
        paddingStrategy: .batch,
        includeSpecialTokens: false,
        poolingStrategy: .mean,
        normalizeOutput: false
    )
    let model = AppleEmbeddingModel(
        backend: backend,
        tokenizer: tokenizer,
        configuration: cfg,
        id: ModelID(provider: "test", name: "batchpad", version: "1.0"),
        dimensions: 4,
        device: .cpu
    )

    let texts = ["a", "a b c d e f", "a b"]
    _ = try await model.embedBatch(texts, options: BatchOptions())

    let lengths = await backend.lastLengths
    #expect(!lengths.isEmpty)
    #expect(lengths.dropFirst().allSatisfy { $0 == lengths.first })
}
}
