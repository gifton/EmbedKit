// Test target: EmbedKitTests (migrated)
import Testing
@testable import EmbedKit

@Suite("Batch Truncation Strategy")
struct BatchTruncationStrategyTests {
    // Backend that returns [tokens, dim=4] with simple ascending values per token for easy reasoning
    actor AscendingBackend: CoreMLProcessingBackend {
        private(set) var isLoaded: Bool = false
        var memoryUsage: Int64 { 0 }
        func load() async throws { isLoaded = true }
        func unload() async throws { isLoaded = false }
        func process(_ input: CoreMLInput) async throws -> CoreMLOutput {
            let dim = 4
            var values: [Float] = []
            values.reserveCapacity(input.tokenIDs.count * dim)
            for i in 0..<input.tokenIDs.count {
                let f = Float(i + 1)
                values.append(contentsOf: [f, f, f, f])
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

    // Backend that uses token IDs as values to make truncation slices observable
    actor IDBackend: CoreMLProcessingBackend {
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
                values.append(contentsOf: [f, f, f, f])
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

    // Scaffolds a tokenizer with PAD and basic alphabet
    private func makeWPTokenizer() -> WordPieceTokenizer {
        let vocab = Vocabulary(tokens: [
            "[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]",
            "a", "b", "c", "d", "e", "f"
        ])
        return WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
    }

    @Test
    func batchTruncation_throwsOnNone_scaffold() async {
        // Expect: inputTooLong when preLen > targetLen and truncationStrategy == .none
        let backend = AscendingBackend()
        let tokenizer = makeWPTokenizer()
        let cfg = EmbeddingConfiguration(
            maxTokens: 4, // cap below input length to force truncation condition
            truncationStrategy: .none,
            paddingStrategy: .batch,
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            id: ModelID(provider: "test", name: "batch-trunc-none", version: "1.0"),
            dimensions: 4,
            device: .cpu
        )
        let opts = BatchOptions(
            bucketSize: 4 // targetLen will be min(maxTokens, ceil(len/4)*4) = 4
        )
        do {
            _ = try await model.embedBatch(["a b c d e"], options: opts) // len 5 > target 4
            #expect(Bool(false), "Expected inputTooLong when truncationStrategy == .none")
        } catch {
            guard let ek = error as? EmbedKitError else {
                return #expect(Bool(false), "Unexpected error type: \(error)")
            }
            switch ek {
            case .inputTooLong:
                #expect(true)
            default:
                #expect(Bool(false), "Unexpected error: \(ek)")
            }
        }
    }

    @Test
    func batchTruncation_respectsStart_scaffold() async throws {
        // Verify .start truncation keeps suffix(targetLen)
        let backend = IDBackend()
        let tokenizer = makeWPTokenizer()
        let cfg = EmbeddingConfiguration(
            maxTokens: 4, // force truncation to 4
            truncationStrategy: .start,
            paddingStrategy: .batch,
            includeSpecialTokens: false,
            poolingStrategy: .mean,
            normalizeOutput: false
        )
        let model = AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: cfg, id: ModelID(provider: "test", name: "batch-trunc-start", version: "1.0"), dimensions: 4, device: .cpu)
        let opts = BatchOptions(
            bucketSize: 4 // targetLen = 4
        )
        let out = try await model.embedBatch(["a b c d e f"], options: opts)
        #expect(out.count == 1)
        let v = out[0].vector
        #expect(v.count == 4)
        // Expect kept IDs: c(7), d(8), e(9), f(10); mean = (7+8+9+10)/4 = 8.5
        for x in v { #expect(abs(x - 8.5) <= 1e-5) }
    }

    @Test
    func batchTruncation_respectsMiddle_scaffold() async throws {
        // Verify .middle truncation keeps head/tail = (2, 2) for targetLen 4
        let backend = IDBackend()
        let tokenizer = makeWPTokenizer()
        let cfg = EmbeddingConfiguration(
            maxTokens: 4,
            truncationStrategy: .middle,
            paddingStrategy: .batch,
            includeSpecialTokens: false,
            poolingStrategy: .mean,
            normalizeOutput: false
        )
        let model = AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: cfg, id: ModelID(provider: "test", name: "batch-trunc-middle", version: "1.0"), dimensions: 4, device: .cpu)
        let opts = BatchOptions(
            bucketSize: 4 // targetLen = 4
        )
        let out = try await model.embedBatch(["a b c d e f"], options: opts)
        #expect(out.count == 1)
        let v = out[0].vector
        #expect(v.count == 4)
        // Expect kept IDs: a(5), b(6), e(9), f(10); mean = (5+6+9+10)/4 = 7.5
        for x in v { #expect(abs(x - 7.5) <= 1e-5) }
    }
}
