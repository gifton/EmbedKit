import Testing
import Foundation
@testable import EmbedKit

@Suite("Tokenization Concurrency")
struct TokenizationConcurrencyTests {
    actor NoOpBackend: CoreMLProcessingBackend {
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

    struct SlowTokenizer: Tokenizer {
        let base = SimpleTokenizer()
        let delayNs: UInt64
        init(delayMs: UInt64 = 6) { self.delayNs = delayMs * 1_000_000 }
        var vocabularySize: Int { base.vocabularySize }
        var specialTokens: SpecialTokens { base.specialTokens }
        func encode(_ text: String, config: TokenizerConfig) async throws -> TokenizedText {
            try await Task.sleep(nanoseconds: delayNs)
            return try await base.encode(text, config: config)
        }
        func decode(_ ids: [Int]) async throws -> String { try await base.decode(ids) }
    }

    @Test
    func embedBatch_withConcurrency_isFaster() async throws {
        let backend = NoOpBackend()
        let tokenizer = SlowTokenizer(delayMs: 6)
        let cfg = EmbeddingConfiguration(
            maxTokens: 32,
            paddingStrategy: .none,
            includeSpecialTokens: false
        )
        let model = AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: cfg, id: ModelID(provider: "test", name: "tok-conc", version: "1.0"), dimensions: 4, device: .cpu)

        let texts = Array(repeating: "a b c d e f", count: 8)

        let optsSeq = BatchOptions(
            tokenizationConcurrency: 1
        )
        let t0 = CFAbsoluteTimeGetCurrent()
        _ = try await model.embedBatch(texts, options: optsSeq)
        let seq = CFAbsoluteTimeGetCurrent() - t0

        let optsConc = BatchOptions(
            tokenizationConcurrency: 4
        )
        let t1 = CFAbsoluteTimeGetCurrent()
        _ = try await model.embedBatch(texts, options: optsConc)
        let conc = CFAbsoluteTimeGetCurrent() - t1

        #expect(conc < seq * 0.75)
    }
}
