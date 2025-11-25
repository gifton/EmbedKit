import Testing
import Foundation
@testable import EmbedKit

@Suite("Batch Performance")
struct BatchPerformanceTests {
    actor OverheadBackend: CoreMLProcessingBackend {
        private(set) var isLoaded: Bool = false
        var memoryUsage: Int64 { 0 }
        let perCallOverheadNs: UInt64 = 2_000_000 // ~2ms
        func load() async throws { isLoaded = true }
        func unload() async throws { isLoaded = false }
        func process(_ input: CoreMLInput) async throws -> CoreMLOutput {
            try await Task.sleep(nanoseconds: perCallOverheadNs)
            let dim = 4
            return CoreMLOutput(values: Array(repeating: 0, count: input.tokenIDs.count * dim), shape: [input.tokenIDs.count, dim])
        }
        func processBatch(_ inputs: [CoreMLInput]) async throws -> [CoreMLOutput] {
            // Single overhead for the whole batch
            try await Task.sleep(nanoseconds: perCallOverheadNs)
            let dim = 4
            return inputs.map { inp in
                CoreMLOutput(values: Array(repeating: 0, count: inp.tokenIDs.count * dim), shape: [inp.tokenIDs.count, dim])
            }
        }
    }

    @Test
    func batchFasterThanSequential_underBackendOverhead() async throws {
        let backend = OverheadBackend()
        let vocab = Vocabulary(tokens: ["[PAD]","a","b","c"]) // PAD for batch padding
        let tokenizer = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
        var cfg = EmbeddingConfiguration()
        cfg.includeSpecialTokens = false
        cfg.maxTokens = 64
        cfg.paddingStrategy = .batch
        cfg.poolingStrategy = .mean
        cfg.normalizeOutput = false
        let model = AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: cfg, id: ModelID(provider: "test", name: "perf", version: "1.0"), dimensions: 4, device: .cpu)

        let texts = Array(repeating: "a b c", count: 8)

        // Sequential
        let t0 = CFAbsoluteTimeGetCurrent()
        for t in texts { _ = try await model.embed(t) }
        let seqTime = CFAbsoluteTimeGetCurrent() - t0

        // Batched (one overhead)
        let t1 = CFAbsoluteTimeGetCurrent()
        _ = try await model.embedBatch(texts, options: .init())
        let batTime = CFAbsoluteTimeGetCurrent() - t1

        // With per-call overhead ~2ms, sequential should pay it 8x vs 1x in batch
        #expect(batTime < seqTime * 0.6)
    }
}

