// EmbedKit - Batch Tokenization Tests

import Testing
import Foundation
@testable import EmbedKit

@Suite("Batch Tokenization")
struct BatchTokenizationTests {

    // MARK: - Basic Batch Tests

    @Test("Empty batch returns empty array")
    func emptyBatch() async throws {
        let vocab = MockVocabulary.standard
        let tokenizer = WordPieceTokenizer(vocabulary: vocab)
        let config = TokenizerConfig()

        let results = try await tokenizer.encodeBatch([], config: config)
        #expect(results.isEmpty)
    }

    @Test("Single item batch matches single encode")
    func singleItemBatch() async throws {
        let vocab = MockVocabulary.standard
        let tokenizer = WordPieceTokenizer(vocabulary: vocab)
        let config = TokenizerConfig()

        let text = "hello world"
        let single = try await tokenizer.encode(text, config: config)
        let batch = try await tokenizer.encodeBatch([text], config: config)

        #expect(batch.count == 1)
        #expect(batch[0].ids == single.ids)
        #expect(batch[0].tokens == single.tokens)
    }

    @Test("Batch preserves input order")
    func batchPreservesOrder() async throws {
        let vocab = MockVocabulary.standard
        let tokenizer = WordPieceTokenizer(vocabulary: vocab)
        let config = TokenizerConfig()

        let texts = ["first", "second", "third", "fourth", "fifth"]
        let batch = try await tokenizer.encodeBatch(texts, config: config)

        #expect(batch.count == 5)

        // Verify each result matches individual encoding
        for (i, text) in texts.enumerated() {
            let single = try await tokenizer.encode(text, config: config)
            #expect(batch[i].ids == single.ids, "Mismatch at index \(i)")
        }
    }

    @Test("Batch handles varying text lengths")
    func varyingLengths() async throws {
        let vocab = MockVocabulary.standard
        let tokenizer = WordPieceTokenizer(vocabulary: vocab)
        let config = TokenizerConfig()

        let texts = [
            "a",
            "hello world this is a longer sentence",
            "medium length",
            "x"
        ]

        let results = try await tokenizer.encodeBatch(texts, config: config)
        #expect(results.count == 4)

        // Shortest texts should have fewer tokens
        #expect(results[0].ids.count < results[1].ids.count)
        #expect(results[3].ids.count < results[2].ids.count)
    }

    // MARK: - Concurrency Tests

    @Test("Batch with explicit concurrency limit")
    func explicitConcurrencyLimit() async throws {
        let vocab = MockVocabulary.standard
        let tokenizer = WordPieceTokenizer(vocabulary: vocab)
        let config = TokenizerConfig()

        let texts = (0..<20).map { "text number \($0)" }

        // Limit to 2 concurrent tasks
        let results = try await tokenizer.encodeBatch(texts, config: config, maxConcurrency: 2)
        #expect(results.count == 20)

        // Verify all results are correct
        for (i, text) in texts.enumerated() {
            let single = try await tokenizer.encode(text, config: config)
            #expect(results[i].ids == single.ids)
        }
    }

    @Test("Batch with high concurrency")
    func highConcurrency() async throws {
        let vocab = MockVocabulary.standard
        let tokenizer = WordPieceTokenizer(vocabulary: vocab)
        let config = TokenizerConfig()

        let texts = (0..<50).map { "sample text \($0) for parallel processing" }

        // Use all available processors
        let results = try await tokenizer.encodeBatch(texts, config: config, maxConcurrency: nil)
        #expect(results.count == 50)
    }

    @Test("Batch with single concurrency (sequential)")
    func sequentialFallback() async throws {
        let vocab = MockVocabulary.standard
        let tokenizer = WordPieceTokenizer(vocabulary: vocab)
        let config = TokenizerConfig()

        let texts = ["one", "two", "three"]

        let results = try await tokenizer.encodeBatch(texts, config: config, maxConcurrency: 1)
        #expect(results.count == 3)
    }

    // MARK: - Large Batch Tests

    @Test("Large batch performance", .tags(.performance))
    func largeBatchPerformance() async throws {
        let vocab = MockVocabulary.standard
        let tokenizer = WordPieceTokenizer(vocabulary: vocab)
        let config = TokenizerConfig()

        // Generate 100 texts of varying lengths
        let texts = (0..<100).map { i in
            String(repeating: "word ", count: (i % 10) + 1)
        }

        let start = Date()
        let results = try await tokenizer.encodeBatch(texts, config: config)
        let elapsed = Date().timeIntervalSince(start)

        #expect(results.count == 100)
        #expect(elapsed < 5.0, "Batch tokenization should complete in reasonable time")
    }

    // MARK: - Error Handling Tests

    @Test("Batch propagates tokenization errors")
    func batchPropagatesErrors() async throws {
        let vocab = MockVocabulary.standard
        let tokenizer = WordPieceTokenizer(vocabulary: vocab)

        // Config that throws on long input
        let config = TokenizerConfig(
            maxLength: 5,
            truncation: .none // Should throw on long input
        )
        let texts = ["short", "this is a much longer text that exceeds the maximum length"]

        await #expect(throws: EmbedKitError.self) {
            _ = try await tokenizer.encodeBatch(texts, config: config)
        }
    }

    // MARK: - Config Propagation Tests

    @Test("Batch respects tokenizer config")
    func batchRespectsConfig() async throws {
        let vocab = MockVocabulary.standard
        let tokenizer = WordPieceTokenizer(vocabulary: vocab)

        let config = TokenizerConfig(

            maxLength: 10,

            truncation: .end,

            addSpecialTokens: true
        )
        let texts = ["hello", "world"]
        let results = try await tokenizer.encodeBatch(texts, config: config)

        #expect(results.count == 2)

        // With addSpecialTokens, first token should be [CLS]
        let clsId = vocab["[CLS]"]
        if let clsId = clsId {
            #expect(results[0].ids.first == clsId)
            #expect(results[1].ids.first == clsId)
        }
    }
}

// MARK: - Mock Vocabulary

private enum MockVocabulary {
    static var standard: Vocabulary {
        let tokens = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            "hello", "world", "this", "is", "a",
            "test", "text", "for", "token", "##ization",
            "first", "second", "third", "fourth", "fifth",
            "short", "medium", "long", "##er", "sentence",
            "sample", "parallel", "processing", "number", "word",
            "one", "two", "three", "x"
        ]
        return Vocabulary(tokens: tokens)
    }
}
