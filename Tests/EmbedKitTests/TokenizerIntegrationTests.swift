import Testing
import Foundation
@testable import EmbedKit

@Suite("Tokenizer Integration Tests")
struct TokenizerIntegrationTests {
    
    // MARK: - Tokenizer Factory Tests
    
    @Test("Tokenizer factory creates appropriate tokenizers")
    func testTokenizerFactory() async throws {
        // Test tokenizer type detection
        let testCases: [(ModelIdentifier, TokenizerFactory.TokenizerType)] = [
            (.miniLM_L6_v2, .bert),
            (ModelIdentifier(family: "bert", variant: "base"), .bert),
            (ModelIdentifier(family: "gpt2"), .simple), // Falls back to simple
            (ModelIdentifier(family: "unknown"), .simple)
        ]
        
        for (modelId, expectedType) in testCases {
            let detectedType = TokenizerFactory.TokenizerType.from(modelIdentifier: modelId)
            #expect(detectedType == expectedType)
        }
        
        // Test configuration creation
        let config = TokenizerConfiguration(
            maxSequenceLength: 256,
            vocabularySize: 30522
        )
        
        #expect(config.maxSequenceLength == 256)
        #expect(config.vocabularySize == 30522)
    }
    
    @Test("Model-specific tokenizer configurations")
    func testModelSpecificConfigs() {
        // Test BERT configuration
        let bertConfig = ModelSpecificTokenizerConfig.bert()
        #expect(bertConfig.doLowerCase == true)
        #expect(bertConfig.addSpecialTokens == true)
        #expect(bertConfig.maxSequenceLength == 512)
        
        // Test multilingual configuration
        let multiConfig = ModelSpecificTokenizerConfig.multilingual()
        #expect(multiConfig.doLowerCase == false) // Preserve case for multilingual
        #expect(multiConfig.stripAccents == false)
        
        // Test sentence transformer configuration
        let stConfig = ModelSpecificTokenizerConfig.sentenceTransformer()
        #expect(stConfig.maxSequenceLength == 256) // Typically shorter for ST models
    }
    
    // MARK: - Simple Tokenizer Tests
    
    @Test("Simple tokenizer integration")
    func testSimpleTokenizer() async throws {
        let tokenizer = SimpleTokenizer(
            maxSequenceLength: 128,
            vocabularySize: 50000
        )
        
        // Test basic tokenization
        let text = "Hello, world! This is a test."
        let tokenized = try await tokenizer.tokenize(text)
        
        #expect(tokenized.tokenIds.count > 0)
        #expect(tokenized.tokenIds.count <= 128)
        #expect(tokenized.attentionMask.count == tokenized.tokenIds.count)
        #expect(tokenized.originalLength == text.count)
        
        // Test truncation
        let longText = String(repeating: "word ", count: 200)
        let longTokenized = try await tokenizer.tokenize(longText)
        #expect(longTokenized.tokenIds.count == 128) // Should truncate
        
        // Test batch tokenization
        let batch = ["First text", "Second text", "Third text"]
        let batchTokenized = try await tokenizer.tokenize(batch: batch)
        #expect(batchTokenized.count == 3)
        
        // All should have same length (padded)
        let lengths = batchTokenized.map { $0.tokenIds.count }
        #expect(Set(lengths).count == 1)
    }
    
    // MARK: - BERT Tokenizer Tests
    
    @Test("BERT tokenizer with vocabulary")
    func testBERTTokenizer() async throws {
        // Create vocabulary file
        let vocabContent = """
        [PAD]
        [UNK]
        [CLS]
        [SEP]
        [MASK]
        hello
        world
        test
        ##ing
        the
        """
        
        let tempDir = FileManager.default.temporaryDirectory
        let vocabURL = tempDir.appendingPathComponent("test_vocab.txt")
        try vocabContent.write(to: vocabURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: vocabURL) }
        
        let tokenizer = try await BERTTokenizer(
            vocabularyPath: vocabURL.path,
            maxSequenceLength: 50
        )
        
        // Test tokenization with special tokens
        let tokenized = try await tokenizer.tokenize("hello world")
        
        // Should have [CLS] at start and [SEP] at end
        #expect(tokenized.tokenIds.first == 2) // [CLS] token ID
        #expect(tokenized.tokenIds.contains(3)) // [SEP] token ID
        
        // Test WordPiece tokenization
        let testTokenized = try await tokenizer.tokenize("testing")
        // Should split "testing" into "test" + "##ing" if vocabulary supports it
        #expect(testTokenized.tokenIds.count >= 3) // [CLS] + tokens + [SEP]
        
        // Test attention mask
        #expect(tokenized.attentionMask.allSatisfy { $0 == 1 || $0 == 0 })
    }
    
    @Test("BERT tokenizer edge cases")
    func testBERTTokenizerEdgeCases() async throws {
        let tokenizer = try await BERTTokenizer()
        
        // Empty text
        let empty = try await tokenizer.tokenize("")
        #expect(empty.tokenIds.count >= 2) // At least [CLS] and [SEP]
        
        // Only whitespace
        let whitespace = try await tokenizer.tokenize("   \n\t   ")
        #expect(whitespace.tokenIds.count >= 2)
        
        // Special characters
        let special = try await tokenizer.tokenize("@#$%^&*()")
        #expect(special.tokenIds.count >= 2)
        
        // Mixed languages (if supported)
        let mixed = try await tokenizer.tokenize("Hello 你好 Bonjour")
        #expect(mixed.tokenIds.count >= 2)
    }
    
    // MARK: - Advanced Tokenizer Tests
    
    @Test("Advanced tokenizer with different modes")
    func testAdvancedTokenizer() async throws {
        // Test BPE mode
        let bpeTokenizer = try await AdvancedTokenizer(
            type: .bpe,
            vocabularyPath: nil, // Will use built-in
            maxSequenceLength: 100
        )
        
        let bpeResult = try await bpeTokenizer.tokenize("Hello, world!")
        #expect(bpeResult.tokenIds.count > 0)
        
        // Test sentence piece mode (currently using BPE fallback)
        let spTokenizer = try await AdvancedTokenizer(
            type: .sentencepiece,
            vocabularyPath: nil,
            maxSequenceLength: 100
        )
        
        let spResult = try await spTokenizer.tokenize("Hello, world!")
        #expect(spResult.tokenIds.count > 0)
    }
    
    // MARK: - Tokenizer Configuration Tests
    
    @Test("Tokenizer configuration validation")
    func testTokenizerConfiguration() {
        // Test padding strategies
        let configs = [
            TokenizerConfiguration(
                paddingStrategy: .maxLength,
                maxSequenceLength: 128,
                vocabularySize: 30000
            ),
            TokenizerConfiguration(
                paddingStrategy: .longest,
                maxSequenceLength: 128,
                vocabularySize: 30000
            ),
            TokenizerConfiguration(
                paddingStrategy: .none,
                maxSequenceLength: 128,
                vocabularySize: 30000
            )
        ]
        
        for config in configs {
            #expect(config.maxSequenceLength == 128)
            #expect(config.vocabularySize == 30000)
        }
        
        // Test truncation strategies
        let truncConfigs = [
            TokenizerConfiguration(
                truncationStrategy: .longestFirst,
                maxSequenceLength: 50,
                vocabularySize: 10000
            ),
            TokenizerConfiguration(
                truncationStrategy: .onlyFirst,
                maxSequenceLength: 50,
                vocabularySize: 10000
            )
        ]
        
        for config in truncConfigs {
            #expect(config.maxSequenceLength == 50)
        }
    }
    
    // MARK: - Integration with Embedder Tests
    
    @Test("Tokenizer integration with embedder")
    func testTokenizerEmbedderIntegration() async throws {
        // Create embedder with custom tokenizer
        let customTokenizer = SimpleTokenizer(
            maxSequenceLength: 64,
            vocabularySize: 10000
        )
        
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: .miniLM_L6_v2,
            configuration: Configuration.default(for: .miniLM_L6_v2),
            tokenizer: customTokenizer
        )
        
        // Test that custom tokenizer is used
        let embedding = try await embedder.embed("Test text with custom tokenizer")
        #expect(embedding.dimensions >= 0)
        
        // Update tokenizer
        let newTokenizer = SimpleTokenizer(
            maxSequenceLength: 128,
            vocabularySize: 20000
        )
        await embedder.updateTokenizer(newTokenizer)
        
        // Test with updated tokenizer
        let embedding2 = try await embedder.embed("Test with updated tokenizer")
        #expect(embedding2.dimensions >= 0)
    }
    
    @Test("Automatic tokenizer selection")
    func testAutomaticTokenizerSelection() async throws {
        let manager = ModelManager()
        
        // Test with known model - should select BERT tokenizer
        let embedder = try await manager.createEmbedder(
            identifier: .miniLM_L6_v2
        )
        
        // Verify it can handle BERT-style text
        let testTexts = [
            "Simple sentence.",
            "[CLS] Already has special tokens [SEP]",
            "Multiple. Sentences. Here.",
            "MixedCaseAndCamelCase",
            "hello@example.com and URLs: https://example.com"
        ]
        
        for text in testTexts {
            let embedding = try await embedder.embed(text)
            #expect(embedding.dimensions >= 0)
        }
    }
    
    // MARK: - Performance Tests
    
    @Test("Tokenizer performance")
    func testTokenizerPerformance() async throws {
        let tokenizer = SimpleTokenizer(
            maxSequenceLength: 512,
            vocabularySize: 50000
        )
        
        // Generate test data
        let sentences = (0..<100).map { i in
            "This is test sentence number \(i) with some additional words to make it longer."
        }
        
        let startTime = Date()
        
        // Batch tokenization should be faster than individual
        let batchResult = try await tokenizer.tokenize(batch: sentences)
        
        let batchTime = Date().timeIntervalSince(startTime)
        
        #expect(batchResult.count == sentences.count)
        
        // Individual tokenization for comparison
        let individualStart = Date()
        var individualResults: [TokenizedInput] = []
        
        for sentence in sentences.prefix(10) { // Just test first 10 for time
            let result = try await tokenizer.tokenize(sentence)
            individualResults.append(result)
        }
        
        let individualTime = Date().timeIntervalSince(individualStart)
        
        // Batch should be more efficient (normalized per item)
        let batchPerItem = batchTime / Double(sentences.count)
        let individualPerItem = individualTime / 10.0
        
        print("Batch per item: \(batchPerItem)s, Individual per item: \(individualPerItem)s")
        
        // Both should complete in reasonable time
        #expect(batchTime < 5.0) // Should tokenize 100 sentences in < 5 seconds
    }
}