import Testing
@testable import EmbedKit

@Suite("BERT Tokenizer Tests")
struct BERTTokenizerTests {
    
    @Test("BERT tokenizer initialization")
    func testInitialization() async throws {
        let tokenizer = try await BERTTokenizer()
        
        #expect(await tokenizer.maxSequenceLength == 512)
        #expect(tokenizer.vocabularySize > 0)
        #expect(tokenizer.specialTokens.cls == 101)
        #expect(tokenizer.specialTokens.sep == 102)
        #expect(tokenizer.specialTokens.pad == 0)
    }
    
    @Test("Basic tokenization")
    func testBasicTokenization() async throws {
        let tokenizer = try await BERTTokenizer()
        let text = "Hello world!"
        
        let result = try await tokenizer.tokenize(text)
        
        // Should have [CLS] at start and [SEP] at end
        #expect(result.tokenIds.first == 101) // [CLS]
        #expect(result.tokenIds[result.originalLength - 1] == 102) // [SEP]
        
        // Should have correct attention mask
        #expect(result.attentionMask.prefix(result.originalLength).allSatisfy { $0 == 1 })
        #expect(result.attentionMask.suffix(from: result.originalLength).allSatisfy { $0 == 0 })
        
        // Should be padded to max length
        #expect(result.tokenIds.count == 512)
        #expect(result.attentionMask.count == 512)
    }
    
    @Test("WordPiece tokenization")
    func testWordPieceTokenization() async throws {
        let tokenizer = try await BERTTokenizer()
        
        // Test with a word that should be split into subwords
        let text = "unbelievable"
        let result = try await tokenizer.tokenize(text)
        
        // The word might be split into subwords
        // We can't test exact token IDs without a real vocabulary,
        // but we can verify the structure
        #expect(result.tokenIds.count == 512)
        #expect(result.originalLength > 2) // At least [CLS], some tokens, [SEP]
    }
    
    @Test("Batch tokenization")
    func testBatchTokenization() async throws {
        let tokenizer = try await BERTTokenizer()
        let texts = [
            "Hello world!",
            "This is a test.",
            "BERT tokenization works!"
        ]
        
        let results = try await tokenizer.tokenize(batch: texts)
        
        #expect(results.count == 3)
        for result in results {
            #expect(result.tokenIds.count == 512)
            #expect(result.attentionMask.count == 512)
            #expect(result.tokenIds.first == 101) // [CLS]
        }
    }
    
    @Test("Long text truncation")
    func testLongTextTruncation() async throws {
        let tokenizer = try await BERTTokenizer()
        
        // Create a very long text
        let longText = String(repeating: "This is a very long sentence. ", count: 100)
        let result = try await tokenizer.tokenize(longText)
        
        // Should be truncated to max length
        #expect(result.tokenIds.count == 512)
        #expect(result.attentionMask.count == 512)
        #expect(result.originalLength == 512)
    }
    
    @Test("Special characters handling")
    func testSpecialCharacters() async throws {
        let tokenizer = try await BERTTokenizer()
        
        let text = "Test with special chars: @#$% & *()!"
        let result = try await tokenizer.tokenize(text)
        
        // Should handle special characters without crashing
        #expect(result.tokenIds.count == 512)
        #expect(result.originalLength > 2)
    }
    
    @Test("Empty text handling")
    func testEmptyText() async throws {
        let tokenizer = try await BERTTokenizer()
        
        let result = try await tokenizer.tokenize("")
        
        // Should still have [CLS] and [SEP]
        #expect(result.originalLength == 2)
        #expect(result.tokenIds[0] == 101) // [CLS]
        #expect(result.tokenIds[1] == 102) // [SEP]
    }
    
    @Test("Case sensitivity")
    func testCaseSensitivity() async throws {
        // Test with lowercase enabled (default)
        let lowercaseTokenizer = try await BERTTokenizer(doLowerCase: true)
        let result1 = try await lowercaseTokenizer.tokenize("HELLO World")
        
        // Test with lowercase disabled
        let caseSensitiveTokenizer = try await BERTTokenizer(doLowerCase: false)
        let result2 = try await caseSensitiveTokenizer.tokenize("HELLO World")
        
        // Both should tokenize successfully
        #expect(result1.tokenIds.count == 512)
        #expect(result2.tokenIds.count == 512)
    }
    
    @Test("Decode functionality")
    func testDecode() async throws {
        let tokenizer = try await BERTTokenizer()
        
        // Tokenize and then decode
        // Use simple text that's more likely to be in the default vocabulary
        let originalText = "the a"
        let tokenized = try await tokenizer.tokenize(originalText)
        let decoded = await tokenizer.decode(tokenized.tokenIds)
        
        // Should decode back to something similar (may have differences due to tokenization)
        #expect(decoded.lowercased().contains("the") || decoded.lowercased().contains("a") || decoded.lowercased().contains("[unk]"))
        
        // Test with punctuation
        let punctText = "!"
        let punctTokenized = try await tokenizer.tokenize(punctText)
        let punctDecoded = await tokenizer.decode(punctTokenized.tokenIds)
        #expect(punctDecoded.contains("!") || punctDecoded.contains("[unk]"))
    }
    
    @Test("Vocabulary statistics")
    func testVocabularyStats() async throws {
        let tokenizer = try await BERTTokenizer()
        let stats = await tokenizer.getVocabularyStats()
        
        #expect(stats.totalSize > 0)
        #expect(stats.specialTokens >= 5) // At least [PAD], [UNK], [CLS], [SEP], [MASK]
        #expect(stats.wholeWords >= 0)
        #expect(stats.subwords >= 0)
        #expect(stats.totalSize == stats.specialTokens + stats.wholeWords + stats.subwords)
    }
}

@Suite("Tokenizer Factory Tests")
struct TokenizerFactoryTests {
    
    @Test("Tokenizer type detection from model identifier")
    func testTokenizerTypeDetection() {
        // BERT models
        #expect(TokenizerFactory.TokenizerType.from(modelIdentifier: .miniLM_L6_v2) == .bert)
        #expect(TokenizerFactory.TokenizerType.from(modelIdentifier: .init(family: "bert-base-uncased")) == .bert)
        
        // Default to simple for unknown models
        #expect(TokenizerFactory.TokenizerType.from(modelIdentifier: .init(family: "unknown-model")) == .simple)
    }
    
    @Test("Create tokenizer by type")
    func testCreateTokenizerByType() async throws {
        // Test creating different tokenizer types
        let config = TokenizerConfiguration(
            maxSequenceLength: 512,
            vocabularySize: 30522
        )
        
        let simpleTokenizer = try await TokenizerFactory.createTokenizer(type: .simple, configuration: config)
        #expect(simpleTokenizer is SimpleTokenizer)
        
        let bertTokenizer = try await TokenizerFactory.createTokenizer(type: .bert, configuration: config)
        #expect(bertTokenizer is BERTTokenizer)
        
        let wordpieceTokenizer = try await TokenizerFactory.createTokenizer(type: .wordpiece, configuration: config)
        #expect(wordpieceTokenizer is BERTTokenizer) // WordPiece uses BERT implementation
    }
    
    @Test("Create tokenizer for model identifier")
    func testCreateTokenizerForModel() async throws {
        let tokenizer = try await TokenizerFactory.createTokenizer(for: .miniLM_L6_v2)
        #expect(tokenizer is BERTTokenizer)
        
        // Verify it has correct configuration
        if let bertTokenizer = tokenizer as? BERTTokenizer {
            #expect(await bertTokenizer.maxSequenceLength == 512)
        } else {
            #expect(tokenizer.maxSequenceLength == 512)
        }
    }
}