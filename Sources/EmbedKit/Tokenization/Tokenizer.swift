import Foundation

/// Protocol for text tokenization
public protocol Tokenizer: Sendable {
    /// Tokenize a single text
    /// - Parameter text: Input text
    /// - Returns: Tokenized input ready for the model
    func tokenize(_ text: String) async throws -> TokenizedInput

    /// Tokenize multiple texts
    /// - Parameter texts: Array of input texts
    /// - Returns: Array of tokenized inputs
    func tokenize(batch texts: [String]) async throws -> [TokenizedInput]

    /// Maximum sequence length supported
    var maxSequenceLength: Int { get }

    /// Vocabulary size
    var vocabularySize: Int { get }

    /// Special token IDs
    var specialTokens: SpecialTokens { get }
}

/// Special tokens used by the tokenizer
public struct SpecialTokens: Sendable {
    public let cls: Int
    public let sep: Int
    public let pad: Int
    public let unk: Int
    public let mask: Int?

    public init(
        cls: Int = 101,
        sep: Int = 102,
        pad: Int = 0,
        unk: Int = 100,
        mask: Int? = 103
    ) {
        self.cls = cls
        self.sep = sep
        self.pad = pad
        self.unk = unk
        self.mask = mask
    }
}

/// Basic whitespace tokenizer for testing
public actor SimpleTokenizer: Tokenizer {
    public let maxSequenceLength: Int
    public let vocabularySize: Int
    public let specialTokens: SpecialTokens

    private var vocabulary: [String: Int] = [:]
    private var inverseVocabulary: [Int: String] = [:]

    public init(
        maxSequenceLength: Int,
        vocabularySize: Int,
        specialTokens: SpecialTokens = SpecialTokens()
    ) {
        self.maxSequenceLength = maxSequenceLength
        self.vocabularySize = vocabularySize
        self.specialTokens = specialTokens

        // Initialize with special tokens
        vocabulary["[PAD]"] = specialTokens.pad
        vocabulary["[UNK]"] = specialTokens.unk
        vocabulary["[CLS]"] = specialTokens.cls
        vocabulary["[SEP]"] = specialTokens.sep
        if let mask = specialTokens.mask {
            vocabulary["[MASK]"] = mask
        }
    }

    public func tokenize(_ text: String) async throws -> TokenizedInput {
        // Simple whitespace tokenization
        let words = text.lowercased()
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }

        // Convert to token IDs with CLS and SEP
        var tokenIds = [specialTokens.cls]

        for word in words {
            if tokenIds.count >= maxSequenceLength - 1 {
                break
            }

            // Simple hash-based token ID assignment
            let tokenId = vocabulary[word] ?? (abs(word.hashValue) % vocabularySize)
            tokenIds.append(tokenId)
        }

        tokenIds.append(specialTokens.sep)

        // Pad to max length
        let originalLength = tokenIds.count
        while tokenIds.count < maxSequenceLength {
            tokenIds.append(specialTokens.pad)
        }

        // Create attention mask
        let attentionMask = Array(repeating: 1, count: originalLength) +
                           Array(repeating: 0, count: maxSequenceLength - originalLength)

        return TokenizedInput(
            tokenIds: Array(tokenIds.prefix(maxSequenceLength)),
            attentionMask: Array(attentionMask.prefix(maxSequenceLength)),
            tokenTypeIds: nil,
            originalLength: originalLength
        )
    }

    public func tokenize(batch texts: [String]) async throws -> [TokenizedInput] {
        var results: [TokenizedInput] = []
        results.reserveCapacity(texts.count)

        for text in texts {
            let tokenized = try await tokenize(text)
            results.append(tokenized)
        }

        return results
    }
}

/// Tokenizer configuration
public struct TokenizerConfiguration: Sendable {
    /// Whether to lowercase input text
    public let doLowerCase: Bool

    /// Whether to strip accents
    public let stripAccents: Bool

    /// Whether to add special tokens (CLS, SEP)
    public let addSpecialTokens: Bool

    /// Padding strategy
    public let paddingStrategy: PaddingStrategy

    /// Truncation strategy
    public let truncationStrategy: TruncationStrategy

    /// Maximum sequence length
    public let maxSequenceLength: Int

    /// Vocabulary size
    public let vocabularySize: Int

    public init(
        doLowerCase: Bool = true,
        stripAccents: Bool = true,
        addSpecialTokens: Bool = true,
        paddingStrategy: PaddingStrategy = .maxLength,
        truncationStrategy: TruncationStrategy = .longestFirst,
        maxSequenceLength: Int,
        vocabularySize: Int
    ) {
        self.doLowerCase = doLowerCase
        self.stripAccents = stripAccents
        self.addSpecialTokens = addSpecialTokens
        self.paddingStrategy = paddingStrategy
        self.truncationStrategy = truncationStrategy
        self.maxSequenceLength = maxSequenceLength
        self.vocabularySize = vocabularySize
    }
}

/// Padding strategy for tokenization
public enum PaddingStrategy: String, CaseIterable, Sendable {
    /// Pad to model's max length
    case maxLength
    /// Pad to longest sequence in batch
    case longest
    /// No padding
    case none
}

/// Truncation strategy for tokenization
public enum TruncationStrategy: String, CaseIterable, Sendable {
    /// Truncate longest sequence first
    case longestFirst
    /// Only truncate first sequence
    case onlyFirst
    /// Only truncate second sequence
    case onlySecond
    /// No truncation
    case none
}
