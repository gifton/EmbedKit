// EmbedKit - HuggingFace tokenizer.json Parser
// Parses the standard HuggingFace tokenizer configuration format.

import Foundation

// MARK: - Top-Level Schema

/// Represents a HuggingFace `tokenizer.json` file.
public struct HFTokenizerJSON: Codable, Sendable {
    public let model: HFModel
    public let addedTokens: [HFAddedToken]?
    public let normalizer: HFNormalizer?
    public let preTokenizer: HFPreTokenizer?
    public let postProcessor: HFPostProcessor?

    private enum CodingKeys: String, CodingKey {
        case model
        case addedTokens = "added_tokens"
        case normalizer
        case preTokenizer = "pre_tokenizer"
        case postProcessor = "post_processor"
    }
}

// MARK: - Model

/// The tokenization model (BPE, WordPiece, Unigram, etc.)
public struct HFModel: Codable, Sendable {
    public let type: String
    public let vocab: [String: Int]?
    public let merges: [String]?
    public let unkToken: String?
    public let continuingSubwordPrefix: String?
    public let endOfWordSuffix: String?
    public let fuseUnk: Bool?
    public let byteLevel: Bool?

    private enum CodingKeys: String, CodingKey {
        case type
        case vocab
        case merges
        case unkToken = "unk_token"
        case continuingSubwordPrefix = "continuing_subword_prefix"
        case endOfWordSuffix = "end_of_word_suffix"
        case fuseUnk = "fuse_unk"
        case byteLevel = "byte_level"
    }

    /// Whether this is a BPE model.
    public var isBPE: Bool { type == "BPE" }
}

// MARK: - Added Tokens

/// A special token added to the vocabulary (e.g., <s>, </s>, <pad>).
public struct HFAddedToken: Codable, Sendable {
    public let id: Int
    public let content: String
    public let special: Bool?
    public let singleWord: Bool?
    public let lstrip: Bool?
    public let rstrip: Bool?
    public let normalized: Bool?

    private enum CodingKeys: String, CodingKey {
        case id, content, special
        case singleWord = "single_word"
        case lstrip, rstrip, normalized
    }
}

// MARK: - Normalizer

/// Text normalizer applied before tokenization.
public struct HFNormalizer: Codable, Sendable {
    public let type: String
    public let normalizers: [HFNormalizer]?

    /// The normalization mode derived from the type field.
    public var mode: NormalizationMode {
        switch type {
        case "NFC": return .nfc
        case "NFKC": return .nfkc
        case "NFD": return .nfd
        case "NFKD": return .nfkd
        case "Lowercase": return .lowercase
        case "Sequence":
            guard let normalizers = normalizers else { return .none }
            return .sequence(normalizers.map(\.mode))
        default: return .none
        }
    }

    public enum NormalizationMode: Sendable {
        case nfc
        case nfkc
        case nfd
        case nfkd
        case lowercase
        case sequence([NormalizationMode])
        case none
    }
}

// MARK: - Pre-Tokenizer

/// Splits text before BPE/WordPiece tokenization.
public struct HFPreTokenizer: Codable, Sendable {
    public let type: String
    public let addPrefixSpace: Bool?
    public let trimOffsets: Bool?
    public let useRegex: Bool?
    public let pattern: HFPattern?
    public let behavior: String?
    public let pretokenizers: [HFPreTokenizer]?

    private enum CodingKeys: String, CodingKey {
        case type
        case addPrefixSpace = "add_prefix_space"
        case trimOffsets = "trim_offsets"
        case useRegex = "use_regex"
        case pattern, behavior, pretokenizers
    }

    /// The pre-tokenization strategy derived from the config.
    public var strategy: PreTokenizerStrategy {
        switch type {
        case "ByteLevel":
            return .byteLevel(addPrefixSpace: addPrefixSpace ?? false)
        case "Whitespace":
            return .whitespace
        case "Split":
            return .split(pattern: pattern?.value, behavior: behavior)
        case "Sequence":
            guard let pretokenizers = pretokenizers else { return .none }
            return .sequence(pretokenizers.map(\.strategy))
        case "Digits":
            return .digits
        case "Punctuation":
            return .punctuation
        default:
            return .none
        }
    }

    public enum PreTokenizerStrategy: Sendable {
        case byteLevel(addPrefixSpace: Bool)
        case whitespace
        case split(pattern: String?, behavior: String?)
        case sequence([PreTokenizerStrategy])
        case digits
        case punctuation
        case none
    }
}

/// A regex or string pattern used in pre-tokenization.
public struct HFPattern: Codable, Sendable {
    public let regex: String?
    public let string: String?

    private enum CodingKeys: String, CodingKey {
        case regex = "Regex"
        case string = "String"
    }

    /// The pattern value (regex preferred, then string).
    public var value: String? { regex ?? string }
}

// MARK: - Post-Processor

/// Post-processor applied after tokenization (adds special tokens).
public struct HFPostProcessor: Codable, Sendable {
    public let type: String
    public let single: [HFTemplateItem]?
    public let pair: [HFTemplateItem]?
    public let specialTokens: [String: HFSpecialTokenEntry]?

    private enum CodingKeys: String, CodingKey {
        case type, single, pair
        case specialTokens = "special_tokens"
    }
}

/// A template item in post-processing (either a special token or a sequence placeholder).
public struct HFTemplateItem: Codable, Sendable {
    public let specialToken: HFTemplateSpecialToken?
    public let sequence: HFTemplateSequence?

    private enum CodingKeys: String, CodingKey {
        case specialToken = "SpecialToken"
        case sequence = "Sequence"
    }
}

public struct HFTemplateSpecialToken: Codable, Sendable {
    public let id: String
    public let typeId: Int?

    private enum CodingKeys: String, CodingKey {
        case id
        case typeId = "type_id"
    }
}

public struct HFTemplateSequence: Codable, Sendable {
    public let id: String
    public let typeId: Int?

    private enum CodingKeys: String, CodingKey {
        case id
        case typeId = "type_id"
    }
}

public struct HFSpecialTokenEntry: Codable, Sendable {
    public let id: String
    public let ids: [Int]
    public let tokens: [String]
}

// MARK: - Loader

extension HFTokenizerJSON {

    /// Load and parse a HuggingFace `tokenizer.json` file.
    public static func load(from url: URL) throws -> HFTokenizerJSON {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(HFTokenizerJSON.self, from: data)
    }

    /// Extract the vocabulary as an EmbedKit `Vocabulary`.
    public func buildVocabulary() -> Vocabulary? {
        guard let vocab = model.vocab else { return nil }
        var combined = vocab
        // Merge added tokens into vocabulary
        if let addedTokens = addedTokens {
            for token in addedTokens {
                combined[token.content] = token.id
            }
        }
        return Vocabulary(tokenToID: combined)
    }

    /// Extract the merge list as ordered pairs of strings.
    /// The index in the array is the merge priority (lower = higher priority).
    public func buildMergeRanks() -> [String: Int]? {
        guard let merges = model.merges else { return nil }
        var ranks: [String: Int] = [:]
        ranks.reserveCapacity(merges.count)
        for (i, merge) in merges.enumerated() {
            ranks[merge] = i
        }
        return ranks
    }

    /// Extract special tokens from the added_tokens list.
    public func buildSpecialTokens() -> SpecialTokens {
        guard let addedTokens = addedTokens else {
            return SpecialTokens()
        }

        var cls: SpecialTokens.Token?
        var sep: SpecialTokens.Token?
        var pad: SpecialTokens.Token?
        var unk: SpecialTokens.Token?
        var mask: SpecialTokens.Token?
        var bos: SpecialTokens.Token?
        var eos: SpecialTokens.Token?

        for token in addedTokens {
            let content = token.content
            let id = token.id

            switch content {
            // BERT-style
            case "[CLS]": cls = .init(text: content, id: id)
            case "[SEP]": sep = .init(text: content, id: id)
            case "[PAD]": pad = .init(text: content, id: id)
            case "[UNK]": unk = .init(text: content, id: id)
            case "[MASK]": mask = .init(text: content, id: id)
            // BPE-style
            case "<s>": bos = .init(text: content, id: id)
            case "</s>": eos = .init(text: content, id: id)
            case "<pad>": pad = pad ?? .init(text: content, id: id)
            case "<unk>": unk = unk ?? .init(text: content, id: id)
            case "<mask>": mask = mask ?? .init(text: content, id: id)
            // Llama/Mistral variants
            case "<|begin_of_text|>": bos = bos ?? .init(text: content, id: id)
            case "<|end_of_text|>": eos = eos ?? .init(text: content, id: id)
            default: break
            }
        }

        return SpecialTokens(
            cls: cls,
            sep: sep,
            pad: pad,
            unk: unk,
            mask: mask,
            bos: bos,
            eos: eos
        )
    }

    /// Determine the normalization to apply before tokenization.
    public func normalizationMode() -> HFNormalizer.NormalizationMode {
        normalizer?.mode ?? .none
    }

    /// Determine the pre-tokenizer strategy.
    public func preTokenizerStrategy() -> HFPreTokenizer.PreTokenizerStrategy {
        preTokenizer?.strategy ?? .none
    }
}
