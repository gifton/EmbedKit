// EmbedKitV2 - WordPiece Tokenizer (Scaffold)

import Foundation

public struct WordPieceTokenizer: Tokenizer {
    public let vocabulary: Vocabulary
    public let unkToken: String
    public let lowercase: Bool

    public init(vocabulary: Vocabulary, unkToken: String = "[UNK]", lowercase: Bool = true) {
        self.vocabulary = vocabulary
        self.unkToken = unkToken
        self.lowercase = lowercase
    }

    public var vocabularySize: Int { vocabulary.count }

    public var specialTokens: SpecialTokens { SpecialTokens(cls: .init(text: "[CLS]", id: vocabulary["[CLS]"] ?? 101),
                                                           sep: .init(text: "[SEP]", id: vocabulary["[SEP]"] ?? 102),
                                                           pad: .init(text: "[PAD]", id: vocabulary["[PAD]"] ?? 0),
                                                           unk: .init(text: unkToken, id: vocabulary[unkToken] ?? 100),
                                                           mask: .init(text: "[MASK]", id: vocabulary["[MASK]"] ?? 103)) }

    public func encode(_ text: String, config: TokenizerConfig) async throws -> TokenizedText {
        throw EmbedKitError.tokenizationFailed("WordPieceTokenizer.encode not implemented (scaffold)")
    }

    public func decode(_ ids: [Int]) async throws -> String {
        // Scaffold: minimal decode via vocab reverse lookup; unknowns as [UNK]
        let pieces = ids.map { vocabulary[id: $0] ?? unkToken }
        return pieces.joined(separator: " ")
    }
}

