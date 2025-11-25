// EmbedKit - SentencePiece Tokenizer

import Foundation

/// Minimal SentencePiece-style tokenizer scaffold.
/// - Uses the boundary marker "▁" (U+2581) to denote word starts.
/// - Greedy longest match against the provided vocabulary; falls back to character pieces
///   with the first character prefixed by "▁" for each word.
public struct SentencePieceTokenizer: Tokenizer {
    public let vocabulary: Vocabulary
    public let lowercase: Bool
    public let unkToken: String

    public init(vocabulary: Vocabulary, unkToken: String = "[UNK]", lowercase: Bool = true) {
        self.vocabulary = vocabulary
        self.lowercase = lowercase
        self.unkToken = unkToken
    }

    // Load from a SentencePiece .vocab-style file (token and optional score per line).
    public static func load(spVocabURL: URL, unkToken: String = "[UNK]", lowercase: Bool = true) throws -> SentencePieceTokenizer {
        let tokens = try Self.readSentencePieceVocab(spVocabURL)
        let vocab = Vocabulary(tokens: tokens)
        return SentencePieceTokenizer(vocabulary: vocab, unkToken: unkToken, lowercase: lowercase)
    }

    public var vocabularySize: Int { vocabulary.count }

    public var specialTokens: SpecialTokens {
        let cls = vocabulary["[CLS]"]
        let sep = vocabulary["[SEP]"]
        let pad = vocabulary["[PAD]"]
        let unk = vocabulary[unkToken]
        let mask = vocabulary["[MASK]"]
        return SpecialTokens(
            cls: cls.map { .init(text: "[CLS]", id: $0) },
            sep: sep.map { .init(text: "[SEP]", id: $0) },
            pad: pad.map { .init(text: "[PAD]", id: $0) },
            unk: unk.map { .init(text: unkToken, id: $0) },
            mask: mask.map { .init(text: "[MASK]", id: $0) }
        )
    }

    public func encode(_ text: String, config: TokenizerConfig) async throws -> TokenizedText {
        let normalized = lowercase ? text.lowercased() : text
        let words = normalized.split(whereSeparator: { $0.isWhitespace }).map(String.init)

        var pieces: [String] = []
        pieces.reserveCapacity(words.count * 2)
        for w in words {
            let wordPieces = spPiece(word: w)
            pieces.append(contentsOf: wordPieces)
        }

        if config.addSpecialTokens {
            if let cls = specialTokens.cls?.text { pieces.insert(cls, at: 0) }
            if let sep = specialTokens.sep?.text { pieces.append(sep) }
        }

        // Truncation
        if config.maxLength > 0 && pieces.count > config.maxLength {
            switch config.truncation {
            case .none:
                throw EmbedKitError.inputTooLong(length: pieces.count, max: config.maxLength)
            case .end:
                pieces = Array(pieces.prefix(config.maxLength))
            case .start:
                pieces = Array(pieces.suffix(config.maxLength))
            case .middle:
                let keep = config.maxLength
                let head = keep / 2
                let tail = keep - head
                pieces = Array(pieces.prefix(head) + pieces.suffix(tail))
            }
        }

        // IDs
        let unkId = specialTokens.unk?.id ?? 100
        var ids: [Int] = []
        ids.reserveCapacity(pieces.count)
        for p in pieces { ids.append(vocabulary[p] ?? unkId) }

        // Padding (only .max here)
        var mask = Array(repeating: 1, count: ids.count)
        if config.padding == .max, config.maxLength > 0, ids.count < config.maxLength {
            guard let pad = specialTokens.pad else {
                throw EmbedKitError.invalidConfiguration("Padding requires PAD token in vocabulary")
            }
            let padCount = config.maxLength - ids.count
            ids += Array(repeating: pad.id, count: padCount)
            pieces += Array(repeating: pad.text, count: padCount)
            mask += Array(repeating: 0, count: padCount)
        }

        return TokenizedText(ids: ids, tokens: pieces, attentionMask: mask)
    }

    public func decode(_ ids: [Int]) async throws -> String {
        // Convert tokens back to text: space before tokens starting with "▁"
        let specials: Set<String> = [
            specialTokens.cls?.text,
            specialTokens.sep?.text,
            specialTokens.pad?.text,
            specialTokens.mask?.text,
            specialTokens.bos?.text,
            specialTokens.eos?.text
        ].compactMap { $0 }.reduce(into: Set<String>()) { $0.insert($1) }

        let toks = ids.compactMap { vocabulary[id: $0] }.filter { !specials.contains($0) }
        var out = ""
        for t in toks {
            if t.hasPrefix("▁") {
                if !out.isEmpty { out.append(" ") }
                out.append(String(t.dropFirst()))
            } else {
                out.append(t)
            }
        }
        return out
    }

    // MARK: - Private helpers
    private func spPiece(word: String) -> [String] {
        // Greedy longest match for the whole word with leading boundary token
        let boundaryWord = "▁" + word
        if vocabulary[boundaryWord] != nil { return [boundaryWord] }

        // Otherwise, attempt to greedily match substrings from the boundary
        var pieces: [String] = []
        var remaining = boundaryWord
        while !remaining.isEmpty {
            var match: String? = nil
            var end = remaining.endIndex
            while end > remaining.startIndex {
                let cand = String(remaining[..<end])
                if vocabulary[cand] != nil { match = cand; break }
                end = remaining.index(before: end)
            }
            if let m = match {
                pieces.append(m)
                remaining = String(remaining[m.endIndex...])
            } else {
                // Fallback: take one scalar (or char) and mark unknown if not in vocab
                let first = String(remaining.prefix(1))
                if vocabulary[first] != nil { pieces.append(first) }
                else { pieces.append(unkToken) }
                remaining.removeFirst()
            }
        }
        return pieces
    }
}

// MARK: - File helpers
extension SentencePieceTokenizer {
    static func readSentencePieceVocab(_ url: URL) throws -> [String] {
        let data = try Data(contentsOf: url)
        guard let content = String(data: data, encoding: .utf8) else {
            throw EmbedKitError.tokenizationFailed("Invalid SentencePiece vocab encoding")
        }
        var tokens: [String] = []
        tokens.reserveCapacity(1024)
        for raw in content.split(whereSeparator: { $0 == "\n" || $0 == "\r" }) {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            if line.hasPrefix("#") { continue }
            // First column is token, rest (e.g., score) ignored
            let cols = line.split(separator: " ", omittingEmptySubsequences: true)
            if let tok = cols.first { tokens.append(String(tok)) }
        }
        return tokens
    }
}
