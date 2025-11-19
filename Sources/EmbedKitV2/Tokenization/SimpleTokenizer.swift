// EmbedKitV2 - Simple Tokenizer (Week 1)

import Foundation

public struct SimpleTokenizer: Tokenizer {
    public init() {}

    public var specialTokens: SpecialTokens { SpecialTokens() }
    public var vocabularySize: Int { 100_000 }

    public func encode(_ text: String, config: TokenizerConfig) async throws -> TokenizedText {
        var parts = tokenize(text)

        if config.addSpecialTokens {
            // Minimal simulation of [CLS] and [SEP]
            parts.insert("[CLS]", at: 0)
            parts.append("[SEP]")
        }

        // Truncation
        if config.maxLength > 0 && parts.count > config.maxLength {
            switch config.truncation {
            case .none:
                throw EmbedKitError.inputTooLong(length: parts.count, max: config.maxLength)
            case .end:
                parts = Array(parts.prefix(config.maxLength))
            case .start:
                parts = Array(parts.suffix(config.maxLength))
            case .middle:
                let keep = config.maxLength
                let head = keep / 2
                let tail = keep - head
                parts = Array(parts.prefix(head) + parts.suffix(tail))
            }
        }

        // Padding (no-op for Week 1; would pad with [PAD] to length)
        // We only set mask for existing tokens.

        let ids = parts.map { token in
            let h = stableHash64(token)
            return Int(h % UInt64(vocabularySize))
        }
        let mask = Array(repeating: 1, count: ids.count)
        return TokenizedText(ids: ids, tokens: parts, attentionMask: mask)
    }

    public func decode(_ ids: [Int]) async throws -> String {
        // Not needed for Week 1
        return ids.map { String($0) }.joined(separator: " ")
    }

    private func stableHash64(_ s: String) -> UInt64 {
        var h: UInt64 = 5381
        for b in s.utf8 { h = ((h << 5) &+ h) &+ UInt64(b) }
        return h
    }

    private func tokenize(_ text: String) -> [String] {
        // Extremely simple word+punct tokenization suitable for Week 1
        var tokens: [String] = []
        var current = ""
        func flush() { if !current.isEmpty { tokens.append(current); current.removeAll(keepingCapacity: true) } }
        let punct: Set<Character> = Set(",.!?;:\"'()[]{}")
        for ch in text {
            if ch.isWhitespace {
                flush()
            } else if punct.contains(ch) {
                flush()
                tokens.append(String(ch))
            } else {
                current.append(ch)
            }
        }
        flush()
        return tokens
    }
}
