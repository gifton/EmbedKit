import Foundation

/// A basic WordPiece tokenizer compatible with BERT-style models.
/// Note: This is a simplified implementation for Week 1.
public struct BertTokenizer: Tokenizer {
    private let vocabulary: [String: Int]
    private let unkToken = "[UNK]"
    private let clsToken = "[CLS]"
    private let sepToken = "[SEP]"
    private let padToken = "[PAD]"
    
    public init(vocabulary: [String: Int]) {
        self.vocabulary = vocabulary
    }
    
    public func encode(_ text: String) async throws -> [Int] {
        // 1. Basic normalization (lowercase, strip accents - simplified)
        let normalized = text.lowercased()
        
        // 2. Split by whitespace
        let words = normalized.components(separatedBy: .whitespacesAndNewlines)
        
        var tokens: [String] = [clsToken]
        
        for word in words {
            if word.isEmpty { continue }
            
            // 3. WordPiece tokenization (greedy match)
            var start = word.startIndex
            var hasUnknown = false
            
            while start < word.endIndex {
                var end = word.endIndex
                var foundSubword: String?
                
                while start < end {
                    var subword = String(word[start..<end])
                    if start > word.startIndex {
                        subword = "##" + subword
                    }
                    
                    if vocabulary[subword] != nil {
                        foundSubword = subword
                        break
                    }
                    
                    end = word.index(before: end)
                }
                
                if let subword = foundSubword {
                    tokens.append(subword)
                    start = end
                } else {
                    hasUnknown = true
                    break
                }
            }
            
            if hasUnknown {
                tokens.append(unkToken)
            }
        }
        
        tokens.append(sepToken)
        
        // 4. Convert to IDs
        return tokens.map { vocabulary[$0] ?? vocabulary[unkToken] ?? 0 }
    }
    
    public func decode(_ tokens: [Int]) async throws -> String {
        // Reverse vocabulary lookup
        let idToToken = Dictionary(uniqueKeysWithValues: vocabulary.map { ($1, $0) })
        
        var words: [String] = []
        
        for id in tokens {
            guard let token = idToToken[id] else { continue }
            if [clsToken, sepToken, padToken].contains(token) { continue }
            
            if token.hasPrefix("##") {
                if var last = words.last {
                    last += String(token.dropFirst(2))
                    words[words.count - 1] = last
                }
            } else {
                words.append(token)
            }
        }
        
        return words.joined(separator: " ")
    }
}
