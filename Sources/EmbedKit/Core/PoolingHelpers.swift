// EmbedKit - Pooling Helpers

import Foundation

public enum PoolingHelpers {

    /// Mean-pool a flattened sequence (row-major: tokens x dim).
    /// - Parameters:
    ///   - sequence: Flattened token embeddings [t0_d0, t0_d1, ..., t1_d0, ...]
    ///   - tokens: Number of tokens (rows)
    ///   - dim: Embedding dimension (cols)
    ///   - mask: Optional attention mask (1 = keep, 0 = ignore)
    /// - Returns: Pooled vector of length `dim`
    public static func mean(sequence: [Float], tokens: Int, dim: Int, mask: [Int]? = nil) -> [Float] {
        precondition(tokens >= 1 && dim >= 1)
        precondition(sequence.count == tokens * dim, "sequence shape mismatch")

        var acc = Array<Float>(repeating: 0, count: dim)
        var count: Int = 0

        if let mask {
            precondition(mask.count == tokens, "mask length mismatch")
            for t in 0..<tokens where mask[t] != 0 {
                let base = t * dim
                for d in 0..<dim { acc[d] += sequence[base + d] }
                count += 1
            }
        } else {
            for t in 0..<tokens {
                let base = t * dim
                for d in 0..<dim { acc[d] += sequence[base + d] }
            }
            count = tokens
        }

        if count == 0 {
            // Fallback to unmasked mean: aggregate across all tokens
            for t in 0..<tokens {
                let base = t * dim
                for d in 0..<dim { acc[d] += sequence[base + d] }
            }
            count = tokens
        }
        let inv = 1.0 / Float(count)
        for d in 0..<dim { acc[d] *= inv }
        return acc
    }

    /// CLS pooling: returns the first token vector.
    public static func cls(sequence: [Float], tokens: Int, dim: Int) -> [Float] {
        precondition(tokens >= 1 && dim >= 1)
        precondition(sequence.count == tokens * dim, "sequence shape mismatch")
        let start = 0
        return Array(sequence[start..<(start + dim)])
    }

    /// Max-pool across tokens (optionally masked). If mask selects no tokens,
    /// falls back to unmasked max across all tokens.
    public static func max(sequence: [Float], tokens: Int, dim: Int, mask: [Int]? = nil) -> [Float] {
        precondition(tokens >= 1 && dim >= 1)
        precondition(sequence.count == tokens * dim, "sequence shape mismatch")
        var acc = Array<Float>(repeating: -Float.greatestFiniteMagnitude, count: dim)
        var selected = 0
        if let mask {
            precondition(mask.count == tokens, "mask length mismatch")
            for t in 0..<tokens where mask[t] != 0 {
                let base = t * dim
                for d in 0..<dim { acc[d] = Swift.max(acc[d], sequence[base + d]) }
                selected += 1
            }
        }
        if selected == 0 {
            for t in 0..<tokens {
                let base = t * dim
                for d in 0..<dim { acc[d] = Swift.max(acc[d], sequence[base + d]) }
            }
        }
        return acc
    }

    /// L2-normalize a vector (returns input if magnitude is zero).
    public static func normalize(_ v: [Float]) -> [Float] {
        let s = v.reduce(0) { $0 + $1 * $1 }
        let mag = sqrt(Swift.max(1e-12, s))
        return v.map { $0 / Float(mag) }
    }
}
