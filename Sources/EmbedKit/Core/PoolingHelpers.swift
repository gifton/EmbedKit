// EmbedKit - Pooling Helpers
// CPU pooling operations with Accelerate optimization

import Foundation

public enum PoolingHelpers {

    /// Mean-pool a flattened sequence (row-major: tokens x dim).
    ///
    /// Uses Accelerate/vDSP for SIMD-optimized computation.
    /// - Parameters:
    ///   - sequence: Flattened token embeddings [t0_d0, t0_d1, ..., t1_d0, ...]
    ///   - tokens: Number of tokens (rows)
    ///   - dim: Embedding dimension (cols)
    ///   - mask: Optional attention mask (1 = keep, 0 = ignore)
    /// - Returns: Pooled vector of length `dim`
    public static func mean(sequence: [Float], tokens: Int, dim: Int, mask: [Int]? = nil) -> [Float] {
        AccelerateBLAS.meanPool(sequence: sequence, tokens: tokens, dim: dim, mask: mask)
    }

    /// CLS pooling: returns the first token vector.
    public static func cls(sequence: [Float], tokens: Int, dim: Int) -> [Float] {
        precondition(tokens >= 1 && dim >= 1)
        precondition(sequence.count == tokens * dim, "sequence shape mismatch")
        let start = 0
        return Array(sequence[start..<(start + dim)])
    }

    /// Max-pool across tokens (optionally masked).
    ///
    /// Uses Accelerate/vDSP for SIMD-optimized computation.
    /// If mask selects no tokens, falls back to unmasked max across all tokens.
    public static func max(sequence: [Float], tokens: Int, dim: Int, mask: [Int]? = nil) -> [Float] {
        AccelerateBLAS.maxPool(sequence: sequence, tokens: tokens, dim: dim, mask: mask)
    }

    /// L2-normalize a vector (returns input if magnitude is zero).
    ///
    /// Uses Accelerate/vDSP for SIMD-optimized computation.
    public static func normalize(_ v: [Float]) -> [Float] {
        AccelerateBLAS.normalize(v)
    }

    /// Attention-weighted pooling.
    ///
    /// Uses Accelerate/vDSP for SIMD-optimized computation.
    /// - Parameters:
    ///   - sequence: Flattened token embeddings [t0_d0, t0_d1, ..., t1_d0, ...]
    ///   - weights: Attention weights per token
    ///   - tokens: Number of tokens (rows)
    ///   - dim: Embedding dimension (cols)
    /// - Returns: Weighted pooled vector of length `dim`
    public static func attention(sequence: [Float], weights: [Float], tokens: Int, dim: Int) -> [Float] {
        AccelerateBLAS.attentionPool(sequence: sequence, weights: weights, tokens: tokens, dim: dim)
    }
}
