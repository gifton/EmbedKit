import Foundation

/// Pooling strategy for embeddings
///
/// After the transformer processes input tokens, we need to reduce the sequence
/// of token embeddings into a single fixed-size vector. The pooling strategy
/// determines how this reduction is performed.
///
/// Performance characteristics:
/// - `cls`: O(1) - fastest, just extracts first token
/// - `mean`: O(n) - requires averaging all tokens
/// - `max`: O(n) - requires comparing all tokens
/// - `attentionWeighted`: O(n) - most expensive but potentially most accurate
public enum PoolingStrategy: String, CaseIterable, Sendable {
    /// Average all token embeddings
    case mean
    /// Use only the CLS token embedding
    case cls
    /// Take the maximum value across all tokens
    case max
    /// Average pooling with attention weights
    case attentionWeighted
}
