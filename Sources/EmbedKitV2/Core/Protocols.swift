// EmbedKitV2 - Core Protocols (Week 1)

import Foundation

// MARK: - EmbeddingModel Protocol

/// Fundamental abstraction for embedding models used by EmbedKitV2.
/// Conforming types must be actors to ensure thread safety.
public protocol EmbeddingModel: Actor {
    nonisolated var id: ModelID { get }
    nonisolated var dimensions: Int { get }
    nonisolated var device: ComputeDevice { get }

    func embed(_ text: String) async throws -> Embedding
    func embedBatch(_ texts: [String], options: BatchOptions) async throws -> [Embedding]

    func warmup() async throws
    func release() async throws

    var metrics: ModelMetrics { get async }

    /// Reset any internal metrics collected so far (useful for benchmark warmup).
    func resetMetrics() async throws
}

// MARK: - Tokenizer Protocol

/// Tokenization interface used to convert text to model-ready inputs.
public protocol Tokenizer: Sendable {
    func encode(_ text: String, config: TokenizerConfig) async throws -> TokenizedText
    func decode(_ ids: [Int]) async throws -> String
    var vocabularySize: Int { get }
    var specialTokens: SpecialTokens { get }
}

// MARK: - Model Backend Protocol

/// Abstraction of the underlying compute backend (CoreML/ONNX/etc.)
public protocol ModelBackend: Actor {
    associatedtype Input
    associatedtype Output

    func process(_ input: Input) async throws -> Output
    func processBatch(_ inputs: [Input]) async throws -> [Output]

    func load() async throws
    func unload() async throws

    var isLoaded: Bool { get }
    var memoryUsage: Int64 { get }
}
