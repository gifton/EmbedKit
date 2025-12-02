// EmbedKit - Mock Embedding Model

import Foundation
import Logging

actor MockEmbeddingModel: EmbeddingModel {
    nonisolated let id: ModelID
    nonisolated let dimensions: Int
    nonisolated let device: ComputeDevice
    private let configuration: EmbeddingConfiguration
    private let logger = Logger(label: "EmbedKit.MockEmbeddingModel")

    private var metricsData = MetricsData()

    /// Creates a mock embedding model with default settings.
    init(
        dimensions: Int = 384,
        configuration: EmbeddingConfiguration = .default,
        device: ComputeDevice = .cpu,
        id: ModelID? = nil
    ) {
        self.dimensions = dimensions
        self.configuration = configuration
        self.device = device
        self.id = id ?? ModelID(provider: "mock", name: "test", version: "1.0")
    }

    func embed(_ text: String) async throws -> Embedding {
        let start = CFAbsoluteTimeGetCurrent()

        // Generate semantic-aware embedding using bag-of-words approach
        // This ensures texts with common words produce similar embeddings
        let vector = generateSemanticVector(for: text)

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        metricsData.record(tokenCount: text.count, time: elapsed)

        return Embedding(
            vector: configuration.normalizeOutput ? normalize(vector) : vector,
            metadata: EmbeddingMetadata(
                modelID: id,
                tokenCount: text.count,
                processingTime: elapsed,
                normalized: configuration.normalizeOutput,
                poolingStrategy: configuration.poolingStrategy,
                truncated: false,
                custom: [:]
            )
        )
    }

    /// Generates a semantic-aware vector using bag-of-words with hashing.
    /// Words are hashed to dimension indices, creating overlapping vectors for texts with common words.
    private func generateSemanticVector(for text: String) -> [Float] {
        var vector = [Float](repeating: 0, count: dimensions)

        // Tokenize: lowercase, split on non-alphanumeric, filter short words
        let baseWords = text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count >= 2 }
            .map { simpleStem($0) }

        // Expand words with semantic relations and flatten
        let allTerms = baseWords.flatMap { expandSemanticTerms($0) }

        // For each term, hash to multiple dimensions (improves distribution)
        for (termIndex, term) in allTerms.enumerated() {
            // Primary hash - deterministic based on word content
            let hash1 = term.utf8.reduce(0) { ($0 &* 31) &+ Int($1) }
            let hash2 = term.utf8.reduce(0) { ($0 &* 37) &+ Int($1) }

            // Map to dimension indices
            let idx1 = abs(hash1) % dimensions
            let idx2 = abs(hash2) % dimensions
            let idx3 = abs(hash1 &+ hash2) % dimensions

            // Weight: full weight for original words, reduced for expanded terms
            let isOriginal = termIndex < baseWords.count
            let weight: Float = isOriginal ? 1.0 : 0.3

            // Accumulate weights (TF-like weighting)
            vector[idx1] += 1.0 * weight
            vector[idx2] += 0.5 * weight
            vector[idx3] += 0.25 * weight
        }

        // Add small noise for uniqueness (based on full text hash)
        let textHash = text.utf8.reduce(0) { $0 &+ Int($1) }
        for i in 0..<dimensions {
            vector[i] += Float(sin(Double(textHash &+ i))) * 0.01
        }

        return vector
    }

    /// Simple stemming: removes common suffixes for better word matching.
    private func simpleStem(_ word: String) -> String {
        var w = word
        // Remove common plural/verb suffixes
        let suffixes = ["ing", "ed", "es", "s"]
        for suffix in suffixes {
            if w.count > suffix.count + 2 && w.hasSuffix(suffix) {
                w = String(w.dropLast(suffix.count))
                break
            }
        }
        return w
    }

    /// Expands a word to include semantically related terms for better mock similarity.
    /// This enables tests like "fruit" matching "apple" or "banana".
    private func expandSemanticTerms(_ word: String) -> [String] {
        // Semantic clusters for common test scenarios - more specific clusters
        // to avoid over-matching (e.g., "machine" shouldn't match "neural network")
        let semanticClusters: [[String]] = [
            // Fruits
            ["fruit", "apple", "banana", "orange", "grape", "berry", "mango"],
            // Vehicles
            ["vehicle", "car", "truck", "bus", "motorcycle", "automobile", "auto"],
            // Animals
            ["animal", "dog", "cat", "bird", "fish", "pet"],
            // Food
            ["food", "meal", "eat", "cuisine", "dish"],
            // Neural networks specifically (deep learning)
            ["neural", "network", "deep", "layer", "layers"],
            // Machine learning general (separate from neural nets)
            ["machine", "learn", "ml", "ai", "artificial", "intelligence"],
            // NLP
            ["nlp", "natural", "language", "text", "word", "process"],
            // Computer/software
            ["computer", "software", "program", "code"],
        ]

        var terms = [word]

        // Find which cluster(s) this word belongs to and add related terms
        for cluster in semanticClusters {
            if cluster.contains(word) {
                // Add all terms from the cluster with reduced weight
                terms.append(contentsOf: cluster)
            }
        }

        return terms
    }

    func embedBatch(_ texts: [String], options: BatchOptions) async throws -> [Embedding] {
        var results: [Embedding] = []
        results.reserveCapacity(texts.count)
        for text in texts {
            results.append(try await embed(text))
        }
        return results
    }

    func warmup() async throws { logger.debug("warmup()") }
    func release() async throws {}

    var metrics: ModelMetrics { metricsData.snapshot(memoryUsage: currentMemoryUsage()) }

    func resetMetrics() async throws { metricsData = MetricsData() }

    private func normalize(_ v: [Float]) -> [Float] {
        let sumSquares = v.reduce(0) { $0 + $1 * $1 }
        let mag = sqrt(max(1e-12, sumSquares))
        return v.map { $0 / mag }
    }
}
