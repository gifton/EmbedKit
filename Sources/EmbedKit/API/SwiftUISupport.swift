// EmbedKit - SwiftUI Support
// Observable view models and property wrappers for SwiftUI integration

#if canImport(SwiftUI)
import SwiftUI

// MARK: - Embedding View Model

/// A SwiftUI-compatible view model for embedding operations.
///
/// This class provides observable state for loading, embedding, and displaying results
/// in a SwiftUI application. It's main-actor isolated for safe UI updates.
///
/// Example:
/// ```swift
/// struct EmbeddingView: View {
///     @State private var viewModel = EmbeddingViewModel()
///     @State private var inputText = ""
///
///     var body: some View {
///         VStack {
///             TextField("Enter text", text: $inputText)
///             Button("Embed") {
///                 Task {
///                     await viewModel.embed(inputText)
///                 }
///             }
///             if viewModel.isLoading {
///                 ProgressView(value: viewModel.progress)
///             }
///             if let embedding = viewModel.lastEmbedding {
///                 Text("Dimensions: \(embedding.dimensions)")
///             }
///         }
///     }
/// }
/// ```
@MainActor
@Observable
public final class EmbeddingViewModel {
    // MARK: - Published State

    /// Whether an embedding operation is currently in progress.
    public private(set) var isLoading = false

    /// Progress of the current operation (0.0 to 1.0).
    public private(set) var progress: Double = 0

    /// The most recent error that occurred, if any.
    public private(set) var error: Error?

    /// The most recently computed embedding.
    public private(set) var lastEmbedding: Embedding?

    /// The most recently computed batch of embeddings.
    public private(set) var lastBatchEmbeddings: [Embedding] = []

    // MARK: - Private State

    private let manager = ModelManager()
    private var model: (any EmbeddingModel)?

    // MARK: - Initialization

    public init() {}

    // MARK: - Single Embedding

    /// Embed a single text and update observable state.
    ///
    /// - Parameter text: The text to embed.
    /// - Returns: The computed embedding, or nil if an error occurred.
    @discardableResult
    public func embed(_ text: String) async -> Embedding? {
        isLoading = true
        error = nil
        progress = 0

        do {
            progress = 0.3
            if model == nil {
                model = try await manager.loadMockModel()
            }

            progress = 0.7
            let embedding = try await model!.embed(text)

            progress = 1.0
            lastEmbedding = embedding
            isLoading = false
            return embedding

        } catch {
            self.error = error
            isLoading = false
            return nil
        }
    }

    // MARK: - Batch Embedding

    /// Embed multiple texts and update observable state.
    ///
    /// - Parameter texts: The texts to embed.
    /// - Returns: The computed embeddings, or empty array if an error occurred.
    @discardableResult
    public func embedBatch(_ texts: [String]) async -> [Embedding] {
        isLoading = true
        error = nil
        progress = 0
        lastBatchEmbeddings = []

        do {
            progress = 0.2
            if model == nil {
                model = try await manager.loadMockModel()
            }

            progress = 0.4
            let embeddings = try await model!.embedBatch(texts, options: BatchOptions())

            progress = 1.0
            lastBatchEmbeddings = embeddings
            isLoading = false
            return embeddings

        } catch {
            self.error = error
            isLoading = false
            return []
        }
    }

    // MARK: - Semantic Search

    /// Search result containing the document index, text, and similarity score.
    public struct SearchResult: Identifiable {
        public let id = UUID()
        public let index: Int
        public let text: String
        public let score: Float
    }

    /// The most recent search results.
    public private(set) var searchResults: [SearchResult] = []

    /// Perform semantic search and update observable state.
    ///
    /// - Parameters:
    ///   - query: The search query.
    ///   - documents: The documents to search through.
    ///   - topK: Maximum results to return.
    /// - Returns: Array of search results.
    @discardableResult
    public func search(
        query: String,
        in documents: [String],
        topK: Int = 10
    ) async -> [SearchResult] {
        isLoading = true
        error = nil
        progress = 0
        searchResults = []

        do {
            progress = 0.3
            let results = try await manager.semanticSearch(
                query: query,
                in: documents,
                topK: topK
            )

            progress = 1.0
            searchResults = results.map {
                SearchResult(index: $0.index, text: documents[$0.index], score: $0.score)
            }
            isLoading = false
            return searchResults

        } catch {
            self.error = error
            isLoading = false
            return []
        }
    }

    // MARK: - Reset

    /// Clear all state and cached model.
    public func reset() {
        isLoading = false
        progress = 0
        error = nil
        lastEmbedding = nil
        lastBatchEmbeddings = []
        searchResults = []
        model = nil
    }
}

// MARK: - Similarity View Model

/// A view model for computing and displaying similarity between texts.
@MainActor
@Observable
public final class SimilarityViewModel {
    /// The computed similarity score (-1 to 1).
    public private(set) var similarity: Float?

    /// Whether computation is in progress.
    public private(set) var isLoading = false

    /// Any error that occurred.
    public private(set) var error: Error?

    private let manager = ModelManager()

    public init() {}

    /// Compute similarity between two texts.
    ///
    /// - Parameters:
    ///   - text1: First text.
    ///   - text2: Second text.
    /// - Returns: Similarity score, or nil on error.
    @discardableResult
    public func computeSimilarity(between text1: String, and text2: String) async -> Float? {
        isLoading = true
        error = nil
        similarity = nil

        do {
            let model = try await manager.loadMockModel()
            let emb1 = try await model.embed(text1)
            let emb2 = try await model.embed(text2)

            let sim = emb1.similarity(to: emb2)
            similarity = sim
            isLoading = false
            return sim

        } catch {
            self.error = error
            isLoading = false
            return nil
        }
    }
}

// MARK: - Clustering View Model

/// A view model for clustering documents.
@MainActor
@Observable
public final class ClusteringViewModel {
    /// Cluster result with document indices.
    public struct Cluster: Identifiable {
        public let id = UUID()
        public let index: Int
        public let documentIndices: [Int]
    }

    /// The computed clusters.
    public private(set) var clusters: [Cluster] = []

    /// Whether clustering is in progress.
    public private(set) var isLoading = false

    /// Any error that occurred.
    public private(set) var error: Error?

    private let manager = ModelManager()

    public init() {}

    /// Cluster documents into groups.
    ///
    /// - Parameters:
    ///   - documents: Documents to cluster.
    ///   - k: Number of clusters.
    /// - Returns: Array of clusters.
    @discardableResult
    public func cluster(_ documents: [String], into k: Int) async -> [Cluster] {
        isLoading = true
        error = nil
        clusters = []

        do {
            let result = try await manager.clusterDocuments(documents, numberOfClusters: k)

            clusters = result.enumerated().map { index, indices in
                Cluster(index: index, documentIndices: indices)
            }
            isLoading = false
            return clusters

        } catch {
            self.error = error
            isLoading = false
            return []
        }
    }
}

// MARK: - Preview Helpers

#if DEBUG
public extension EmbeddingViewModel {
    /// Create a preview instance with mock data.
    static var preview: EmbeddingViewModel {
        let vm = EmbeddingViewModel()
        vm.lastEmbedding = Embedding(
            vector: [0.1, 0.2, 0.3, 0.4],
            metadata: EmbeddingMetadata(
                modelID: ModelID(provider: "preview", name: "mock", version: "1"),
                tokenCount: 5,
                processingTime: 0.01,
                normalized: true
            )
        )
        return vm
    }
}
#endif

#endif // canImport(SwiftUI)
