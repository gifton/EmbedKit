// EmbedKit - SwiftUI Support
// Observable view models and property wrappers for SwiftUI integration

#if canImport(SwiftUI)
import SwiftUI

// MARK: - Model Provider Configuration

/// Specifies which embedding model to use in SwiftUI ViewModels.
///
/// Use `.system()` (default) for production apps with real semantic embeddings,
/// or `.mock` for testing and SwiftUI previews.
public enum ModelProvider: Sendable {
    /// Use Apple's system NLContextualEmbedding model (default, recommended for production).
    /// - Parameter language: Language code (default: "en")
    case system(language: String = "en")

    /// Use a mock model for testing and SwiftUI previews.
    case mock

    /// The default provider uses the system model.
    public static let `default`: ModelProvider = .system()
}

// MARK: - Embedding View Model

/// A SwiftUI-compatible view model for embedding operations.
///
/// This class provides observable state for loading, embedding, and displaying results
/// in a SwiftUI application. It's main-actor isolated for safe UI updates.
///
/// By default, uses Apple's NLContextualEmbedding for real semantic embeddings.
/// Use `ModelProvider.mock` for testing or SwiftUI previews.
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

    // MARK: - Configuration

    private let modelProvider: ModelProvider
    private let manager = ModelManager()
    private var model: (any EmbeddingModel)?

    // MARK: - Initialization

    /// Creates a view model with the specified model provider.
    ///
    /// - Parameter modelProvider: The model to use (default: `.system()` for real embeddings)
    ///
    /// Example:
    /// ```swift
    /// // Production: use real embeddings
    /// let viewModel = EmbeddingViewModel()
    ///
    /// // Testing: use mock model
    /// let testViewModel = EmbeddingViewModel(modelProvider: .mock)
    /// ```
    public init(modelProvider: ModelProvider = .default) {
        self.modelProvider = modelProvider
    }

    // MARK: - Private Helpers

    private func ensureModel() async throws -> any EmbeddingModel {
        if let model = self.model {
            return model
        }

        let newModel: any EmbeddingModel
        switch modelProvider {
        case .system(let language):
            newModel = try await manager.loadNLContextualEmbedding(language: language)
        case .mock:
            newModel = try await manager.loadMockModel()
        }

        self.model = newModel
        return newModel
    }

    // MARK: - Single Embedding

    /// Embed a single text and update observable state.
    ///
    /// Uses the configured model provider (system model by default) to generate
    /// semantically meaningful embeddings.
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
            let model = try await ensureModel()

            progress = 0.7
            let embedding = try await model.embed(text)

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
            let model = try await ensureModel()

            progress = 0.4
            let embeddings = try await model.embedBatch(texts, options: BatchOptions())

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
    /// Uses the ModelManager's semanticSearch which now uses real embeddings.
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
///
/// By default, uses Apple's NLContextualEmbedding for real semantic similarity.
/// Use `ModelProvider.mock` for testing or SwiftUI previews.
@MainActor
@Observable
public final class SimilarityViewModel {
    /// The computed similarity score (-1 to 1).
    public private(set) var similarity: Float?

    /// Whether computation is in progress.
    public private(set) var isLoading = false

    /// Any error that occurred.
    public private(set) var error: Error?

    private let modelProvider: ModelProvider
    private let manager = ModelManager()
    private var model: (any EmbeddingModel)?

    /// Creates a view model with the specified model provider.
    ///
    /// - Parameter modelProvider: The model to use (default: `.system()` for real embeddings)
    public init(modelProvider: ModelProvider = .default) {
        self.modelProvider = modelProvider
    }

    private func ensureModel() async throws -> any EmbeddingModel {
        if let model = self.model {
            return model
        }

        let newModel: any EmbeddingModel
        switch modelProvider {
        case .system(let language):
            newModel = try await manager.loadNLContextualEmbedding(language: language)
        case .mock:
            newModel = try await manager.loadMockModel()
        }

        self.model = newModel
        return newModel
    }

    /// Compute similarity between two texts.
    ///
    /// Uses the configured model provider (system model by default) to generate
    /// semantically meaningful embeddings and compute cosine similarity.
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
            let model = try await ensureModel()
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

/// A view model for clustering documents using semantic embeddings.
///
/// Uses Apple's NLContextualEmbedding via `ModelManager.clusterDocuments()` for
/// semantically meaningful document clustering using k-means.
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

    /// Cluster documents into groups using semantic embeddings.
    ///
    /// Uses the ModelManager's clusterDocuments() which generates real semantic
    /// embeddings via NLContextualEmbedding and applies k-means clustering.
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
    /// Create a preview instance with mock model and sample data.
    ///
    /// Uses `ModelProvider.mock` for fast, deterministic previews.
    static var preview: EmbeddingViewModel {
        let vm = EmbeddingViewModel(modelProvider: .mock)
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

public extension SimilarityViewModel {
    /// Create a preview instance with mock model.
    ///
    /// Uses `ModelProvider.mock` for fast, deterministic previews.
    static var preview: SimilarityViewModel {
        SimilarityViewModel(modelProvider: .mock)
    }
}
#endif

#endif // canImport(SwiftUI)
