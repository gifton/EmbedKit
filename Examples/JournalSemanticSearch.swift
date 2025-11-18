/// Complete example of semantic search for a journaling app using EmbedKit
/// This demonstrates the full pipeline from text to searchable embeddings

import Foundation
import CoreML
import EmbedKit
import VectorCore
import VectorIndex

// MARK: - Journal Entry Model

struct JournalEntry: Identifiable, Codable {
    let id: UUID
    let date: Date
    let title: String?
    let content: String
    let mood: String?
    let tags: [String]

    var searchableText: String {
        var components = [String]()
        if let title = title { components.append(title) }
        components.append(content)
        if let mood = mood { components.append("Mood: \(mood)") }
        if !tags.isEmpty { components.append("Tags: \(tags.joined(separator: ", "))") }
        return components.joined(separator: ". ")
    }
}

// MARK: - Journal Semantic Search Service

/// Service that manages semantic search for journal entries
@MainActor
final class JournalSemanticSearch: ObservableObject {

    // Published properties for SwiftUI
    @Published private(set) var isIndexing = false
    @Published private(set) var indexedCount = 0
    @Published private(set) var searchResults: [SearchResult] = []

    struct SearchResult {
        let entry: JournalEntry
        let score: Float
        let snippet: String
    }

    // Private properties
    private var pipeline: EmbeddingPipeline?
    private var vectorStorage: VectorIndexBridge?
    private let entryMap = NSCache<NSString, JournalEntry>()

    // MARK: - Initialization

    /// Initialize the semantic search with the CoreML model
    func initialize() async throws {
        print("📱 Initializing Journal Semantic Search...")

        // 1. Load the CoreML model
        guard let modelURL = Bundle.main.url(
            forResource: "MiniLM-L12-v2",  // Or use quantized version
            withExtension: "mlpackage"
        ) else {
            // For development, try loading from the conversion directory
            let devPath = URL(fileURLWithPath: "/Users/goftin/dev/gsuite/VSK/EmbedKit/MiniLM-L12-v2.mlpackage")
            guard FileManager.default.fileExists(atPath: devPath.path) else {
                throw EmbeddingError.modelError("Model not found. Please add MiniLM-L12-v2.mlpackage to your bundle")
            }
            try await setupPipeline(modelURL: devPath)
            return
        }

        try await setupPipeline(modelURL: modelURL)
    }

    private func setupPipeline(modelURL: URL) async throws {
        // 2. Create the model backend
        let backend = CoreMLBackend()
        try await backend.loadModel(from: modelURL)

        // 3. Create tokenizer (BERT for MiniLM)
        let tokenizer = BERTTokenizer(
            vocabSize: 30522,
            maxLength: 512
        )

        // 4. Configure the pipeline
        let config = EmbeddingPipelineConfiguration(
            poolingStrategy: .mean,          // Mean pooling for sentence embeddings
            normalize: true,                  // Normalize for cosine similarity
            useGPUAcceleration: true,         // Use GPU/Neural Engine
            cacheConfiguration: EmbeddingCacheConfiguration(
                maxEntries: 1000,             // Cache recent embeddings
                ttlSeconds: 3600              // 1 hour TTL
            )
        )

        // 5. Create the pipeline
        self.pipeline = EmbeddingPipeline(
            tokenizer: tokenizer,
            modelBackend: backend,
            configuration: config
        )

        // 6. Initialize vector storage with HNSW for fast search
        self.vectorStorage = VectorIndexBridge(
            dimensions: 384,                  // MiniLM-L12 dimension
            distanceMetric: .cosine,
            indexType: .hnsw(
                maxConnections: 16,            // Good balance for mobile
                efConstruction: 200            // Higher = better quality, slower build
            )
        )

        print("✅ Semantic search initialized successfully")
    }

    // MARK: - Indexing

    /// Index a batch of journal entries for semantic search
    func indexEntries(_ entries: [JournalEntry]) async throws {
        guard let pipeline = pipeline, let storage = vectorStorage else {
            throw EmbeddingError.pipelineNotInitialized
        }

        isIndexing = true
        indexedCount = 0

        // Process in batches for efficiency
        let batchSize = 10
        for batchStart in stride(from: 0, to: entries.count, by: batchSize) {
            let batchEnd = min(batchStart + batchSize, entries.count)
            let batch = Array(entries[batchStart..<batchEnd])

            // Get searchable text for each entry
            let texts = batch.map { $0.searchableText }

            // Generate embeddings for the batch
            let embeddings = try await pipeline.embed(batch: texts)

            // Add to vector index
            for (entry, embedding) in zip(batch, embeddings) {
                let metadata: [String: Any] = [
                    "id": entry.id.uuidString,
                    "date": entry.date.timeIntervalSince1970,
                    "title": entry.title ?? "",
                    "snippet": String(entry.content.prefix(200))
                ]

                _ = try await storage.add(
                    vector: embedding.toArray(),
                    metadata: metadata
                )

                // Cache the full entry
                entryMap.setObject(entry, forKey: entry.id.uuidString as NSString)

                indexedCount += 1
            }

            print("📊 Indexed \(indexedCount)/\(entries.count) entries...")
        }

        isIndexing = false
        print("✅ Indexed \(entries.count) journal entries")
    }

    /// Add a single new entry to the index
    func addEntry(_ entry: JournalEntry) async throws {
        guard let pipeline = pipeline, let storage = vectorStorage else {
            throw EmbeddingError.pipelineNotInitialized
        }

        // Generate embedding
        let embedding = try await pipeline.embed(entry.searchableText)

        // Add to index
        let metadata: [String: Any] = [
            "id": entry.id.uuidString,
            "date": entry.date.timeIntervalSince1970,
            "title": entry.title ?? "",
            "snippet": String(entry.content.prefix(200))
        ]

        _ = try await storage.add(
            vector: embedding.toArray(),
            metadata: metadata
        )

        // Cache the entry
        entryMap.setObject(entry, forKey: entry.id.uuidString as NSString)
        indexedCount += 1
    }

    // MARK: - Searching

    /// Search for journal entries similar to the query
    func search(query: String, limit: Int = 10) async throws -> [SearchResult] {
        guard let pipeline = pipeline, let storage = vectorStorage else {
            throw EmbeddingError.pipelineNotInitialized
        }

        // Generate embedding for the query
        let queryEmbedding = try await pipeline.embed(query)

        // Search the index
        let results = try await storage.search(
            query: queryEmbedding.toArray(),
            k: limit
        )

        // Map results to SearchResult
        var searchResults: [SearchResult] = []
        for result in results {
            guard let idString = result.metadata["id"] as? String,
                  let entry = entryMap.object(forKey: idString as NSString) else {
                continue
            }

            let snippet = result.metadata["snippet"] as? String ?? ""
            searchResults.append(SearchResult(
                entry: entry,
                score: result.score,
                snippet: snippet
            ))
        }

        self.searchResults = searchResults
        return searchResults
    }

    /// Find entries similar to a given entry
    func findSimilar(to entry: JournalEntry, limit: Int = 5) async throws -> [SearchResult] {
        // Use the entry's content as the query
        return try await search(query: entry.searchableText, limit: limit + 1)
            .filter { $0.entry.id != entry.id }  // Exclude the entry itself
            .prefix(limit)
            .map { $0 }
    }

    /// Search by mood or emotional context
    func searchByMood(_ mood: String, limit: Int = 10) async throws -> [SearchResult] {
        let moodQuery = "Feeling \(mood). Mood is \(mood). Emotional state: \(mood)"
        return try await search(query: moodQuery, limit: limit)
    }

    /// Search by topic or theme
    func searchByTopic(_ topic: String, limit: Int = 10) async throws -> [SearchResult] {
        let topicQuery = "About \(topic). Related to \(topic). Discussing \(topic)"
        return try await search(query: topicQuery, limit: limit)
    }
}

// MARK: - Usage Example

func demoJournalSearch() async throws {
    // Initialize the search service
    let searchService = JournalSemanticSearch()
    try await searchService.initialize()

    // Sample journal entries
    let sampleEntries = [
        JournalEntry(
            id: UUID(),
            date: Date(),
            title: "Grateful Day",
            content: "Today I felt incredibly grateful for my family and friends. We had a wonderful dinner together and shared so many laughs.",
            mood: "happy",
            tags: ["gratitude", "family", "social"]
        ),
        JournalEntry(
            id: UUID(),
            date: Date().addingTimeInterval(-86400),
            title: "Workout Success",
            content: "Crushed my morning workout! Ran 5 miles and did strength training. Feeling strong and energized.",
            mood: "energetic",
            tags: ["fitness", "health", "achievement"]
        ),
        JournalEntry(
            id: UUID(),
            date: Date().addingTimeInterval(-172800),
            title: "Work Stress",
            content: "Feeling overwhelmed with deadlines at work. Multiple projects due this week and not enough time.",
            mood: "anxious",
            tags: ["work", "stress", "deadlines"]
        ),
        JournalEntry(
            id: UUID(),
            date: Date().addingTimeInterval(-259200),
            title: "Nature Walk",
            content: "Took a peaceful walk in the park today. The autumn leaves were beautiful and I felt so calm and centered.",
            mood: "peaceful",
            tags: ["nature", "mindfulness", "relaxation"]
        ),
        JournalEntry(
            id: UUID(),
            date: Date().addingTimeInterval(-345600),
            title: "Creative Breakthrough",
            content: "Had an amazing creative session today. New ideas flowing and feeling inspired to start my art project.",
            mood: "inspired",
            tags: ["creativity", "art", "inspiration"]
        )
    ]

    // Index the entries
    print("\n📚 Indexing journal entries...")
    try await searchService.indexEntries(sampleEntries)

    // Example 1: Search by natural query
    print("\n🔍 Search: 'feeling thankful and appreciative'")
    let gratitudeResults = try await searchService.search(
        query: "feeling thankful and appreciative",
        limit: 3
    )
    for result in gratitudeResults {
        print("   📝 \(result.entry.title ?? "Untitled") - Score: \(result.score)")
        print("      \(result.snippet)")
    }

    // Example 2: Find similar entries
    print("\n🔗 Finding entries similar to 'Workout Success'...")
    if let workoutEntry = sampleEntries.first(where: { $0.title == "Workout Success" }) {
        let similarResults = try await searchService.findSimilar(to: workoutEntry, limit: 3)
        for result in similarResults {
            print("   📝 \(result.entry.title ?? "Untitled") - Score: \(result.score)")
        }
    }

    // Example 3: Search by mood
    print("\n😊 Searching for 'happy' mood entries...")
    let moodResults = try await searchService.searchByMood("happy", limit: 3)
    for result in moodResults {
        print("   📝 \(result.entry.title ?? "Untitled") - Mood: \(result.entry.mood ?? "unknown")")
    }

    // Example 4: Search by topic
    print("\n🏷️ Searching for entries about 'stress and work'...")
    let topicResults = try await searchService.searchByTopic("stress and work", limit: 3)
    for result in topicResults {
        print("   📝 \(result.entry.title ?? "Untitled")")
        print("      Tags: \(result.entry.tags.joined(separator: ", "))")
    }
}

// MARK: - SwiftUI View Example

#if canImport(SwiftUI)
import SwiftUI

struct JournalSearchView: View {
    @StateObject private var searchService = JournalSemanticSearch()
    @State private var searchQuery = ""
    @State private var isSearching = false

    var body: some View {
        NavigationView {
            VStack {
                // Search bar
                HStack {
                    Image(systemName: "magnifyingglass")
                        .foregroundColor(.secondary)
                    TextField("Search your journal...", text: $searchQuery)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .onSubmit {
                            Task {
                                await performSearch()
                            }
                        }
                }
                .padding()

                // Results
                if isSearching {
                    ProgressView("Searching...")
                        .padding()
                } else {
                    List(searchService.searchResults, id: \.entry.id) { result in
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text(result.entry.title ?? "Untitled")
                                    .font(.headline)
                                Spacer()
                                Text(String(format: "%.0f%%", result.score * 100))
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }

                            Text(result.snippet)
                                .font(.body)
                                .lineLimit(3)

                            HStack {
                                if let mood = result.entry.mood {
                                    Label(mood, systemImage: "face.smiling")
                                        .font(.caption)
                                        .foregroundColor(.blue)
                                }
                                Spacer()
                                Text(result.entry.date, style: .date)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
            .navigationTitle("Semantic Search")
            .task {
                do {
                    try await searchService.initialize()
                    // Load and index entries here
                } catch {
                    print("Failed to initialize: \(error)")
                }
            }
        }
    }

    private func performSearch() async {
        guard !searchQuery.isEmpty else { return }
        isSearching = true
        do {
            _ = try await searchService.search(query: searchQuery, limit: 20)
        } catch {
            print("Search error: \(error)")
        }
        isSearching = false
    }
}
#endif