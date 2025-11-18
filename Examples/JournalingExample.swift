/*
 Example: Using EmbedKit in a Journaling App

 This example demonstrates how to integrate EmbedKit for semantic search
 in a personal journaling application.
 */

import Foundation
import EmbedKit
import VectorIndex

// MARK: - Model Setup

struct JournalEmbeddingService {
    private let pipeline: EmbeddingPipeline
    private let storage: VectorIndexAdapter

    /// Initialize with a CoreML model
    init() async throws {
        // 1. Load the model (you need to convert and add MiniLM-L6-v2.mlpackage to your bundle)
        let modelURL = Bundle.main.url(
            forResource: "MiniLM-L6-v2",
            withExtension: "mlpackage"
        )!

        // 2. Create the pipeline
        let pipeline = try await EmbeddingPipeline(
            modelURL: modelURL,
            tokenizer: BERTTokenizer(),
            configuration: EmbeddingPipelineConfiguration(
                poolingStrategy: .mean,        // Best for sentence embeddings
                normalize: true,                // Required for cosine similarity
                useGPUAcceleration: true,      // Use Metal when available
                cacheConfiguration: .init(
                    maxEntries: 1000           // Cache recent embeddings
                ),
                batchSize: 32                  // Process in batches
            )
        )
        self.pipeline = pipeline

        // 3. Set up storage with VectorIndex
        self.storage = VectorIndexAdapter.withVectorIndex(
            pipeline: pipeline,
            dimensions: 384,  // MiniLM-L6-v2 dimensions
            distanceMetric: .cosine
        )
    }
}

// MARK: - Data Models

struct JournalEntry {
    let id: UUID
    let date: Date
    let content: String
    let title: String?
    let mood: String?
    let tags: [String]
    let location: String?
}

// MARK: - Indexing Journal Entries

extension JournalEmbeddingService {

    /// Index a single journal entry
    func indexEntry(_ entry: JournalEntry) async throws {
        // Combine relevant text for embedding
        var textToEmbed = entry.content
        if let title = entry.title {
            textToEmbed = "\(title)\n\n\(textToEmbed)"
        }

        // Create metadata
        var metadata: [String: Any] = [
            "id": entry.id.uuidString,
            "date": entry.date.timeIntervalSince1970,
            "content": entry.content
        ]

        if let title = entry.title {
            metadata["title"] = title
        }
        if let mood = entry.mood {
            metadata["mood"] = mood
        }
        if !entry.tags.isEmpty {
            metadata["tags"] = entry.tags.joined(separator: ",")
        }
        if let location = entry.location {
            metadata["location"] = location
        }

        // Store with embedding
        _ = try await storage.addText(
            textToEmbed,
            metadata: VectorMetadata(
                text: textToEmbed,
                additionalData: metadata
            )
        )
    }

    /// Index multiple entries efficiently
    func indexEntries(_ entries: [JournalEntry]) async throws {
        // Prepare texts for batch processing
        let texts = entries.map { entry in
            var text = entry.content
            if let title = entry.title {
                text = "\(title)\n\n\(text)"
            }
            return text
        }

        // Prepare metadata
        let metadata = entries.map { entry in
            VectorMetadata(
                text: entry.content,
                additionalData: [
                    "id": entry.id.uuidString,
                    "date": entry.date.timeIntervalSince1970,
                    "mood": entry.mood ?? "",
                    "tags": entry.tags.joined(separator: ",")
                ]
            )
        }

        // Batch index
        _ = try await storage.addTexts(texts, metadata: metadata)
    }
}

// MARK: - Semantic Search

extension JournalEmbeddingService {

    /// Search for similar journal entries
    func searchEntries(
        query: String,
        limit: Int = 10,
        filters: SearchFilters? = nil
    ) async throws -> [SearchResult] {
        // Perform semantic search
        let results = try await storage.searchByText(
            query,
            k: limit * 2,  // Get more for filtering
            threshold: 0.3  // Minimum similarity score
        )

        // Apply filters and convert to domain objects
        var searchResults: [SearchResult] = []

        for result in results {
            // Apply date filter
            if let dateFilter = filters?.dateRange {
                let timestamp = result.metadata["date"] as? TimeInterval ?? 0
                let date = Date(timeIntervalSince1970: timestamp)
                if date < dateFilter.start || date > dateFilter.end {
                    continue
                }
            }

            // Apply mood filter
            if let moodFilter = filters?.mood {
                let mood = result.metadata["mood"] as? String ?? ""
                if mood != moodFilter {
                    continue
                }
            }

            // Apply tag filter
            if let tagFilter = filters?.tags, !tagFilter.isEmpty {
                let tags = (result.metadata["tags"] as? String ?? "")
                    .split(separator: ",")
                    .map { String($0) }
                let hasMatchingTag = tagFilter.contains { tag in
                    tags.contains(tag)
                }
                if !hasMatchingTag {
                    continue
                }
            }

            // Create search result
            let searchResult = SearchResult(
                id: UUID(uuidString: result.metadata["id"] as? String ?? "") ?? UUID(),
                content: result.metadata["content"] as? String ?? "",
                title: result.metadata["title"] as? String,
                date: Date(timeIntervalSince1970: result.metadata["date"] as? TimeInterval ?? 0),
                score: result.score,
                mood: result.metadata["mood"] as? String
            )

            searchResults.append(searchResult)

            if searchResults.count >= limit {
                break
            }
        }

        return searchResults
    }

    /// Find entries similar to a given entry
    func findSimilarEntries(
        to entry: JournalEntry,
        limit: Int = 5
    ) async throws -> [SearchResult] {
        var query = entry.content
        if let title = entry.title {
            query = "\(title) \(query)"
        }

        let results = try await searchEntries(query: query, limit: limit + 1)

        // Remove the entry itself from results
        return results.filter { $0.id != entry.id }.prefix(limit).map { $0 }
    }
}

// MARK: - Supporting Types

struct SearchFilters {
    let dateRange: (start: Date, end: Date)?
    let mood: String?
    let tags: [String]?
}

struct SearchResult {
    let id: UUID
    let content: String
    let title: String?
    let date: Date
    let score: Float
    let mood: String?
}

// MARK: - Smart Features

extension JournalEmbeddingService {

    /// Suggest related tags based on entry content
    func suggestTags(for text: String) async throws -> [String] {
        // Search for similar entries
        let similar = try await storage.searchByText(text, k: 20, threshold: 0.6)

        // Collect tags from similar entries
        var tagCounts: [String: Int] = [:]

        for result in similar {
            if let tagsString = result.metadata["tags"] as? String {
                let tags = tagsString.split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) }
                for tag in tags where !tag.isEmpty {
                    tagCounts[tag, default: 0] += 1
                }
            }
        }

        // Return top tags
        return tagCounts
            .sorted { $0.value > $1.value }
            .prefix(5)
            .map { $0.key }
    }

    /// Detect mood from entry content
    func detectMood(from text: String) async throws -> String? {
        // Define mood exemplars
        let moodExemplars = [
            "happy": "I feel great, wonderful, amazing, joyful, excited, delighted",
            "sad": "I feel down, depressed, unhappy, miserable, sorrowful",
            "anxious": "I feel worried, nervous, stressed, anxious, tense",
            "grateful": "I feel thankful, grateful, appreciative, blessed",
            "peaceful": "I feel calm, relaxed, peaceful, serene, tranquil"
        ]

        // Generate embeddings for text and exemplars
        let textEmbedding = try await pipeline.embed(text)

        var bestMood: String?
        var bestScore: Float = 0

        for (mood, exemplar) in moodExemplars {
            let exemplarEmbedding = try await pipeline.embed(exemplar)
            let similarity = try textEmbedding.cosineSimilarity(to: exemplarEmbedding)

            if similarity > bestScore && similarity > 0.5 {
                bestScore = similarity
                bestMood = mood
            }
        }

        return bestMood
    }

    /// Generate a summary of themes over a time period
    func generateThemes(
        from startDate: Date,
        to endDate: Date,
        topK: Int = 5
    ) async throws -> [Theme] {
        // This would require clustering embeddings
        // Simplified version: find representative entries

        let allEntries = try await searchEntries(
            query: "",  // Get all entries
            limit: 1000,
            filters: SearchFilters(
                dateRange: (start: startDate, end: endDate),
                mood: nil,
                tags: nil
            )
        )

        // Group by similarity (simplified - in production use clustering)
        // For now, return top entries as themes
        return allEntries.prefix(topK).map { entry in
            Theme(
                title: entry.title ?? "Untitled Theme",
                summary: String(entry.content.prefix(100)),
                entryCount: 1,
                dateRange: (entry.date, entry.date)
            )
        }
    }
}

struct Theme {
    let title: String
    let summary: String
    let entryCount: Int
    let dateRange: (start: Date, end: Date)
}

// MARK: - Usage Example

@main
struct JournalingApp {
    static func main() async throws {
        // Initialize service
        let service = try await JournalEmbeddingService()

        // Index a new entry
        let entry = JournalEntry(
            id: UUID(),
            date: Date(),
            content: "Today I went for a long walk in the park. The autumn leaves were beautiful, and I felt a deep sense of peace and gratitude for nature.",
            title: "Autumn Walk",
            mood: "peaceful",
            tags: ["nature", "gratitude", "walking"],
            location: "Central Park"
        )

        try await service.indexEntry(entry)

        // Search for similar entries
        let results = try await service.searchEntries(
            query: "feeling grateful for nature",
            limit: 5
        )

        print("Found \(results.count) similar entries:")
        for result in results {
            print("  - \(result.title ?? "Untitled") (score: \(result.score))")
        }

        // Suggest tags for new content
        let newText = "Spent the morning meditating and doing yoga. Feeling very centered and calm."
        let suggestedTags = try await service.suggestTags(for: newText)
        print("\nSuggested tags: \(suggestedTags.joined(separator: ", "))")

        // Detect mood
        if let mood = try await service.detectMood(from: newText) {
            print("Detected mood: \(mood)")
        }
    }
}