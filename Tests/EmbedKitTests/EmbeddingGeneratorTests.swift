// EmbedKit - EmbeddingGenerator Tests

import Testing
import Foundation
@testable import EmbedKit
import VectorCore

@Suite("EmbeddingGenerator")
struct EmbeddingGeneratorTests {

    // MARK: - VectorProducer Conformance Tests

    @Test("Conforms to VectorProducer protocol")
    func testVectorProducerConformance() async throws {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)

        // VectorProducer requirements
        #expect(generator.dimensions == mockModel.dimensions)
        #expect(generator.producesNormalizedVectors == true)  // default config normalizes
    }

    @Test("produce() returns correct dimensions")
    func testProduceDimensions() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 384)
        let generator = EmbeddingGenerator(model: mockModel)

        let vectors = try await generator.produce(["Hello", "World"])

        #expect(vectors.count == 2)
        #expect(vectors[0].count == 384)
        #expect(vectors[1].count == 384)
    }

    @Test("produce() handles empty input")
    func testProduceEmpty() async throws {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)

        let vectors = try await generator.produce([])

        #expect(vectors.isEmpty)
    }

    @Test("produce() single text")
    func testProduceSingle() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: mockModel)

        let vector = try await generator.produce("Hello, World!")

        #expect(vector.count == 128)
    }

    // MARK: - Progress Streaming Tests

    @Test("generateWithProgress yields all embeddings")
    func testGenerateWithProgressCount() async throws {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)

        var count = 0
        let stream = await generator.generateWithProgress(["a", "b", "c"])
        for try await (_, _) in stream {
            count += 1
        }

        #expect(count == 3)
    }

    @Test("generateWithProgress updates progress correctly")
    func testGenerateWithProgressValues() async throws {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)

        var progressValues: [Int] = []
        let stream = await generator.generateWithProgress(["a", "b", "c", "d"])
        for try await (_, progress) in stream {
            progressValues.append(progress.current)
        }

        #expect(progressValues == [1, 2, 3, 4])
    }

    @Test("generateWithProgress reaches 100%")
    func testGenerateWithProgressComplete() async throws {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)

        var finalProgress: OperationProgress?
        let stream = await generator.generateWithProgress(["a", "b"])
        for try await (_, progress) in stream {
            finalProgress = progress
        }

        #expect(finalProgress?.isComplete == true)
        #expect(finalProgress?.percentage == 100)
    }

    // MARK: - BatchProgress Streaming Tests

    @Test("generateEmbeddingsWithProgress yields Embedding objects")
    func testGenerateEmbeddingsWithProgress() async throws {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)

        var embeddings: [Embedding] = []
        let stream = await generator.generateEmbeddingsWithProgress(["a", "b"])
        for try await (embedding, _) in stream {
            embeddings.append(embedding)
        }

        #expect(embeddings.count == 2)
        #expect(embeddings[0].dimensions == mockModel.dimensions)
    }

    @Test("generateEmbeddingsWithProgress provides BatchProgress")
    func testGenerateEmbeddingsWithProgressBatchInfo() async throws {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(
            model: mockModel,
            batchOptions: BatchOptions(maxBatchSize: 2)
        )

        var progressList: [BatchProgress] = []
        let stream = await generator.generateEmbeddingsWithProgress(["a", "b", "c", "d"])
        for try await (_, progress) in stream {
            progressList.append(progress)
        }

        #expect(progressList.count == 4)
        // Should have batch info
        #expect(progressList.last?.totalBatches ?? 0 > 0)
    }

    // MARK: - Batch Processing Tests

    @Test("generateBatch returns all embeddings")
    func testGenerateBatch() async throws {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)

        let vectors = try await generator.generateBatch(["a", "b", "c", "d", "e"])

        #expect(vectors.count == 5)
    }

    @Test("generateBatch calls progress callback")
    func testGenerateBatchProgress() async throws {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(
            model: mockModel,
            batchOptions: BatchOptions(maxBatchSize: 2)
        )

        // Use actor-isolated storage for progress updates
        actor ProgressCollector {
            var updates: [BatchProgress] = []
            func append(_ progress: BatchProgress) { updates.append(progress) }
        }
        let collector = ProgressCollector()

        let vectors = try await generator.generateBatch(["a", "b", "c", "d"]) { progress in
            Task { await collector.append(progress) }
        }

        // Wait a moment for async progress callbacks to complete
        try await Task.sleep(for: .milliseconds(50))
        let progressUpdates = await collector.updates

        #expect(vectors.count == 4)
        #expect(progressUpdates.count > 0)
        #expect(progressUpdates.first?.phase == "Starting")
        #expect(progressUpdates.last?.phase == "Complete")
    }

    @Test("generateBatch handles empty input")
    func testGenerateBatchEmpty() async throws {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)

        let vectors = try await generator.generateBatch([])

        #expect(vectors.isEmpty)
    }

    // MARK: - Configuration Tests

    @Test("Generator respects configuration")
    func testGeneratorConfiguration() async throws {
        let mockModel = MockEmbeddingModel()
        let config = EmbeddingConfiguration(
            normalizeOutput: false
        )
        let generator = EmbeddingGenerator(
            model: mockModel,
            configuration: config
        )

        #expect(generator.producesNormalizedVectors == false)
    }

    @Test("Generator provides hints")
    func testGeneratorHints() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 768)
        let generator = EmbeddingGenerator(
            model: mockModel,
            batchOptions: BatchOptions(maxBatchSize: 64)
        )

        let hints = await generator.hints

        #expect(hints.dimensions == 768)
        #expect(hints.optimalBatchSize == 64)
        #expect(hints.maxBatchSize == 64)
    }

    // MARK: - Model Access Tests

    @Test("Generator provides model ID")
    func testModelID() async throws {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)

        let modelID = await generator.modelID

        #expect(modelID.provider == "mock")
    }

    @Test("Generator provides metrics")
    func testMetrics() async throws {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)

        // Generate some embeddings to accumulate metrics
        _ = try await generator.produce(["test"])

        let metrics = await generator.metrics
        #expect(metrics.totalTokensProcessed >= 0)
    }

    // MARK: - Lifecycle Tests

    @Test("Warmup delegates to model")
    func testWarmup() async throws {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)

        // Should not throw
        try await generator.warmup()
    }

    @Test("Release delegates to model")
    func testRelease() async throws {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)

        // Should not throw
        try await generator.release()
    }
}

// MARK: - GeneratorConfiguration Tests

@Suite("GeneratorConfiguration")
struct GeneratorConfigurationTests {

    @Test("Default configuration")
    func testDefault() {
        let config = GeneratorConfiguration.default

        #expect(config.embedding.maxTokens == 512)
        #expect(config.batch.maxBatchSize == 32)
    }

    @Test("High throughput configuration")
    func testHighThroughput() {
        let config = GeneratorConfiguration.highThroughput

        #expect(config.batch.maxBatchSize == 64)
    }

    @Test("Low latency configuration")
    func testLowLatency() {
        let config = GeneratorConfiguration.lowLatency

        #expect(config.batch.maxBatchSize == 8)
    }

    @Test("Semantic search configuration")
    func testSemanticSearch() {
        let config = GeneratorConfiguration.forSemanticSearch(maxLength: 256)

        #expect(config.embedding.maxTokens == 256)
        #expect(config.embedding.normalizeOutput == true)
    }

    @Test("RAG configuration")
    func testRAG() {
        let config = GeneratorConfiguration.forRAG(chunkSize: 128)

        #expect(config.embedding.maxTokens == 128)
    }
}

// MARK: - ID-Preserving Batch Tests

@Suite("EmbeddingGenerator ID-Preserving")
struct EmbeddingGeneratorIDPreservingTests {

    // MARK: - produceWithIDs Tests

    @Test("produceWithIDs preserves Int IDs")
    func testProduceWithIntIDs() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: mockModel)

        let items: [(id: Int, text: String)] = [
            (id: 100, text: "Hello"),
            (id: 200, text: "World"),
            (id: 300, text: "Test")
        ]

        let results = try await generator.produceWithIDs(items)

        #expect(results.count == 3)
        #expect(results[0].id == 100)
        #expect(results[1].id == 200)
        #expect(results[2].id == 300)
    }

    @Test("produceWithIDs preserves UUID IDs (GournalCore use case)")
    func testProduceWithUUIDs() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 384)
        let generator = EmbeddingGenerator(model: mockModel)

        let uuid1 = UUID()
        let uuid2 = UUID()
        let uuid3 = UUID()

        let items: [(id: UUID, text: String)] = [
            (id: uuid1, text: "Fragment one content"),
            (id: uuid2, text: "Fragment two content"),
            (id: uuid3, text: "Fragment three content")
        ]

        let results = try await generator.produceWithIDs(items)

        #expect(results.count == 3)
        #expect(results[0].id == uuid1)
        #expect(results[1].id == uuid2)
        #expect(results[2].id == uuid3)
        #expect(results[0].vector.count == 384)
    }

    @Test("produceWithIDs preserves String IDs")
    func testProduceWithStringIDs() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: mockModel)

        let items: [(id: String, text: String)] = [
            (id: "doc-001", text: "First document"),
            (id: "doc-002", text: "Second document")
        ]

        let results = try await generator.produceWithIDs(items)

        #expect(results.count == 2)
        #expect(results[0].id == "doc-001")
        #expect(results[1].id == "doc-002")
    }

    @Test("produceWithIDs handles empty input")
    func testProduceWithIDsEmpty() async throws {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)

        let items: [(id: Int, text: String)] = []
        let results = try await generator.produceWithIDs(items)

        #expect(results.isEmpty)
    }

    @Test("produceWithIDs maintains order across batch boundaries")
    func testProduceWithIDsOrderAcrossBatches() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 64)
        // Small batch size to force multiple batches
        let generator = EmbeddingGenerator(
            model: mockModel,
            batchOptions: BatchOptions(maxBatchSize: 3)
        )

        // 10 items will span 4 batches (3+3+3+1)
        let items: [(id: Int, text: String)] = (0..<10).map { i in
            (id: i * 10, text: "Text \(i)")
        }

        let results = try await generator.produceWithIDs(items)

        #expect(results.count == 10)
        for i in 0..<10 {
            #expect(results[i].id == i * 10, "ID mismatch at index \(i)")
        }
    }

    @Test("produceWithIDs allows duplicate IDs")
    func testProduceWithDuplicateIDs() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: mockModel)

        let items: [(id: Int, text: String)] = [
            (id: 1, text: "First occurrence"),
            (id: 2, text: "Unique"),
            (id: 1, text: "Second occurrence of ID 1")
        ]

        let results = try await generator.produceWithIDs(items)

        #expect(results.count == 3)
        #expect(results[0].id == 1)
        #expect(results[1].id == 2)
        #expect(results[2].id == 1)  // Duplicate ID preserved in order
    }

    // MARK: - produceWithIDsAndProgress Tests

    @Test("produceWithIDsAndProgress streams all items with IDs")
    func testProduceWithIDsAndProgressCount() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: mockModel)

        let uuid1 = UUID()
        let uuid2 = UUID()
        let uuid3 = UUID()

        let items: [(id: UUID, text: String)] = [
            (id: uuid1, text: "Text A"),
            (id: uuid2, text: "Text B"),
            (id: uuid3, text: "Text C")
        ]

        var results: [(id: UUID, vector: [Float])] = []
        var progressValues: [Int] = []

        for try await (id, vector, progress) in await generator.produceWithIDsAndProgress(items) {
            results.append((id: id, vector: vector))
            progressValues.append(progress.current)
        }

        #expect(results.count == 3)
        #expect(results[0].id == uuid1)
        #expect(results[1].id == uuid2)
        #expect(results[2].id == uuid3)
        #expect(progressValues == [1, 2, 3])
    }

    @Test("produceWithIDsAndProgress provides accurate progress")
    func testProduceWithIDsAndProgressAccuracy() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: mockModel)

        let items: [(id: Int, text: String)] = (0..<5).map { (id: $0, text: "Text \($0)") }

        var lastProgress: OperationProgress?
        for try await (_, _, progress) in await generator.produceWithIDsAndProgress(items) {
            lastProgress = progress
        }

        #expect(lastProgress?.isComplete == true)
        #expect(lastProgress?.percentage == 100)
        #expect(lastProgress?.current == 5)
        #expect(lastProgress?.total == 5)
    }

    // MARK: - generateBatchWithIDs Tests

    @Test("generateBatchWithIDs returns all results with IDs")
    func testGenerateBatchWithIDs() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 256)
        let generator = EmbeddingGenerator(model: mockModel)

        let items: [(id: String, text: String)] = [
            (id: "frag-a", text: "Fragment A content"),
            (id: "frag-b", text: "Fragment B content"),
            (id: "frag-c", text: "Fragment C content"),
            (id: "frag-d", text: "Fragment D content")
        ]

        let results = try await generator.generateBatchWithIDs(items)

        #expect(results.count == 4)
        #expect(results[0].id == "frag-a")
        #expect(results[1].id == "frag-b")
        #expect(results[2].id == "frag-c")
        #expect(results[3].id == "frag-d")
        #expect(results[0].vector.count == 256)
    }

    @Test("generateBatchWithIDs calls progress callback")
    func testGenerateBatchWithIDsProgress() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(
            model: mockModel,
            batchOptions: BatchOptions(maxBatchSize: 2)
        )

        let items: [(id: Int, text: String)] = (0..<6).map { (id: $0, text: "Text \($0)") }

        actor ProgressCollector {
            var updates: [BatchProgress] = []
            func append(_ p: BatchProgress) { updates.append(p) }
        }
        let collector = ProgressCollector()

        let results = try await generator.generateBatchWithIDs(items) { progress in
            Task { await collector.append(progress) }
        }

        // Wait for async progress callbacks
        try await Task.sleep(for: .milliseconds(50))
        let progressUpdates = await collector.updates

        #expect(results.count == 6)
        #expect(progressUpdates.count > 0)
        #expect(progressUpdates.first?.phase == "Starting")
        #expect(progressUpdates.last?.phase == "Complete")
    }

    @Test("generateBatchWithIDs handles empty input")
    func testGenerateBatchWithIDsEmpty() async throws {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)

        let items: [(id: Int, text: String)] = []
        let results = try await generator.generateBatchWithIDs(items)

        #expect(results.isEmpty)
    }

    @Test("generateBatchWithIDs with large batch maintains order")
    func testGenerateBatchWithIDsLargeBatch() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(
            model: mockModel,
            batchOptions: BatchOptions(maxBatchSize: 10)
        )

        // 100 items across 10 batches
        let items: [(id: Int, text: String)] = (0..<100).map { (id: $0, text: "Document \($0)") }

        let results = try await generator.generateBatchWithIDs(items)

        #expect(results.count == 100)
        for i in 0..<100 {
            #expect(results[i].id == i, "ID mismatch at index \(i): expected \(i), got \(results[i].id)")
        }
    }

    // MARK: - Custom ID Type Test

    @Test("produceWithIDs works with custom Hashable ID type")
    func testProduceWithCustomIDType() async throws {
        // Custom ID type that GournalCore might use
        struct FragmentID: Hashable, Sendable {
            let entryID: UUID
            let index: Int
        }

        let mockModel = MockEmbeddingModel(dimensions: 384)
        let generator = EmbeddingGenerator(model: mockModel)

        let entryID = UUID()
        let items: [(id: FragmentID, text: String)] = [
            (id: FragmentID(entryID: entryID, index: 0), text: "First fragment"),
            (id: FragmentID(entryID: entryID, index: 1), text: "Second fragment"),
            (id: FragmentID(entryID: entryID, index: 2), text: "Third fragment")
        ]

        let results = try await generator.produceWithIDs(items)

        #expect(results.count == 3)
        #expect(results[0].id.index == 0)
        #expect(results[1].id.index == 1)
        #expect(results[2].id.index == 2)
        #expect(results[0].id.entryID == entryID)
    }
}
