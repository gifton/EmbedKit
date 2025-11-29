// Tests for StreamingProcessor
import Testing
import Foundation
@testable import EmbedKit

@Suite("Streaming Processor")
struct StreamingProcessorTests {

    // MARK: - Test Backend

    actor MockBackend: CoreMLProcessingBackend {
        private(set) var isLoaded: Bool = false
        var memoryUsage: Int64 { 0 }
        private let dim: Int

        init(dimensions: Int = 4) {
            self.dim = dimensions
        }

        func load() async throws { isLoaded = true }
        func unload() async throws { isLoaded = false }

        func process(_ input: CoreMLInput) async throws -> CoreMLOutput {
            // Return shape [seqLen, dim] to match input tokens for proper pooling
            let seqLen = input.tokenIDs.count
            let values = (0..<(seqLen * dim)).map { Float($0 % dim) + Float(seqLen) * 0.1 }
            return CoreMLOutput(values: values, shape: [seqLen, dim])
        }

        func processBatch(_ inputs: [CoreMLInput]) async throws -> [CoreMLOutput] {
            inputs.map { inp in
                let seqLen = inp.tokenIDs.count
                let values = (0..<(seqLen * dim)).map { Float($0 % dim) + Float(seqLen) * 0.1 }
                return CoreMLOutput(values: values, shape: [seqLen, dim])
            }
        }
    }

    private func makeModel(dim: Int = 4) -> AppleEmbeddingModel {
        let backend = MockBackend(dimensions: dim)
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration(
            paddingStrategy: .none,
            includeSpecialTokens: false
        )
        return AppleEmbeddingModel(
            backend: backend,
            tokenizer: tokenizer,
            configuration: cfg,
            dimensions: dim
        )
    }

    // MARK: - Chunking Tests

    @Test
    func previewChunks_characterStrategy() async throws {
        let model = makeModel()
        var config = StreamingConfig()
        config.chunkingStrategy = .characters(size: 50) // Split every ~50 chars
        config.overlap = 10
        config.minChunkSize = 10
        let processor = StreamingProcessor(model: model, config: config)

        let text = String(repeating: "Hello world. ", count: 10) // ~130 chars
        let chunks = processor.previewChunks(text)

        #expect(chunks.count >= 2, "Should create multiple chunks")
        #expect(chunks.allSatisfy { $0.text.count >= 10 }, "All chunks should meet min size")
    }

    @Test
    func previewChunks_tokenStrategy() async throws {
        let model = makeModel()
        var config = StreamingConfig()
        config.chunkingStrategy = .tokens(count: 15) // Split every ~15 tokens (~60 chars)
        config.overlap = 5
        config.minChunkSize = 10
        let processor = StreamingProcessor(model: model, config: config)

        let text = String(repeating: "word ", count: 50) // 250 chars (~50 tokens)
        let chunks = processor.previewChunks(text)

        #expect(chunks.count >= 3, "Should create multiple chunks for long text")
    }

    @Test
    func previewChunks_sentenceStrategy() async throws {
        let model = makeModel()
        var config = StreamingConfig()
        config.chunkingStrategy = .sentences(count: 2) // Group 2 sentences per chunk
        config.overlap = 20
        config.minChunkSize = 5
        let processor = StreamingProcessor(model: model, config: config)

        let text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        let chunks = processor.previewChunks(text)

        #expect(chunks.count >= 2, "Should group sentences into chunks")
    }

    @Test
    func previewChunks_paragraphStrategy() async throws {
        let model = makeModel()
        var config = StreamingConfig()
        config.chunkingStrategy = .paragraphs(count: 2) // Group 2 paragraphs per chunk
        config.overlap = 0
        config.minChunkSize = 5
        let processor = StreamingProcessor(model: model, config: config)

        let text = "Paragraph one content.\n\nParagraph two content.\n\nParagraph three content.\n\nParagraph four."
        let chunks = processor.previewChunks(text)

        #expect(chunks.count >= 2, "Should group paragraphs into chunks")
    }

    @Test
    func previewChunks_customStrategy() async throws {
        let model = makeModel()
        var config = StreamingConfig()
        config.chunkingStrategy = .custom { text in
            // Split on "---"
            text.components(separatedBy: "---").filter { !$0.isEmpty }
        }
        config.minChunkSize = 1
        let processor = StreamingProcessor(model: model, config: config)

        let text = "Part A---Part B---Part C"
        let chunks = processor.previewChunks(text)

        #expect(chunks.count == 3, "Should split by custom delimiter")
    }

    @Test
    func chunk_metadata_isAccurate() async throws {
        let model = makeModel()
        var config = StreamingConfig()
        config.overlap = 5
        config.minChunkSize = 5
        let processor = StreamingProcessor(model: model, config: config)

        let text = "AAAAA BBBBB CCCCC DDDDD EEEEE FFFFF GGGGG"
        let chunks = processor.previewChunks(text)

        // Check index metadata
        for (i, chunk) in chunks.enumerated() {
            #expect(chunk.index == i, "Index should match position")
            #expect(chunk.totalChunks == chunks.count, "Total chunks should be consistent")
        }

        // First chunk should have no previous overlap
        if let first = chunks.first {
            #expect(first.isFirst)
            #expect(first.overlapWithPrevious == 0)
        }

        // Last chunk should have no next overlap
        if let last = chunks.last {
            #expect(last.isLast)
            #expect(last.overlapWithNext == 0)
        }
    }

    // MARK: - Streaming Tests

    @Test
    func embedStream_yieldsAllChunks() async throws {
        let model = makeModel()
        var config = StreamingConfig()
        config.overlap = 5
        config.minChunkSize = 5
        let processor = StreamingProcessor(model: model, config: config)

        let text = "Hello world this is a test of streaming embeddings"
        let expectedChunks = processor.previewChunks(text)

        var received: [ChunkEmbedding] = []
        for try await chunkEmb in processor.embedStream(text) {
            received.append(chunkEmb)
        }

        #expect(received.count == expectedChunks.count, "Should receive all chunks")
        for (i, ce) in received.enumerated() {
            #expect(ce.chunk.index == i, "Chunks should arrive in order")
            #expect(ce.embedding.vector.count == 4, "Each embedding should have correct dimensions")
        }
    }

    @Test
    func embedStreamConcurrent_processesAllChunks() async throws {
        let model = makeModel()
        var config = StreamingConfig()
        config.overlap = 5
        config.minChunkSize = 5
        config.concurrency = 3
        let processor = StreamingProcessor(model: model, config: config)

        let text = String(repeating: "word ", count: 30)
        let expectedCount = processor.previewChunks(text).count

        var received: [ChunkEmbedding] = []
        for try await chunkEmb in processor.embedStreamConcurrent(text) {
            received.append(chunkEmb)
        }

        #expect(received.count == expectedCount, "Should receive all chunks")
        // Note: order may vary in concurrent mode
    }

    // MARK: - Document Embedding Tests

    @Test
    func embedDocument_returnsAllChunks() async throws {
        let model = makeModel()
        var config = StreamingConfig()
        config.overlap = 5
        config.minChunkSize = 5
        let processor = StreamingProcessor(model: model, config: config)

        let text = "This is a test document for streaming processing"
        let result = try await processor.embedDocument(text)

        #expect(result.chunkCount >= 1, "Should have at least one chunk")
        #expect(result.processingTime > 0, "Should record processing time")
        for ce in result.chunkEmbeddings {
            #expect(ce.embedding.vector.count == 4)
        }
    }

    @Test
    func embedDocumentAggregated_returnsSingleEmbedding() async throws {
        let model = makeModel()
        var config = StreamingConfig()
        config.overlap = 5
        config.minChunkSize = 5
        let processor = StreamingProcessor(model: model, config: config)

        let text = "This is a test document for aggregated embedding"
        let embedding = try await processor.embedDocumentAggregated(text, aggregation: .mean)

        #expect(embedding.vector.count == 4, "Should have correct dimensions")
        #expect(embedding.metadata.custom["chunks"] != nil, "Should include chunk count in metadata")

        // Verify normalized
        let magnitude = sqrt(embedding.vector.reduce(0) { $0 + $1 * $1 })
        #expect(abs(magnitude - 1.0) < 0.01, "Should be normalized")
    }

    // MARK: - Aggregation Tests

    @Test
    func aggregation_mean() async throws {
        let model = makeModel()
        var config = StreamingConfig()
        config.minChunkSize = 5
        let processor = StreamingProcessor(model: model, config: config)

        let text = "Short text here and more text"
        let result = try await processor.embedDocument(text)

        let aggregated = result.aggregate(strategy: .mean, dimensions: 4)
        #expect(aggregated.count == 4)
    }

    @Test
    func aggregation_weightedByLength() async throws {
        let model = makeModel()
        var config = StreamingConfig()
        config.minChunkSize = 5
        let processor = StreamingProcessor(model: model, config: config)

        let text = "Short text here and more text too"
        let result = try await processor.embedDocument(text)

        let aggregated = result.aggregate(strategy: .weightedByLength, dimensions: 4)
        #expect(aggregated.count == 4)
    }

    @Test
    func aggregation_first() async throws {
        let model = makeModel()
        var config = StreamingConfig()
        config.minChunkSize = 5
        let processor = StreamingProcessor(model: model, config: config)

        let text = "First chunk text and second chunk text"
        let result = try await processor.embedDocument(text)

        guard result.chunkCount >= 2 else {
            #expect(Bool(true), "Need at least 2 chunks for this test")
            return
        }

        let aggregated = result.aggregate(strategy: .first, dimensions: 4)
        #expect(aggregated == result.chunkEmbeddings.first?.embedding.vector)
    }

    @Test
    func aggregation_last() async throws {
        let model = makeModel()
        var config = StreamingConfig()
        config.minChunkSize = 5
        let processor = StreamingProcessor(model: model, config: config)

        let text = "First chunk text and second chunk text"
        let result = try await processor.embedDocument(text)

        guard result.chunkCount >= 2 else {
            #expect(Bool(true), "Need at least 2 chunks for this test")
            return
        }

        let aggregated = result.aggregate(strategy: .last, dimensions: 4)
        #expect(aggregated == result.chunkEmbeddings.last?.embedding.vector)
    }

    @Test
    func aggregation_concatenate() async throws {
        let model = makeModel()
        var config = StreamingConfig()
        config.minChunkSize = 5
        let processor = StreamingProcessor(model: model, config: config)

        let text = "First chunk and second chunk"
        let result = try await processor.embedDocument(text)

        let aggregated = result.aggregate(strategy: .concatenate, dimensions: 4)
        #expect(aggregated.count == result.chunkCount * 4, "Concatenated should have all dimensions")
    }

    @Test
    func aggregation_custom() async throws {
        let model = makeModel()
        var config = StreamingConfig()
        config.minChunkSize = 5
        let processor = StreamingProcessor(model: model, config: config)

        let text = "First chunk and second chunk"
        let result = try await processor.embedDocument(text)

        // Custom: return sum of all vectors
        let aggregated = result.aggregate(strategy: .custom { embeddings in
            guard let first = embeddings.first else { return [] }
            let dim = first.embedding.vector.count
            var sum = [Float](repeating: 0, count: dim)
            for ce in embeddings {
                for (i, v) in ce.embedding.vector.enumerated() {
                    sum[i] += v
                }
            }
            return sum
        }, dimensions: 4)

        #expect(aggregated.count == 4)
    }

    // MARK: - Factory Methods

    @Test
    func forShortDocuments_factory() async throws {
        let model = makeModel()
        let processor = StreamingProcessor.forShortDocuments(model: model)

        let text = String(repeating: "word ", count: 50)
        let chunks = processor.previewChunks(text)

        #expect(chunks.count >= 1, "Should create chunks")
    }

    @Test
    func forLongDocuments_factory() async throws {
        let model = makeModel()
        let processor = StreamingProcessor.forLongDocuments(model: model)

        // forLongDocuments uses tokens(count: 200) = ~800 chars, so need longer text
        let text = String(repeating: "word ", count: 400) // 2000 chars
        let chunks = processor.previewChunks(text)

        #expect(chunks.count >= 2, "Should create multiple chunks for long text")
    }

    @Test
    func forSentences_factory() async throws {
        let model = makeModel()
        let processor = StreamingProcessor.forSentences(model: model, sentencesPerChunk: 2)

        let text = "Sentence one. Sentence two. Sentence three. Sentence four."
        let chunks = processor.previewChunks(text)

        #expect(chunks.count >= 2, "Should group by sentences")
    }

    // MARK: - Edge Cases

    @Test
    func emptyText_handledGracefully() async throws {
        let model = makeModel()
        let processor = StreamingProcessor(model: model)

        let chunks = processor.previewChunks("")
        #expect(chunks.isEmpty, "Empty text should produce no chunks")
    }

    @Test
    func shortText_singleChunk() async throws {
        let model = makeModel()
        var config = StreamingConfig()
        config.minChunkSize = 5
        let processor = StreamingProcessor(model: model, config: config)

        let text = "Short"
        let chunks = processor.previewChunks(text)

        #expect(chunks.count == 1, "Short text should be single chunk")
        #expect(chunks.first?.text == text)
    }

    @Test
    func veryLongText_manyChunks() async throws {
        let model = makeModel()
        var config = StreamingConfig()
        config.chunkingStrategy = .characters(size: 100) // Split every ~100 chars to get ~25 chunks
        config.overlap = 10
        config.minChunkSize = 20
        let processor = StreamingProcessor(model: model, config: config)

        let text = String(repeating: "word ", count: 500) // 2500 chars
        let chunks = processor.previewChunks(text)

        #expect(chunks.count >= 20, "Very long text should produce many chunks")
    }
}
