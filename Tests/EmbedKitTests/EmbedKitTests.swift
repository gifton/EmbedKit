import Testing
import Foundation
@testable import EmbedKit

@Test("EmbeddingVector initialization and operations")
func testEmbeddingVector() {
    let values: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5]
    let vector = EmbeddingVector(values)
    
    #expect(vector.dimensions == 5)
    #expect(vector.array == values)
    #expect(vector[0] == 0.1)
    #expect(vector[4] == 0.5)
}

@Test("EmbeddingVector cosine similarity")
func testCosineSimilarity() {
    let vector1 = EmbeddingVector([1.0, 0.0, 0.0])
    let vector2 = EmbeddingVector([1.0, 0.0, 0.0])
    let vector3 = EmbeddingVector([0.0, 1.0, 0.0])
    let vector4 = EmbeddingVector([-1.0, 0.0, 0.0])
    
    // Same vectors should have similarity 1.0
    #expect(abs(vector1.cosineSimilarity(with: vector2) - 1.0) < 0.001)
    
    // Orthogonal vectors should have similarity 0.0
    #expect(abs(vector1.cosineSimilarity(with: vector3)) < 0.001)
    
    // Opposite vectors should have similarity -1.0
    #expect(abs(vector1.cosineSimilarity(with: vector4) - (-1.0)) < 0.001)
}

@Test("TokenizedInput creation")
func testTokenizedInput() {
    let tokenIds = [101, 2054, 2003, 1996, 4633, 102]
    let attentionMask = [1, 1, 1, 1, 1, 1]
    
    let input = TokenizedInput(
        tokenIds: tokenIds,
        attentionMask: attentionMask,
        originalLength: 6
    )
    
    #expect(input.tokenIds == tokenIds)
    #expect(input.attentionMask == attentionMask)
    #expect(input.tokenTypeIds == nil)
    #expect(input.originalLength == 6)
}

@Test("SimpleTokenizer basic functionality")
func testSimpleTokenizer() async throws {
    let tokenizer = SimpleTokenizer(maxSequenceLength: 10)
    
    let text = "Hello world"
    let tokenized = try await tokenizer.tokenize(text)
    
    // Should have CLS token at start
    let specialTokens = await tokenizer.specialTokens
    #expect(tokenized.tokenIds[0] == specialTokens.cls)
    
    // Should have SEP token at position 3 (CLS + 2 words + SEP)
    #expect(tokenized.tokenIds[3] == specialTokens.sep)
    
    // Should be padded to max length
    #expect(tokenized.tokenIds.count == 10)
    #expect(tokenized.attentionMask.count == 10)
    
    // Attention mask should be 1 for real tokens, 0 for padding
    #expect(tokenized.attentionMask[0] == 1) // CLS
    #expect(tokenized.attentionMask[1] == 1) // hello
    #expect(tokenized.attentionMask[2] == 1) // world
    #expect(tokenized.attentionMask[3] == 1) // SEP
    #expect(tokenized.attentionMask[4] == 0) // PAD
}

@Test("SimpleTokenizer batch processing")
func testSimpleTokenizerBatch() async throws {
    let tokenizer = SimpleTokenizer(maxSequenceLength: 8)
    
    let texts = ["Hello", "World peace", "Testing tokenization"]
    let tokenizedBatch = try await tokenizer.tokenize(batch: texts)
    
    #expect(tokenizedBatch.count == 3)
    
    // Each should be padded to same length
    for tokenized in tokenizedBatch {
        #expect(tokenized.tokenIds.count == 8)
        #expect(tokenized.attentionMask.count == 8)
    }
}

@Test("Configuration defaults")
func testConfiguration() {
    let config = Configuration()
    
    #expect(config.model.maxSequenceLength == 512)
    #expect(config.model.normalizeEmbeddings == true)
    #expect(config.model.poolingStrategy == .mean)
    #expect(config.resources.batchSize == 32)
    #expect(config.performance.useMetalAcceleration == true)
}

@Test("ModelMetadata creation")
func testModelMetadata() {
    let metadata = ModelMetadata(
        name: "test-model",
        version: "1.0",
        embeddingDimensions: 384,
        maxSequenceLength: 512,
        vocabularySize: 30522,
        modelType: "bert"
    )
    
    #expect(metadata.name == "test-model")
    #expect(metadata.embeddingDimensions == 384)
    #expect(metadata.vocabularySize == 30522)
}

@Test("EmbedTextCommand creation")
func testEmbedTextCommand() {
    let command = EmbedTextCommand(
        text: "Test embedding",
        modelIdentifier: ModelIdentifier(family: "test", variant: "base", version: "v1")
    )
    
    #expect(command.text == "Test embedding")
    #expect(command.modelIdentifier?.family == "test")
}

@Test("BatchEmbedCommand creation")
func testBatchEmbedCommand() {
    let texts = ["First text", "Second text", "Third text"]
    let command = BatchEmbedCommand(
        texts: texts,
        modelIdentifier: ModelIdentifier(family: "test", variant: "base", version: "v1")
    )
    
    #expect(command.texts == texts)
    #expect(command.modelIdentifier?.family == "test")
}

@Test("CacheStatistics calculation")
func testCacheStatistics() {
    let stats = CacheStatistics(
        hits: 80,
        misses: 20,
        evictions: 5,
        currentSize: 50,
        maxSize: 100
    )
    
    #expect(stats.hitRate == 0.8)
    #expect(stats.currentSize == 50)
    #expect(stats.evictions == 5)
}

// Using MockTextEmbedder from ComprehensiveBenchmarks.swift

@Test("MockTextEmbedder basic functionality")
func testMockTextEmbedder() async throws {
    let embedder = MockTextEmbedder()
    
    // Should throw when not loaded
    do {
        _ = try await embedder.embed("test")
        #expect(false, "Should have thrown error")
    } catch {
        #expect(error is ContextualEmbeddingError)
    }
    
    // Load model
    try await embedder.loadModel()
    #expect(await embedder.isReady == true)
    
    // Generate embedding
    let embedding = try await embedder.embed("test text")
    #expect(embedding.dimensions == 768) // MockTextEmbedder default dimensions
    
    // Unload model
    try await embedder.unloadModel()
    #expect(await embedder.isReady == false)
}
