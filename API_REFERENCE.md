# EmbedKit API Reference

Quick reference for all public APIs in EmbedKit.

## Core Types

### Embedding<D>

Generic embedding type with compile-time dimension verification.

```swift
public struct Embedding<D: EmbeddingDimension>: Sendable, Equatable, Hashable, Codable

// Type aliases
public typealias Embedding384 = Embedding<Dim384>   // 384 dimensions
public typealias Embedding768 = Embedding<Dim768>   // 768 dimensions
public typealias Embedding1536 = Embedding<Dim1536> // 1536 dimensions

// Initialization
init(_ values: [Float]) throws
static func zeros() -> Self
static func ones() -> Self
static func random(in range: ClosedRange<Float> = 0...1) -> Self

// Properties
var dimensions: Int { get }
var magnitude: Float { get }
var magnitudeSquared: Float { get }
var isFinite: Bool { get }
var isZero: Bool { get }

// Operations
func normalized() -> Result<Self, VectorError>
func cosineSimilarity(to other: Self) -> Float
func cosineDistance(to other: Self) -> Float
func euclideanDistance(to other: Self) -> Float
func dotProduct(with other: Self) -> Float
func toArray() -> [Float]
```

### DynamicEmbedding

Runtime-typed embedding for heterogeneous collections.

```swift
public enum DynamicEmbedding: Sendable, Codable

// Cases
case dim384(Embedding384)
case dim768(Embedding768)
case dim1536(Embedding1536)

// Initialization
init(values: [Float]) throws
init(from decoder: Decoder) throws

// Properties
var dimensions: Int { get }
var isFinite: Bool { get }
var isZero: Bool { get }

// Operations
func normalized() throws -> Self
func cosineSimilarity(to other: Self) throws -> Float
func toArray() -> [Float]
```

## Pipeline

### EmbeddingPipeline

Main orchestrator for text to embedding conversion.

```swift
public actor EmbeddingPipeline

// Initialization
init(
    tokenizer: any Tokenizer,
    backend: any ModelBackend,
    configuration: EmbeddingPipelineConfiguration = .init()
)

init(
    modelURL: URL,
    tokenizer: any Tokenizer,
    configuration: EmbeddingPipelineConfiguration = .init()
) async throws

// Model Management
func loadModel(from url: URL) async throws
func isReady() -> Bool

// Embedding Generation
func embed(_ text: String) async throws -> DynamicEmbedding
func embed(batch texts: [String]) async throws -> [DynamicEmbedding]

// Statistics
func getStatistics() -> PipelineStatistics
```

### EmbeddingPipelineConfiguration

```swift
public struct EmbeddingPipelineConfiguration: Sendable

init(
    poolingStrategy: PoolingStrategy = .mean,
    normalize: Bool = true,
    useGPUAcceleration: Bool = true,
    cacheConfiguration: CacheConfiguration? = .init(),
    batchSize: Int = 32
)
```

## Tokenization

### Tokenizer Protocol

```swift
public protocol Tokenizer: Actor {
    func tokenize(_ text: String) async throws -> TokenizedInput
    func tokenize(batch texts: [String]) async throws -> [TokenizedInput]
    var maxSequenceLength: Int { get }
    var vocabularySize: Int { get }
    var specialTokens: SpecialTokens { get }
}
```

### BERTTokenizer

```swift
public actor BERTTokenizer: Tokenizer

init(
    vocabularyPath: URL? = nil,
    maxLength: Int = 512,
    doLowerCase: Bool = true,
    configuration: TokenizerConfiguration = .init()
) async throws
```

### TokenizedInput

```swift
public struct TokenizedInput: Sendable {
    let tokenIds: [Int]
    let attentionMask: [Int]
    let tokenTypeIds: [Int]?
    let originalLength: Int
}
```

## Model Management

### ModelManager

```swift
public actor ModelManager

// Initialization
init(configuration: Configuration = .init()) async throws

// Model Loading
func loadModel(_ identifier: String) async throws -> any ModelBackend
func loadModel(_ model: PretrainedModel) async throws -> any ModelBackend

// Pipeline Creation
func getPipeline(
    for model: PretrainedModel,
    configuration: EmbeddingPipelineConfiguration = .init()
) async throws -> EmbeddingPipeline

// Model Management
func downloadModel(
    _ identifier: String,
    progressHandler: ((DownloadProgress) -> Void)? = nil
) async throws -> URL
func isLoaded(_ identifier: String) -> Bool
func unloadModel(_ identifier: String)

// Information
func listAvailableModels() async -> [EmbeddingModelInfo]
func listCachedModels() async -> [CacheEntry]
func getCacheStatistics() async -> ModelCache.CacheStatistics
```

### PretrainedModel

```swift
public enum PretrainedModel: String, CaseIterable {
    case miniLM_L6_v2 = "sentence-transformers/all-MiniLM-L6-v2"
    case miniLM_L12_v2 = "sentence-transformers/all-MiniLM-L12-v2"
    case mpnetBase_v2 = "sentence-transformers/all-mpnet-base-v2"
    case distilBERT_v1 = "sentence-transformers/all-distilroberta-v1"

    var info: EmbeddingModelInfo { get }
    static func recommended(for useCase: UseCase) -> PretrainedModel
}
```

## Storage

### VectorIndexAdapter

```swift
public actor VectorIndexAdapter

// Initialization
init(
    pipeline: EmbeddingPipeline,
    storage: any VectorStorageBackend,
    configuration: VectorIndexConfiguration = .init()
)

// Text Storage
func addText(
    _ text: String,
    metadata: VectorMetadata? = nil
) async throws -> UUID

func addTexts(
    _ texts: [String],
    metadata: [VectorMetadata?] = []
) async throws -> [UUID]

// Embedding Storage
func addEmbedding(
    _ embedding: DynamicEmbedding,
    metadata: [String: Any]
) async throws -> UUID

// Semantic Search with Real Reranking
func semanticSearch(
    query: String,
    k: Int = 10,
    rerankStrategy: (any RerankingStrategy)? = nil,
    rerankOptions: RerankOptions = .default,
    filter: [String: Any]? = nil
) async throws -> [VectorSearchResult]

// Batch Search
func batchSearch(
    queries: [String],
    k: Int = 10,
    rerankStrategy: (any RerankingStrategy)? = nil,
    rerankOptions: RerankOptions = .default
) async throws -> [[VectorSearchResult]]

// Legacy Search Methods
func searchByText(
    _ query: String,
    k: Int = 10,
    threshold: Float? = nil,
    includeEmbeddings: Bool = false
) async throws -> [VectorSearchResult]

func searchByEmbedding(
    _ embedding: DynamicEmbedding,
    k: Int = 10,
    threshold: Float? = nil,
    includeEmbeddings: Bool = false
) async throws -> [VectorSearchResult]

// Management
func remove(id: UUID) async throws
func count() async -> Int
func clear() async throws
```

### VectorStorageBackend Protocol

```swift
public protocol VectorStorageBackend: Actor {
    func add(vector: [Float], metadata: [String: Any]) async throws -> UUID
    func addBatch(vectors: [[Float]], metadata: [[String: Any]]) async throws -> [UUID]
    func search(query: [Float], k: Int, threshold: Float?) async throws -> [(id: UUID, score: Float, metadata: [String: Any])]
    func remove(id: UUID) async throws
    func get(id: UUID) async throws -> (vector: [Float], metadata: [String: Any])?
    func count() async -> Int
    func clear() async throws
}
```

## Reranking

### RerankingStrategy Protocol

```swift
public protocol RerankingStrategy: Sendable {
    func rerank(
        query: DynamicEmbedding,
        candidates: [VectorSearchResult],
        k: Int,
        options: RerankOptions
    ) async throws -> [VectorSearchResult]
}
```

### ExactRerankStrategy

Real reranking that recomputes exact distances from original vectors.

```swift
public struct ExactRerankStrategy: RerankingStrategy {
    init(
        storage: any VectorStorageBackend,
        metric: SupportedDistanceMetric = .euclidean,
        dimension: Int
    )

    // Recomputes distances for accurate reranking
    func rerank(
        query: DynamicEmbedding,
        candidates: [VectorSearchResult],
        k: Int,
        options: RerankOptions = .default
    ) async throws -> [VectorSearchResult]
}
```

### CrossEncoderStrategy

Placeholder for future neural reranking implementation.

```swift
public struct CrossEncoderStrategy: RerankingStrategy {
    // Future implementation for cross-encoder reranking
}
```

### RerankOptions

```swift
public struct RerankOptions: Sendable {
    var candidateMultiplier: Int = 3    // Fetch N * k candidates
    var enableParallel: Bool = true     // Use concurrent processing
    var maxConcurrency: Int = 8         // Max parallel tasks
    var tileSize: Int = 256             // Batch size for processing
    var skipMissing: Bool = true        // Skip vectors not in storage

    // Presets
    static let `default` = RerankOptions()  // Balanced
    static let fast: RerankOptions          // Speed priority (2x, no parallel)
    static let accurate: RerankOptions      // Quality priority (5x, parallel)
}
```

### Usage Example

```swift
// Create reranking strategy
let reranker = ExactRerankStrategy(
    storage: storage,
    metric: .cosine,
    dimension: 384
)

// Search with reranking
let results = try await adapter.semanticSearch(
    query: "machine learning",
    k: 10,
    rerankStrategy: reranker,
    rerankOptions: .accurate
)

// Batch search with reranking
let batchResults = try await adapter.batchSearch(
    queries: ["AI", "ML", "NLP"],
    k: 5,
    rerankStrategy: reranker
)
```

## GPU Acceleration

### MetalAccelerator

```swift
public actor MetalAccelerator

// Singleton
static let shared: MetalAccelerator

// Operations
func normalize(_ vectors: [[Float]]) async throws -> [[Float]]
func poolEmbeddings(
    tokenEmbeddings: [[Float]],
    attentionMask: [Int],
    strategy: PoolingStrategy
) async throws -> [Float]
func cosineSimilarity(_ a: [Float], _ b: [Float]) async throws -> Float
```

### PoolingStrategy

```swift
public enum PoolingStrategy: String, Sendable {
    case mean = "mean"                           // Average all token embeddings
    case cls = "cls"                            // Use [CLS] token only
    case max = "max"                            // Max pooling across tokens
    case attentionWeighted = "attention_weighted" // Weighted by attention
}
```

## Error Types

### EmbeddingError

```swift
public enum EmbeddingError: LocalizedError {
    case dimensionMismatch(expected: Int, actual: Int)
    case unsupportedDimension(Int)
    case invalidInput(String)
}
```

### EmbeddingPipelineError

```swift
public enum EmbeddingPipelineError: LocalizedError {
    case modelNotLoaded
    case tokenizationFailed(Error)
    case inferenceFailed(Error)
    case poolingFailed(Error)
    case normalizationFailed(Error)
    case dimensionMismatch(expected: Int, actual: Int)
    case emptyInput
}
```

### CoreMLError

```swift
public enum CoreMLError: LocalizedError {
    case modelLoadingFailed(URL, Error)
    case invalidModelFormat(String)
    case inputShapeMismatch(expected: [Int], actual: [Int])
    case inferenceFailure(Error)
    case missingRequiredInput(String)
    case missingRequiredOutput(String)
}
```

### MetalError

```swift
public enum MetalError: LocalizedError {
    case deviceNotAvailable
    case bufferCreationFailed
    case pipelineNotFound(String)
    case encoderCreationFailed
    case commandBufferCreationFailed
    case invalidInput(String)
    case dimensionMismatch
}
```

## Configuration Types

### CoreMLConfiguration

```swift
public struct CoreMLConfiguration: Sendable {
    let useNeuralEngine: Bool
    let allowCPUFallback: Bool
    let maxBatchSize: Int
    let inputNames: InputNames
    let outputNames: OutputNames
}
```

### TokenizerConfiguration

```swift
public struct TokenizerConfiguration: Sendable {
    let paddingStrategy: PaddingStrategy
    let truncationStrategy: TruncationStrategy
    let maxLength: Int
    let stride: Int
    let padTokenId: Int
}
```

### VectorIndexConfiguration

```swift
public struct VectorIndexConfiguration: Sendable {
    let distanceMetric: DistanceMetric
    let maxVectors: Int
    let indexType: String
}
```

## Utilities

### Model Output

```swift
public struct ModelOutput: Sendable {
    let tokenEmbeddings: [[Float]]
    let attentionWeights: [[Float]]?
    let metadata: [String: String]
}
```

### Vector Search Result

```swift
public struct VectorSearchResult: Sendable {
    let id: String                  // Changed from UUID for flexibility
    var score: Float                 // Mutable for reranking updates
    let metadata: [String: String]   // Type-safe metadata
    let embedding: DynamicEmbedding?
}
```

### Pipeline Statistics

```swift
public struct PipelineStatistics: Sendable {
    let tokenizationTime: TimeInterval
    let inferenceTime: TimeInterval
    let poolingTime: TimeInterval
    let normalizationTime: TimeInterval
    let totalTime: TimeInterval
    let cacheHitRate: Double
}
```

---

For detailed usage examples and best practices, see the [README](README.md) and [ARCHITECTURE](ARCHITECTURE.md) documents.