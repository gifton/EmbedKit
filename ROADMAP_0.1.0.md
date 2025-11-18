# EmbedKit 0.1.0 Roadmap - Strategic Priorities

## Executive Summary

EmbedKit has solid foundations (tokenization, GPU acceleration, core types) but lacks the critical "brain" - model inference. For your journaling app MVP, we need to focus on getting **text → embedding → storage** working end-to-end.

**Current State**: 60% complete (foundation ready, missing inference layer)
**Target State**: Minimal viable embedding service for on-device semantic search
**Estimated Effort**: 2-3 weeks for 0.1.0 MVP

---

## Priority 1: CoreML Backend Implementation (3-4 days)
**Impact: CRITICAL - Unblocks everything else**

### Why First?
Without model inference, EmbedKit is just an elaborate tokenizer. Your journaling app needs actual embeddings.

### Implementation Plan:

```swift
// Sources/EmbedKit/Core/CoreMLBackend.swift
public actor CoreMLBackend: ModelBackend {
    private var model: MLModel?
    private let config: MLModelConfiguration

    public func loadModel(from url: URL) async throws {
        // 1. Load CoreML model
        // 2. Validate input/output shapes
        // 3. Cache model metadata
    }

    public func generateEmbeddings(for input: TokenizedInput) async throws -> ModelOutput {
        // 1. Convert TokenizedInput → MLMultiArray
        // 2. Run inference
        // 3. Extract token embeddings
        // 4. Return ModelOutput with all token embeddings
    }
}
```

### Recommended Models to Support First:
1. **MiniLM-L6-v2** (384D) - Fastest, good for mobile
2. **all-mpnet-base-v2** (768D) - Best quality/speed tradeoff
3. **BERT-base-uncased** (768D) - Most compatible

### CoreML Model Conversion:
```python
# Convert HuggingFace → CoreML (one-time setup)
from transformers import AutoModel, AutoTokenizer
import coremltools as ct

model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
traced = torch.jit.trace(model, example_inputs)
mlmodel = ct.convert(traced, inputs=[...])
mlmodel.save("MiniLM-L6-v2.mlmodel")
```

---

## Priority 2: EmbeddingPipeline Orchestrator (2 days)
**Impact: HIGH - Makes the library actually usable**

### Purpose:
Single entry point that chains: Text → Tokenize → Inference → Pool → Normalize → Embedding

### Implementation:

```swift
// Sources/EmbedKit/Core/EmbeddingPipeline.swift
public actor EmbeddingPipeline {
    private let tokenizer: any Tokenizer
    private let backend: any ModelBackend
    private let pooling: PoolingStrategy
    private let normalizer: MetalAccelerator?

    // Simple API for your journaling app
    public func embed(_ text: String) async throws -> DynamicEmbedding {
        // 1. Tokenize
        let tokens = try await tokenizer.tokenize(text)

        // 2. Generate token embeddings
        let output = try await backend.generateEmbeddings(for: tokens)

        // 3. Pool tokens → single vector
        let pooled = try await pool(output, strategy: pooling)

        // 4. Normalize (L2)
        let normalized = try await normalize(pooled)

        // 5. Return typed embedding
        return DynamicEmbedding(from: normalized)
    }

    // Batch API for efficiency
    public func embed(batch texts: [String]) async throws -> [DynamicEmbedding] {
        // Batch tokenization + inference for 10x speedup
    }
}
```

### Configuration for Journaling App:
```swift
let pipeline = EmbeddingPipeline(
    modelURL: Bundle.main.url(forResource: "MiniLM-L6-v2", withExtension: "mlmodel")!,
    tokenizer: BERTTokenizer(),
    pooling: .mean,  // Best for sentence embeddings
    useGPU: true
)

// Usage in your app
let entry = "Today I reflected on my morning meditation..."
let embedding = try await pipeline.embed(entry)
```

---

## Priority 3: VectorIndex Integration (1-2 days)
**Impact: HIGH - Enables semantic search in your journaling app**

### Integration Points:

```swift
// Sources/EmbedKit/Storage/VectorIndexAdapter.swift
import VectorIndex

public actor EmbedKitIndexAdapter {
    private let pipeline: EmbeddingPipeline
    private let index: VectorIndex  // From your VectorIndex package

    // Store journal entry with its embedding
    public func addEntry(_ text: String, metadata: [String: Any]) async throws -> UUID {
        let embedding = try await pipeline.embed(text)
        return try await index.add(
            vector: embedding.toFloatArray(),
            metadata: metadata
        )
    }

    // Semantic search across journal
    public func search(_ query: String, k: Int = 10) async throws -> [SearchResult] {
        let queryEmbedding = try await pipeline.embed(query)
        return try await index.search(
            query: queryEmbedding.toFloatArray(),
            k: k
        )
    }
}
```

### Journaling App Usage:
```swift
// Store entries
await adapter.addEntry(
    "Feeling grateful for the sunny weather",
    metadata: ["date": Date(), "mood": "happy"]
)

// Semantic search
let similar = await adapter.search("entries about gratitude")
// Returns journal entries semantically similar to "gratitude"
```

---

## Priority 4: Model Management (2 days)
**Impact: MEDIUM - Improves production readiness**

### Components Needed:

```swift
// Sources/EmbedKit/Models/ModelManager.swift
public actor ModelManager {
    private let cache: ModelCache
    private let registry: HuggingFaceRegistry

    // Auto-download and cache models
    public func loadModel(_ identifier: String) async throws -> URL {
        if let cached = cache.get(identifier) {
            return cached
        }

        let url = try await registry.download(identifier)
        cache.store(identifier, at: url)
        return url
    }
}

// Predefined models for easy use
public enum PretrainedModel: String {
    case miniLM = "sentence-transformers/all-MiniLM-L6-v2"
    case mpnet = "sentence-transformers/all-mpnet-base-v2"
    case distilBERT = "sentence-transformers/distilbert-base-nli-mean-tokens"

    var dimensions: Int {
        switch self {
        case .miniLM: return 384
        case .mpnet: return 768
        case .distilBERT: return 768
        }
    }
}
```

---

## Priority 5: Caching Layer (1 day)
**Impact: MEDIUM - 10x speedup for repeated content**

### LRU Cache for Embeddings:

```swift
// Sources/EmbedKit/Cache/EmbeddingCache.swift
actor EmbeddingCache {
    private var cache: LRUCache<String, DynamicEmbedding>

    func get(_ text: String) -> DynamicEmbedding? {
        return cache[text.sha256()]
    }

    func set(_ text: String, embedding: DynamicEmbedding) {
        cache[text.sha256()] = embedding
    }
}
```

**Impact for Journaling**: Instant responses when re-embedding similar text.

---

## Implementation Order & Timeline

### Week 1: Core Functionality
1. **Day 1-3**: CoreML Backend
   - Model loading
   - Inference execution
   - Output extraction

2. **Day 4-5**: EmbeddingPipeline
   - Chain components
   - Error handling
   - Batch processing

### Week 2: Integration & Polish
3. **Day 6-7**: VectorIndex Adapter
   - Storage interface
   - Search interface
   - Metadata handling

4. **Day 8-9**: Model Management
   - Download system
   - Caching
   - Model registry

5. **Day 10**: Testing & Examples
   - Integration tests
   - Journaling app example
   - Performance benchmarks

---

## Quick Win Implementation Path

For the **absolute fastest path to testing** in your journaling app:

### Option A: Minimal MVP (3 days)
1. Hardcode a single CoreML model (MiniLM)
2. Simple pipeline without caching
3. Direct VectorIndex integration
4. Skip model management for now

### Option B: Production-Ready (2 weeks)
Complete implementation as outlined above.

---

## Testing Strategy for Your Journaling App

### Phase 1: Unit Testing
```swift
// Test individual components
func testTokenization() async throws
func testCoreMLInference() async throws
func testPoolingStrategies() async throws
```

### Phase 2: Integration Testing
```swift
// Test end-to-end flow
func testTextToEmbedding() async throws {
    let pipeline = EmbeddingPipeline(...)
    let embedding = try await pipeline.embed("Test entry")
    XCTAssertEqual(embedding.dimensions, 384)
}
```

### Phase 3: Journaling App Testing
1. Embed all existing journal entries
2. Test semantic search accuracy
3. Measure performance (target: <100ms per entry)
4. Test memory usage (target: <100MB for 10k entries)

---

## Performance Targets for 0.1.0

### Embedding Generation
- Single text: <50ms (iPhone 14 Pro)
- Batch of 10: <200ms
- Batch of 100: <1.5s

### Memory Usage
- Model in memory: ~100MB (MiniLM)
- Per embedding: 1.5KB (384D)
- Cache overhead: <50MB

### Search Performance (with VectorIndex)
- 10k entries: <10ms
- 100k entries: <50ms
- 1M entries: <200ms

---

## Risks & Mitigations

### Risk 1: CoreML Model Compatibility
**Mitigation**: Test with multiple model architectures early. Have ONNX fallback.

### Risk 2: Memory Pressure on Device
**Mitigation**: Implement aggressive caching policies. Use 384D models for mobile.

### Risk 3: VectorIndex Performance
**Mitigation**: You mentioned it's "not incredibly performant" - consider implementing batched indexing and background processing.

---

## Success Criteria for 0.1.0

✅ **Must Have**:
- Embed text end-to-end
- Store embeddings in VectorIndex
- Semantic search works
- <100ms latency per entry
- Runs on iPhone 14+

🔄 **Should Have**:
- Multiple model support
- Batch processing
- Basic caching
- Error recovery

❓ **Nice to Have**:
- Model quantization
- Background indexing
- Cloud model support

---

## Next Steps

1. **Today**: Start CoreML backend implementation
2. **This Week**: Get basic pipeline working
3. **Next Week**: Integration and testing
4. **Week 3**: Deploy to journaling app TestFlight

The key is to **get something working quickly**, then iterate. Your journaling app users won't care about perfect architecture - they want semantic search that works.

## Sample Code to Get Started

```swift
// Minimal implementation to test in your app TODAY
actor QuickStartPipeline {
    func embed(_ text: String) async throws -> [Float] {
        // 1. Use existing tokenizer
        let tokenizer = BERTTokenizer()
        let tokens = try await tokenizer.tokenize(text)

        // 2. TODO: Add CoreML inference here
        // let model = try MLModel(contentsOf: modelURL)
        // let output = try model.prediction(from: tokens)

        // 3. For now, return random embedding to test integration
        return (0..<384).map { _ in Float.random(in: -1...1) }
    }
}
```

This gets you testing the integration points while implementing the real inference.