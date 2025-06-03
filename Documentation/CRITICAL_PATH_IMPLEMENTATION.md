# Critical Path Implementation Plans

## Package Distribution Summary

### EmbedKit
- Metal acceleration completion (cosine similarity, attention pooling)
- Metal test fixes and CI configuration
- Core ML test improvements

### VectorStoreKit  
- PipelineKit commands and handlers
- VectorStore middleware
- Performance test adjustments

### Shared Module (New)
- Shared data types (EmbeddingDocument, etc.)
- Composite commands (IndexDocumentCommand)
- Pipeline factories and streaming bridges

### Both Packages
- Test infrastructure improvements
- CI/CD configuration
- Integration test fixes

---

## 1. Complete EmbedKit Phase 3.5 Metal Work

**Package: EmbedKit**

### Overview
Complete the remaining Metal acceleration features and ensure all Metal tests pass on available hardware in the EmbedKit package.

### Metal Cosine Similarity Kernel

**Objective**: Implement optimized Metal kernel for cosine similarity calculations

**Tasks**:
1. **Create Metal Shader**
   ```metal
   // Sources/EmbedKit/Acceleration/Shaders/CosineSimilarity.metal
   kernel void cosine_similarity(
       constant float* vectorA [[buffer(0)]],
       constant float* vectorB [[buffer(1)]],
       constant uint& dimensions [[buffer(2)]],
       device float* result [[buffer(3)]],
       uint id [[thread_position_in_grid]])
   ```
   - Implement dot product calculation
   - Add normalization handling
   - Optimize for different vector sizes

2. **Update MetalSimilarityProcessor**
   ```swift
   // Add to MetalSimilarityProcessor.swift
   public func cosineSimilarity(
       _ vectorA: [Float],
       _ vectorB: [Float]
   ) async throws -> Float
   ```
   - Create Metal buffers for input vectors
   - Setup compute pipeline
   - Handle edge cases (zero vectors, dimension mismatch)

3. **Optimize for Batch Operations**
   - Implement batch cosine similarity matrix
   - Use threadgroup memory for better performance
   - Add SIMD optimizations where applicable

4. **Testing & Benchmarking**
   - Create unit tests for accuracy validation
   - Benchmark against CPU implementation
   - Test with various vector dimensions (128, 384, 768, 1024)

### Attention-Weighted Pooling Implementation

**Objective**: Complete the attention-weighted pooling support in Metal

**Tasks**:
1. **Design Attention Mechanism**
   - Review transformer attention patterns
   - Design Metal-friendly algorithm
   - Plan memory layout for efficiency

2. **Implement Metal Kernel**
   ```metal
   // Add to Pooling.metal
   kernel void attention_weighted_pooling(
       constant float* embeddings [[buffer(0)]],
       constant float* attention_weights [[buffer(1)]],
       constant uint& sequence_length [[buffer(2)]],
       constant uint& embedding_dim [[buffer(3)]],
       device float* output [[buffer(4)]])
   ```
   - Apply attention weights to embeddings
   - Sum weighted embeddings
   - Normalize output

3. **Update MetalPoolingProcessor**
   ```swift
   case .attentionWeighted(let weights):
       return try await performAttentionWeightedPooling(
           embeddings: embeddings,
           weights: weights
       )
   ```

4. **Integration Testing**
   - Test with various sequence lengths
   - Validate against reference implementation
   - Ensure numerical stability

### Enable Metal Tests & Fix Issues

**Objective**: Get all Metal tests passing in CI/CD environment

**Tasks**:
1. **Environment Detection**
   ```swift
   // Add to MetalAccelerator.swift
   static var isMetalAvailable: Bool {
       #if targetEnvironment(simulator)
       return false
       #else
       return MTLCreateSystemDefaultDevice() != nil
       #endif
   }
   ```

2. **Update Test Infrastructure**
   - Add Metal availability checks to all tests
   - Create CPU fallback paths for CI
   - Mock Metal functionality where needed

3. **Fix Failing Tests**
   - Debug the 11 skipped Metal tests
   - Update test expectations for different platforms
   - Handle precision differences between GPU/CPU

4. **CI/CD Configuration**
   ```yaml
   # .github/workflows/tests.yml
   - name: Run Tests with Metal
     run: |
       if [[ "${{ matrix.os }}" == "macos-14" ]]; then
         swift test --enable-metal-tests
       else
         swift test
       fi
   ```

5. **Documentation**
   - Document Metal requirements
   - Add troubleshooting guide
   - Update README with Metal status

### Success Criteria
- [ ] All Metal tests pass on macOS 14+
- [ ] Cosine similarity performance: 10x faster than CPU for batch operations
- [ ] Attention pooling accuracy within 0.001 of reference implementation
- [ ] CI/CD runs successfully with conditional Metal tests

---

## 2. Add PipelineKit Bridge Between Packages

**Packages: EmbedKit + VectorStoreKit + Shared Module**

### Overview
Create seamless integration between EmbedKit and VectorStoreKit through PipelineKit commands and middleware.

### Design Integration Architecture

**Objective**: Design the command flow and data structures for integration

**Tasks**:
1. **Define Shared Types** (New Shared Module)
   ```swift
   // Create SharedTypes package or module
   public struct EmbeddingDocument: Sendable {
       public let id: String
       public let text: String
       public let embedding: EmbeddingVector
       public let metadata: [String: Any]
   }
   ```

2. **Design Command Flow**
   - Text → EmbedCommand → VectorStoreCommand → Result
   - Batch operations support
   - Streaming pipeline design

3. **Create Integration Protocol**
   ```swift
   public protocol EmbeddingIndexer: Actor {
       func index(_ text: String, metadata: [String: Any]) async throws -> String
       func search(_ query: String, k: Int) async throws -> [SearchResult]
       func indexBatch(_ documents: [Document]) async throws -> [String]
   }
   ```

### Implement VectorStoreKit Commands

**Objective**: Create PipelineKit commands for VectorStoreKit operations

**Tasks**:
1. **Create Store Commands** (VectorStoreKit)
   ```swift
   // VectorStoreKit/Sources/PipelineIntegration/Commands.swift
   public struct StoreEmbeddingCommand: Command {
       public let embedding: EmbeddingVector
       public let metadata: [String: Any]
       
       public func execute() async throws -> String {
           // Store in vector database
       }
   }
   
   public struct SearchCommand: Command {
       public let query: EmbeddingVector
       public let k: Int
       public let filter: MetadataFilter?
   }
   ```

2. **Implement Handlers** (VectorStoreKit)
   ```swift
   public struct VectorStoreHandler: CommandHandler {
       private let store: VectorStore
       
       public func handle<C: Command>(_ command: C) async throws -> C.Output {
           switch command {
           case let cmd as StoreEmbeddingCommand:
               return try await store.add(cmd.embedding, metadata: cmd.metadata)
           case let cmd as SearchCommand:
               return try await store.search(cmd.query, k: cmd.k)
           default:
               throw HandlerError.unsupportedCommand
           }
       }
   }
   ```

3. **Create Middleware** (VectorStoreKit)
   ```swift
   public struct VectorStoreMiddleware: Middleware {
       public func process<C: Command>(
           _ command: C,
           next: Next<C>
       ) async throws -> C.Output {
           // Add caching, logging, metrics
       }
   }
   ```

### Build Integration Pipeline

**Objective**: Create complete pipeline from text to stored embeddings

**Tasks**:
1. **Create Composite Commands** (Shared Module)
   ```swift
   public struct IndexDocumentCommand: Command {
       public let document: Document
       
       public func execute(context: CommandContext) async throws -> IndexResult {
           // Embed text
           let embedding = try await context.execute(
               EmbedTextCommand(text: document.text)
           )
           
           // Store embedding
           let id = try await context.execute(
               StoreEmbeddingCommand(
                   embedding: embedding,
                   metadata: document.metadata
               )
           )
           
           return IndexResult(documentId: id, embedding: embedding)
       }
   }
   ```

2. **Build Pipeline Factory** (Shared Module)
   ```swift
   public struct EmbeddingIndexPipeline {
       public static func create(
           embedder: TextEmbedder,
           store: VectorStore
       ) -> Pipeline {
           return Pipeline()
               .use(ValidationMiddleware())
               .use(EmbeddingMiddleware(embedder: embedder))
               .use(VectorStoreMiddleware(store: store))
               .use(MetricsMiddleware())
       }
   }
   ```

3. **Implement Streaming Bridge** (Shared Module)
   ```swift
   public actor StreamingIndexer {
       func indexStream(
           _ documents: AsyncSequence<Document>
       ) -> AsyncStream<IndexResult> {
           // Stream documents through embedding and storage
       }
   }
   ```

### Testing & Documentation

**Objective**: Ensure robust integration with comprehensive tests

**Tasks**:
1. **Integration Tests**
   ```swift
   @Test("End-to-end document indexing")
   func testDocumentIndexing() async throws {
       let pipeline = EmbeddingIndexPipeline.create(
           embedder: embedder,
           store: store
       )
       
       let result = try await pipeline.execute(
           IndexDocumentCommand(document: testDocument)
       )
       
       #expect(result.documentId != nil)
   }
   ```

2. **Performance Tests**
   - Measure pipeline overhead
   - Test batch performance
   - Validate streaming throughput

3. **Documentation**
   - Create integration guide
   - Add example projects
   - Document best practices

### Success Criteria
- [ ] Seamless text → embedding → storage pipeline
- [ ] Less than 5% overhead from pipeline integration
- [ ] Support for batch and streaming operations
- [ ] Comprehensive test coverage (>90%)

---

## 3. Fix Remaining Test Failures

**Packages: EmbedKit + VectorStoreKit**

### Overview
Ensure all tests pass consistently across different environments and configurations in both packages.

### Test Infrastructure Improvements

**Objective**: Create robust test environment handling

**Tasks**:
1. **Environment Capability Detection** (Both Packages)
   ```swift
   // Tests/TestHelpers/TestEnvironment.swift
   struct TestEnvironment {
       static var hasMetalSupport: Bool {
           #if os(macOS) && !targetEnvironment(simulator)
           return MetalAccelerator.isAvailable
           #else
           return false
           #endif
       }
       
       static var hasCoreMLModels: Bool {
           // Check for test models in bundle
       }
       
       static var hasHighMemory: Bool {
           ProcessInfo.processInfo.physicalMemory > 8_000_000_000
       }
   }
   ```

2. **Conditional Test Execution** (Both Packages)
   ```swift
   // Create test traits
   extension Trait where Self == EnvironmentTrait {
       static var requiresMetal: Self {
           EnvironmentTrait(
               isEnabled: TestEnvironment.hasMetalSupport,
               skipMessage: "Metal not available"
           )
       }
   }
   ```

3. **Mock Implementations** (EmbedKit)
   ```swift
   // Create comprehensive mocks for CI
   actor MockMetalAccelerator: MetalAcceleratorProtocol {
       func normalizeVectors(_ vectors: [[Float]]) async throws -> [[Float]] {
           // CPU implementation for testing
       }
   }
   ```

### Fix Specific Test Categories

**Objective**: Address failing tests by category

**Tasks**:
1. **Metal Tests** (EmbedKit)
   - Add proper capability checks
   - Create CPU reference implementations
   - Handle precision differences

2. **Core ML Tests** (EmbedKit)
   - Bundle small test models
   - Create model loader mocks
   - Handle missing model scenarios

3. **Performance Tests** (Both Packages)
   - Adjust thresholds for CI environments
   - Add warm-up phases
   - Handle resource constraints

4. **Integration Tests** (Both Packages)
   - Fix race conditions
   - Add proper timeouts
   - Handle async test coordination

### CI/CD Configuration

**Objective**: Ensure tests run reliably in CI

**Tasks**:
1. **GitHub Actions Matrix** (Both Packages)
   ```yaml
   strategy:
     matrix:
       include:
         - os: macos-14
           xcode: 16.0
           enable-metal: true
         - os: macos-13
           xcode: 15.4
           enable-metal: false
         - os: ubuntu-latest
           swift: 6.0
           enable-metal: false
   ```

2. **Test Reporting**
   - Add test result artifacts
   - Create coverage reports
   - Set up failure notifications

3. **Performance Regression Detection**
   - Store baseline metrics
   - Compare against previous runs
   - Alert on significant regressions

### Success Criteria
- [ ] 100% test pass rate on all CI environments
- [ ] Clear skip messages for unavailable features
- [ ] Test execution time under 5 minutes
- [ ] Automated performance regression detection