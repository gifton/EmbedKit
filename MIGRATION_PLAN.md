# EmbedKit Architecture Migration Plan

## Overview
Migrating EmbedKit from a BERT-specific implementation to a flexible, multi-model embedding framework that supports Apple's on-device models and other providers.

## Migration Phases

### Phase 1: Core Abstractions (Week 1-2)
- [ ] Define core protocols (`EmbeddingModelProtocol`, `TokenizerProtocol`, `ModelBackendProtocol`)
- [ ] Implement model identification system (`ModelIdentifier`, `ModelProvider`)
- [ ] Create capability advertisement system (`ModelCapabilities`)
- [ ] Design configuration protocol hierarchy

**Breaking Changes:** None (new code only)

### Phase 2: Registry & Factory Pattern (Week 2-3)
- [ ] Implement `ModelRegistry` actor for thread-safe model management
- [ ] Create factory protocol and base implementations
- [ ] Add model discovery and filtering capabilities
- [ ] Implement model caching and lifecycle management

**Breaking Changes:** None (new code only)

### Phase 3: Tokenizer Refactor (Week 3-4)
- [ ] Extract tokenizer strategies into separate implementations
- [ ] Create `WordPieceTokenizer`, `BPETokenizer`, `SentencePieceTokenizer`
- [ ] Implement vocabulary abstraction protocol
- [ ] Add special token configuration system

**Breaking Changes:**
- `VocabularyBuilder` API changes to support strategies
- `BERTTokenizer` becomes a specific implementation of `WordPieceTokenizer`

**Migration Path:**
```swift
// Old
let tokenizer = BERTTokenizer()

// New (backward compatible wrapper)
let tokenizer = BERTTokenizer()  // Still works

// New (recommended)
let tokenizer = WordPieceTokenizer(strategy: .bert)
```

### Phase 4: Backend Abstraction (Week 4-5)
- [ ] Create backend protocol for different compute providers
- [ ] Implement `CoreMLBackend` for Apple models
- [ ] Refactor existing `MetalAccelerator` as `MetalBackend`
- [ ] Add ONNX backend support (optional)

**Breaking Changes:** Internal only

### Phase 5: Apple Model Integration (Week 5-6)
- [ ] Implement `AppleEmbeddingModel`
- [ ] Create Apple-specific tokenizer if needed
- [ ] Add CoreML model loading and caching
- [ ] Implement Neural Engine optimization

**Breaking Changes:** None (new feature)

### Phase 6: API Refinement (Week 6-7)
- [ ] Create high-level `EmbedKitAPI` for simple use cases
- [ ] Implement fluent configuration builders
- [ ] Add comprehensive error handling
- [ ] Create migration helpers

**Breaking Changes:** None (new API layer)

### Phase 7: Testing & Documentation (Week 7-8)
- [ ] Comprehensive protocol conformance tests
- [ ] Performance benchmarks for all models
- [ ] Integration tests across model providers
- [ ] Update all documentation
- [ ] Create migration guides

## Backward Compatibility Strategy

### 1. Wrapper Classes
```swift
// Maintain old API surface
public class BERTTokenizer {
    private let implementation: WordPieceTokenizer

    public init(/* old parameters */) {
        self.implementation = WordPieceTokenizer(
            strategy: .bert,
            // map old parameters
        )
    }

    // Forward all methods
    public func tokenize(_ text: String) async throws -> TokenizedInput {
        try await implementation.tokenize(text)
    }
}
```

### 2. Deprecation Warnings
```swift
@available(*, deprecated, renamed: "WordPieceTokenizer")
public typealias BERTTokenizer = WordPieceTokenizer
```

### 3. Configuration Migration
```swift
extension ModelConfiguration {
    static func fromLegacy(_ old: BERTConfiguration) -> ModelConfiguration {
        ModelConfigurationBuilder()
            .withMaxSequenceLength(old.maxLength)
            .withDimensions(old.hiddenSize)
            .withPoolingStrategy(.cls)  // BERT default
            .build()
    }
}
```

## Testing Strategy

### Unit Tests
- Protocol conformance for all implementations
- Individual component testing
- Mock implementations for testing

### Integration Tests
- End-to-end embedding generation
- Cross-model consistency
- Performance benchmarks

### Regression Tests
- Ensure backward compatibility
- Test migration paths
- Verify deprecated APIs still work

## Performance Considerations

### Memory Management
- Lazy model loading
- Automatic unloading of unused models
- Configurable cache sizes

### Optimization Opportunities
- Batch processing for all models
- Parallel tokenization
- GPU memory pooling
- Token caching

## Risk Mitigation

### Risk 1: Breaking Existing Code
**Mitigation:** Comprehensive backward compatibility layer

### Risk 2: Performance Regression
**Mitigation:** Extensive benchmarking, optimization passes

### Risk 3: Model Compatibility Issues
**Mitigation:** Thorough testing with real models, fallback mechanisms

### Risk 4: API Complexity
**Mitigation:** Simple high-level API, comprehensive documentation

## Success Metrics

1. **API Simplicity:** 3 lines of code for basic embedding
2. **Performance:** < 10% overhead vs. direct model calls
3. **Model Support:** 5+ models at launch
4. **Test Coverage:** > 90% code coverage
5. **Documentation:** 100% public API documented
6. **Breaking Changes:** Zero for existing users

## Timeline

| Phase | Duration | Start Date | End Date | Status |
|-------|----------|------------|----------|--------|
| Phase 1 | 2 weeks | Nov 18 | Dec 1 | Not Started |
| Phase 2 | 1 week | Dec 2 | Dec 8 | Not Started |
| Phase 3 | 1 week | Dec 9 | Dec 15 | Not Started |
| Phase 4 | 1 week | Dec 16 | Dec 22 | Not Started |
| Phase 5 | 1 week | Dec 23 | Dec 29 | Not Started |
| Phase 6 | 1 week | Dec 30 | Jan 5 | Not Started |
| Phase 7 | 1 week | Jan 6 | Jan 12 | Not Started |

## Code Examples

### Simple Usage
```swift
// Automatic model selection
let api = EmbedKitAPI()
let embedding = try await api.embed("Hello, world!")
```

### Specific Model
```swift
// Use Apple's on-device model
let embedding = try await api.embed(
    text: "Hello, world!",
    using: "apple/text-embedding-base-v1.0"
)
```

### Advanced Configuration
```swift
// Custom configuration
let config = ModelConfigurationBuilder()
    .withMaxSequenceLength(256)
    .withPoolingStrategy(.attentionWeighted)
    .withNormalization(false)
    .build()

let model = try await registry.loadModel(
    ModelIdentifier(provider: .apple, name: "text-embedding", version: "1.0"),
    configuration: config
)

let embedding = try await model.embed("Hello, world!")
```

### Batch Processing
```swift
let texts = ["text1", "text2", "text3"]
let embeddings = try await api.embedBatch(
    texts: texts,
    using: .apple
)
```

## Next Steps

1. Review and approve architecture design
2. Set up feature branches for each phase
3. Begin Phase 1 implementation
4. Weekly progress reviews
5. Continuous integration testing