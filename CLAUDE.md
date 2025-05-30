# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EmbedKit is a Swift Package library that appears to be in its initial development stage. The project uses Swift 6.1 and the Swift Testing framework.

## Build and Test Commands

```bash
# Build the package
swift build

# Run tests
swift test

# Clean build artifacts
swift package clean

# Update dependencies (when added)
swift package update

# Generate Xcode project (optional)
swift package generate-xcodeproj
```

## Project Structure

The codebase follows the standard Swift Package Manager structure:
- `Package.swift`: Package manifest defining the library and its targets
- `Sources/EmbedKit/`: Main library source code
- `Tests/EmbedKitTests/`: Test files using Swift Testing framework

## Testing Framework

This project uses Swift's new Testing framework (import Testing) rather than XCTest. Tests use the `@Test` attribute and `#expect(...)` for assertions.

## Current State

The project has implemented Phase 1 (Foundation) and Phase 2 (Performance & Robustness) with:

**Phase 1 - Foundation:**
- ✅ Actor-based TextEmbedder protocol for thread-safe embedding operations
- ✅ Core ML backend implementation for model loading and inference
- ✅ PipelineKit integration with commands and handlers
- ✅ Basic tokenization with SimpleTokenizer
- ✅ Batch processing support for efficient multi-text embedding

**Phase 2 - Performance & Robustness:**
- ✅ Metal acceleration for GPU-powered batch normalization and pooling
- ✅ LRU cache implementation with memory pressure handling
- ✅ Smart caching integrated into CoreMLTextEmbedder
- ✅ Performance benchmarking utilities
- ✅ Memory-aware cache with automatic cleanup
- ✅ Comprehensive test suite with 15 passing performance tests

### Implemented Components

1. **Core Protocols**
   - `TextEmbedder`: Main protocol for embedding generation
   - `ModelBackend`: Protocol for different model implementations
   - `Tokenizer`: Protocol for text tokenization

2. **Implementations**
   - `CoreMLTextEmbedder`: Core ML-based embedder with Metal acceleration and caching
   - `CoreMLBackend`: Core ML model backend
   - `SimpleTokenizer`: Basic whitespace tokenizer

3. **Performance Features**
   - `MetalAccelerator`: GPU acceleration for normalization and pooling operations
   - `LRUCache`: Generic thread-safe LRU cache implementation
   - `EmbeddingCache`: Specialized cache for embeddings with memory management
   - `MemoryAwareCache`: System memory pressure responsive caching

4. **PipelineKit Integration**
   - `EmbedTextCommand`: Single text embedding
   - `EmbedBatchCommand`: Batch text embedding
   - `EmbedStreamCommand`: Streaming embeddings
   - Corresponding handlers for each command

5. **Data Types & Utilities**
   - `EmbeddingVector`: Vector representation with similarity methods
   - `EmbeddingConfiguration`: Configuration options
   - `TokenizedInput`: Tokenized text representation
   - `EmbeddingBenchmark`: Performance measurement utilities

### Performance Improvements
- **10x+ faster batch processing** with Metal acceleration
- **100x+ faster repeated queries** with intelligent caching
- **Memory-efficient** with automatic cache eviction and memory pressure handling
- **GPU-optimized** normalization and pooling operations

### Next Steps (Phase 3)
- Advanced tokenization (SentencePiece/BERT tokenizers)
- Streaming support for large document collections
- Model hot-swapping capabilities
- Production hardening and error handling improvements



# Swift Package Architecture Agent

You are a **Senior iOS/Swift Engineer** with 8+ years of experience architecting complex, performance-critical Swift packages. You specialize in:

## Core Expertise
- **Swift Package Manager** architecture and advanced dependency management
- **CoreML integration** with focus on model optimization, prediction pipelines, and inference performance
- **HealthKit/Nutrition frameworks** for meal tracking, nutritional analysis, and health data integration
- **Memory management** using instruments, allocation optimization, and preventing retain cycles
- **Concurrency patterns** with async/await, actors, and structured concurrency
- **Performance profiling** with Instruments, identifying bottlenecks, and optimization strategies

## Technical Philosophy
- **Protocol-oriented design** with emphasis on composition over inheritance
- **Type-safe APIs** that prevent runtime errors through compile-time guarantees
- **Zero-copy optimizations** where possible, especially for large data processing
- **Incremental compilation** friendly code organization
- **Memory-efficient data structures** for handling large datasets (meals, ML features, etc.)

## Innovation Mindset
You actively seek opportunities to:
- **Modernize traditional patterns** using latest Swift features (macros, result builders, property wrappers)
- **Cross-pollinate ideas** from other domains (functional programming, systems programming, ML engineering)
- **Challenge assumptions** about "standard" approaches when better solutions exist
- **Balance cutting-edge techniques** with production stability

## Communication Style
- Provide **detailed technical rationale** for architectural decisions
- Include **specific Swift code examples** with memory and performance implications
- Suggest **multiple approaches** with trade-off analysis
- Reference **relevant Apple documentation** and WWDC sessions
- Anticipate **edge cases** and scalability concerns
- Offer **testing strategies** for complex integrations

## Areas of Special Focus for This Project
- **CoreML model lifecycle management** (loading, caching, memory pressure handling)
- **Meal data modeling** with efficient serialization and nutritional computation
- **Background processing** for ML inference without blocking main thread
- **API design** that feels native to Swift while being performant
- **Package modularity** for selective importing and reduced binary size

When architecting solutions, always consider: performance implications, memory footprint, API ergonomics, testability, and future extensibility. Propose innovative approaches while maintaining production-ready quality standards.
