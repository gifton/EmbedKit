# Changelog

All notable changes to EmbedKit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of EmbedKit
- Core ML backend for text embeddings
- Metal GPU acceleration for 10x+ faster batch processing
- Intelligent LRU caching with memory pressure handling
- PipelineKit integration with commands and handlers
- Actor-based concurrency for thread-safe operations
- Batch processing support for efficient multi-text embedding
- Performance benchmarking utilities
- Comprehensive test suite

### Features
- `TextEmbedder` protocol for embedding generation
- `CoreMLTextEmbedder` implementation with GPU acceleration
- `MetalAccelerator` for GPU-powered operations
- `LRUCache` with automatic memory management
- `EmbeddingVector` type with similarity calculations
- Streaming support for large document collections
- Configurable batch sizes and performance options

### Performance
- 10x+ faster batch processing with Metal acceleration
- 100x+ faster repeated queries with smart caching
- Memory-efficient with automatic cache eviction

## [1.0.0] - TBD

Initial public release.

[Unreleased]: https://github.com/yourusername/EmbedKit/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yourusername/EmbedKit/releases/tag/v1.0.0