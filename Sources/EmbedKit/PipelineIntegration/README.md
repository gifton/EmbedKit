# EmbedKit PipelineKit Integration

This directory contains the comprehensive PipelineKit integration for EmbedKit, providing a command-driven, middleware-enhanced pipeline for text embedding operations.

## Overview

The integration provides:

- **Commands**: Structured commands for embedding operations, model management, and cache control
- **Handlers**: Specialized handlers that process commands using EmbedKit's text embedders
- **Middleware**: Cross-cutting concerns like caching, GPU acceleration, validation, and telemetry
- **Pipeline**: A fully configured pipeline that combines all components
- **Examples**: Comprehensive examples demonstrating various use cases

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Command   │────▶│  Middleware  │────▶│   Handler   │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   EmbedKit   │
                    └──────────────┘
```

## Components

### Commands

- `EmbedTextCommand`: Embed a single text
- `BatchEmbedCommand`: Embed multiple texts in batch
- `StreamEmbedCommand`: Stream embeddings for large datasets
- `LoadModelCommand`: Load an embedding model
- `SwapModelCommand`: Swap the current model
- `UnloadModelCommand`: Unload the current model
- `ClearCacheCommand`: Clear the embedding cache
- `PreloadCacheCommand`: Preload embeddings into cache

### Middleware

1. **EmbeddingCacheMiddleware**: Integrates with EmbedKit's LRU cache
2. **MetalAccelerationMiddleware**: Enables GPU acceleration via Metal
3. **EmbeddingValidationMiddleware**: Validates input texts and constraints
4. **TelemetryMiddleware**: Records metrics and events
5. **EmbeddingRateLimitMiddleware**: Controls request rates
6. **EmbeddingMonitoringMiddleware**: Monitors performance and alerts

### Pipeline Configuration

The `EmbeddingPipeline` provides a high-level API with configurable features:

```swift
let configuration = EmbeddingPipeline.Configuration(
    enableCache: true,
    enableGPUAcceleration: true,
    enableRateLimiting: true,
    enableMonitoring: true,
    maxTextLength: 10_000,
    maxBatchSize: 1000,
    requestsPerSecond: 100
)
```

## Usage Examples

### Basic Embedding

```swift
// Create pipeline
let pipeline = try await EmbeddingPipeline(
    embedder: embedder,
    modelManager: modelManager
)

// Embed text
let result = try await pipeline.embed("Hello, world!")
print("Embedding dimensions: \(result.embedding.dimensions)")
```

### Batch Processing

```swift
let texts = ["Text 1", "Text 2", "Text 3"]
let batchResult = try await pipeline.embedBatch(texts)
print("Processed \(batchResult.embeddings.count) embeddings")
```

### Streaming Large Datasets

```swift
let textSource = ArrayTextSource(largeTextArray)
let stream = try await pipeline.streamEmbeddings(from: textSource)

for try await result in stream {
    print("Embedded text \(result.index)")
}
```

### Model Management

```swift
// Load a model
let loadResult = try await pipeline.loadModel("all-MiniLM-L6-v2")

// Swap models
let swapResult = try await pipeline.swapModel(to: "all-mpnet-base-v2")

// Unload and clear cache
let unloadResult = try await pipeline.unloadModel(clearCache: true)
```

## Pipeline Factories

Pre-configured pipelines for different use cases:

```swift
// High-performance pipeline (no rate limiting, minimal monitoring)
let highPerf = try await EmbeddingPipelineFactory.highPerformance(
    embedder: embedder,
    modelManager: modelManager
)

// Balanced pipeline (all features enabled with reasonable defaults)
let balanced = try await EmbeddingPipelineFactory.balanced(
    embedder: embedder,
    modelManager: modelManager
)

// Development pipeline (extensive monitoring and logging)
let dev = try await EmbeddingPipelineFactory.development(
    embedder: embedder,
    modelManager: modelManager
)

// Minimal pipeline (only essential features)
let minimal = try await EmbeddingPipelineFactory.minimal(
    embedder: embedder,
    modelManager: modelManager
)
```

## Error Handling

The pipeline provides comprehensive error handling:

```swift
do {
    let result = try await pipeline.embed(text)
} catch let error as ValidationError {
    // Handle validation errors
} catch let error as RateLimitError {
    // Handle rate limit errors
} catch let error as EmbeddingError {
    // Handle embedding-specific errors
} catch {
    // Handle other errors
}
```

## Monitoring and Telemetry

Get pipeline statistics and telemetry data:

```swift
// Get comprehensive statistics
let stats = await pipeline.getStatistics()
print("Cache hit rate: \(stats.cacheStatistics.hitRate)")
print("Memory usage: \(stats.systemMetrics.memoryUsage)MB")

// Export telemetry data
if let telemetryData = await pipeline.getTelemetryData() {
    // Process telemetry data
}
```

## Performance Considerations

1. **Caching**: Enable caching for frequently used texts
2. **Batch Processing**: Use batch operations for multiple texts
3. **GPU Acceleration**: Enable Metal acceleration for better performance
4. **Streaming**: Use streaming for large datasets to manage memory
5. **Model Selection**: Choose appropriate models based on accuracy/speed trade-offs

## Thread Safety

All components are thread-safe and designed for concurrent usage:
- Commands and results are `Sendable`
- Handlers are actors ensuring serial execution
- Middleware properly handles concurrent requests
- Cache and telemetry systems are thread-safe

## Integration with Existing Code

The pipeline can be easily integrated into existing PipelineKit-based applications:

```swift
// Add EmbedKit handlers to existing command bus
let busBuilder = CommandBusBuilder()
await busBuilder.register(embedTextHandler, for: EmbedTextCommand.self)

// Add EmbedKit middleware to existing pipeline
let pipelineBuilder = ContextAwarePipelineBuilder(bus: commandBus)
_ = await pipelineBuilder.with(EmbeddingCacheMiddleware(cache: cache))
```

## Best Practices

1. **Model Loading**: Load models once and reuse them
2. **Cache Management**: Monitor cache hit rates and adjust size as needed
3. **Error Handling**: Always handle potential errors, especially for streaming
4. **Resource Cleanup**: Unload models and clear caches when done
5. **Monitoring**: Use telemetry to understand performance characteristics

## Running Examples

To run all integration examples:

```swift
try await PipelineIntegrationExamples.runAllExamples()
```

Individual examples can be run separately:
- `basicEmbeddingExample()`
- `batchProcessingExample()`
- `streamingExample()`
- `modelManagementExample()`
- `errorHandlingExample()`
- `telemetryExample()`
- `cacheManagementExample()`