# EmbedKit PipelineKit Integration

This directory contains the comprehensive PipelineKit integration for EmbedKit, providing a command-driven, middleware-enhanced pipeline for text embedding operations with elegant operator syntax support.

## Overview

The integration provides:

- **Commands**: Structured commands for embedding operations, model management, and cache control
- **Handlers**: Specialized handlers that process commands using EmbedKit's text embedders
- **Middleware**: Cross-cutting concerns like caching, GPU acceleration, validation, and telemetry
- **Pipeline**: A fully configured pipeline that combines all components
- **Operators**: Fluent operator syntax for elegant pipeline construction
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

## Operator Syntax

PipelineKit provides elegant operator syntax for building pipelines:

### Basic Operators

```swift
// Create pipeline with operators
let pipeline = try await pipeline(for: embedHandler)
    <+ validationMiddleware      // Add middleware
    <+ cacheMiddleware          // Chain middleware
    <+ telemetryMiddleware      // Continue chaining
    .build()                    // Build final pipeline
```

### Priority Operators

```swift
// Add middleware with execution priorities
let pipeline = try await pipeline(for: handler)
    <++ middleware(validationMiddleware, priority: .validation)
    <++ middleware(authMiddleware, priority: .authentication)  
    <++ middleware(cacheMiddleware, priority: .normal)
    <++ middleware(telemetryMiddleware, priority: .monitoring)
    .build()
```

### Pipeline Composition

```swift
// Sequential composition
let composed = pipeline1 |> pipeline2  // Execute pipeline1, then pipeline2

// Parallel composition  
let parallel = pipeline1 || pipeline2  // Execute both, return first result

// Error handling
let safe = pipeline |! { error in
    await handleError(error)
}

// Conditional execution
let conditional = pipeline |? { 
    await checkCondition()
}
```

### Real-World Example

```swift
// Production-ready pipeline with operators
let productionPipeline = try await pipeline(for: embedHandler)
    // Critical middleware with priorities
    <++ middleware(validationMiddleware, priority: .validation)
    <++ middleware(rateLimitMiddleware, priority: .critical)
    
    // Performance optimization
    <+ cacheMiddleware
    <++ middleware(gpuAccelerationMiddleware, priority: .processing)
    
    // Monitoring
    <++ middleware(telemetryMiddleware, priority: .monitoring)
    <+ monitoringMiddleware
    .build()
    
// Add error handling and health checks
let safePipeline = productionPipeline 
    |! { error in await telemetry.recordError(error) }
    |? { await healthCheck.isSystemHealthy() }
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

### Using Operators

```swift
// Build pipeline with operators
let pipeline = try await pipeline(for: embedHandler)
    <++ middleware(validationMiddleware, priority: .validation)
    <+ cacheMiddleware
    <++ middleware(gpuMiddleware, priority: .processing)
    <+ telemetryMiddleware
    .build()

// Execute command
let command = EmbedTextCommand(text: "Hello, operators!")
let result = try await pipeline.execute(command, metadata: DefaultCommandMetadata())
```

### Conditional Pipeline Building

```swift
var builder = pipeline(for: handler)
    <++ middleware(validationMiddleware, priority: .validation)

if enableCache {
    builder = builder <+ cacheMiddleware
}

if enableGPU {
    builder = builder <++ middleware(gpuMiddleware, priority: .processing)
}

let pipeline = try await builder.build()
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

### Operator-Based Factory

```swift
// High-performance pipeline using operators
let pipeline = try await EmbeddingPipelineOperatorFactory.highPerformance(
    embedder: embedder,
    modelManager: modelManager
)

// Custom pipeline with operators
let customPipeline = try await pipeline(for: handler)
    <++ middleware(validationMiddleware, priority: .validation)
    <+ (enableCache ? cacheMiddleware : nil)
    <++ middleware(telemetryMiddleware, priority: .monitoring)
    .build()
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

With operators:

```swift
let safePipeline = pipeline |! { error in
    logger.error("Pipeline error: \(error)")
    await metrics.recordError(error)
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

## Advanced Patterns

### Production Pipeline Pattern

```swift
func createProductionPipeline(config: AppConfig) async throws -> any Pipeline {
    var builder = pipeline(for: createHandler(config))
        // Always validate
        <++ middleware(validationMiddleware, priority: .validation)
    
    // Add auth if required
    if config.requiresAuth {
        builder = builder <++ middleware(authMiddleware, priority: .authentication)
    }
    
    // Add rate limiting by tier
    switch config.tier {
    case .free:
        builder = builder <++ middleware(rateLimiter(10), priority: .critical)
    case .pro:
        builder = builder <++ middleware(rateLimiter(100), priority: .critical)
    case .enterprise:
        // No rate limiting
        break
    }
    
    // Performance features
    builder = builder
        <+ cacheMiddleware
        <++ middleware(gpuMiddleware, priority: .processing)
        <++ middleware(telemetryMiddleware, priority: .monitoring)
    
    // Build and add safety
    return try await builder.build()
        |! { error in await alerting.notify(error) }
        |? { await health.check() }
}
```

### Composed Pipelines

```swift
// Create specialized pipelines
let fastPipeline = try await createFastPipeline()
let accuratePipeline = try await createAccuratePipeline()

// Compose them
let hybridPipeline = fastPipeline |> accuratePipeline  // Fast first, then accurate
let racePipeline = fastPipeline || accuratePipeline    // First to complete wins
```

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

1. **Use operator syntax for clarity** - Operators make pipeline construction more readable
2. **Apply middleware priorities** - Ensure middleware executes in the correct order
3. **Model Loading**: Load models once and reuse them
4. **Cache Management**: Monitor cache hit rates and adjust size as needed
5. **Error Handling**: Always handle potential errors, especially for streaming
6. **Resource Cleanup**: Unload models and clear caches when done
7. **Monitoring**: Use telemetry to understand performance characteristics

## Running Examples

To run all integration examples:

```swift
try await PipelineIntegrationExamples.runAllExamples()
```

To run operator examples:

```swift
try await PipelineOperatorExamples.runAllExamples()
```

Individual examples can be run separately:
- `basicEmbeddingExample()`
- `batchProcessingExample()`
- `streamingExample()`
- `modelManagementExample()`
- `errorHandlingExample()`
- `telemetryExample()`
- `cacheManagementExample()`
- `basicOperatorExample()`
- `priorityOperatorExample()`
- `pipelineCompositionExample()`
- `fluentStyleExample()`

## Example Files

- `Examples.swift` - Traditional API examples
- `OperatorExamples.swift` - Comprehensive operator syntax examples
- `EmbeddingPipelineOperators.swift` - Advanced operator patterns and extensions