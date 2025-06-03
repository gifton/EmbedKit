# EmbedKit Expressive Logging Guide 🎨

## Overview

The ExpressiveLogger provides rich, emoji-enhanced logging with specialized methods for different operations. It makes logs more readable and provides better context for debugging and monitoring.

## Quick Start

```swift
// Create a logger for your component
private let logger = EmbedKitLogger.embeddings()

// Or create a custom logger
private let logger = EmbedKitLogger.custom("MyComponent")
```

## Logging Methods

### Basic Logging

```swift
// Debug information
logger.debug("Detailed trace info", context: "initialization")

// General information  
logger.info("Configuration loaded")

// Success messages
logger.success("Model loaded successfully")

// Warnings
logger.warning("Cache miss rate high")

// Errors with exception
logger.error("Failed to load model", error: error)

// Critical issues
logger.critical("System failure", error: error)
```

### Specialized Logging

#### Performance Logging ⚡
```swift
logger.performance("Batch processing", duration: 1.5, throughput: 1000.0)
// Output: ⚡ [Embeddings] Batch processing completed in 1.50s • 1.0K ops/sec
```

#### Memory Logging 💾
```swift
logger.memory("Model loaded", bytes: 512_000_000, peak: 1_024_000_000)
// Output: 💾 [Embeddings] Model loaded • Memory: 512.0MB (Peak: 1.0GB)
```

#### Model Operations 🤖
```swift
logger.model("Loading embedder", modelId: "text-embedding-3", version: "1.2.0")
// Output: 🤖 [ModelMgmt] Loading embedder • Model: text-embedding-3 v1.2.0
```

#### Cache Operations 📦
```swift
logger.cache("Cache hit", hitRate: 0.92, size: 1000)
// Output: 📦 [Cache] Cache hit • Hit Rate: 92.0% • Size: 1000 items
```

#### Security Operations 🔒
```swift
logger.security("Signature verified", status: "trusted")
// Output: 🔒 [Security] Signature verified • Status: trusted
```

#### Operation Lifecycle

```swift
// Starting an operation
logger.start("Batch embedding", details: "1000 documents")
// Output: 🚀 [Embeddings] Starting Batch embedding • 1000 documents

// Processing with progress
logger.processing("Document batch", progress: 0.75)
// Output: ⚙️ [Embeddings] Processing Document batch • ███████░░░ 75%

// Thinking/Analysis
logger.thinking("optimization strategy")
// Output: 🤔 [Embeddings] Analyzing optimization strategy

// Completion
logger.complete("Batch embedding", result: "1000/1000 successful")
// Output: 🎉 [Embeddings] Completed Batch embedding • 1000/1000 successful
```

## Advanced Features

### Timed Operations

Automatically measure and log operation duration:

```swift
let result = try await logger.timed("Complex operation") {
    // Your async operation here
    return await performComplexTask()
}
// Automatically logs start, duration, and any errors
```

### Structured Context

```swift
let context = LogContext(
    requestId: "req-123",
    userId: "user-456",
    modelId: "model-789",
    operation: "batch_embed"
)

logger.log(level: .info, "Processing request", context: context)
// Output: ℹ️ [Embeddings] {req:req-123|user:user-456|model:model-789} Processing request
```

## Formatting Features

### Duration Formatting
- Microseconds: `125μs`
- Milliseconds: `45.2ms`
- Seconds: `1.35s`
- Minutes: `2m 15.3s`

### Throughput Formatting
- Low: `0.75 ops/sec`
- Medium: `156.3 ops/sec`
- High: `45.2K ops/sec`
- Very High: `1.3M ops/sec`

### Memory Formatting
- Automatic unit selection (B, KB, MB, GB)
- Binary format (1024-based)

### Progress Bars
- Visual progress: `███████░░░ 75%`
- 10-segment bar with percentage

## Best Practices

1. **Use specialized methods** - Choose the right method for your operation type
2. **Provide context** - Include relevant IDs and metrics
3. **Log lifecycle** - Use start/complete pairs for long operations
4. **Include metrics** - Add performance data when available
5. **Be consistent** - Use the same logger instance throughout a component

## Integration Example

```swift
public actor MyComponent {
    private let logger = EmbedKitLogger.custom("MyComponent")
    
    public func process(data: Data) async throws -> Result {
        return try await logger.timed("Data processing") {
            logger.start("Processing", details: "\(data.count) bytes")
            
            // Process data
            let result = try await performProcessing(data)
            
            logger.complete("Processing", result: "Generated \(result.count) items")
            logger.performance("Processing", duration: 0.5, throughput: Double(result.count) / 0.5)
            
            return result
        }
    }
}
```

## Logger Categories

Pre-configured loggers for different components:

- `EmbedKitLogger.embeddings()` - For embedding operations
- `EmbedKitLogger.metal()` - For GPU acceleration
- `EmbedKitLogger.cache()` - For caching operations
- `EmbedKitLogger.streaming()` - For streaming operations
- `EmbedKitLogger.modelManagement()` - For model management
- `EmbedKitLogger.security()` - For security operations
- `EmbedKitLogger.telemetry()` - For telemetry
- `EmbedKitLogger.benchmarks()` - For performance benchmarks

## Emoji Reference

| Emoji | Meaning | Usage |
|-------|---------|-------|
| 🔍 | Debug | Detailed debugging information |
| ℹ️ | Info | General information |
| ✅ | Success | Successful operations |
| ⚠️ | Warning | Warning conditions |
| ❌ | Error | Error conditions |
| 🚨 | Critical | Critical failures |
| ⚡ | Performance | Performance metrics |
| 💾 | Memory | Memory usage |
| 🌐 | Network | Network operations |
| 🤖 | Model | Model operations |
| 📦 | Cache | Cache operations |
| 🔒 | Security | Security operations |
| 🚀 | Start | Operation start |
| 🎉 | Complete | Operation completion |
| ⚙️ | Processing | Active processing |
| 🤔 | Thinking | Analysis/computation |