# EmbedKit Examples

This directory contains example code demonstrating various features and usage patterns of EmbedKit.

## Basic Examples

### BasicUsage.swift
Simple examples showing the fundamentals of using EmbedKit for text embeddings.

### CosineSimilarityExample.swift
Demonstrates how to calculate cosine similarity between embeddings for semantic similarity tasks.

### ExpressiveLoggingExample.swift
Shows how to use EmbedKit's expressive logging system for debugging and monitoring.

## Pipeline Integration Examples

### PipelineIntegrationExample.swift
Comprehensive example showing how to integrate EmbedKit with PipelineKit for complex workflows.

### PipelineIntegrationUsage.swift
Practical usage patterns for pipeline integration in real-world scenarios.

### PipelineOperatorUsage.swift
Advanced examples using custom operators for elegant pipeline construction.

## Running Examples

To run these examples in your own project:

1. Add EmbedKit as a dependency in your `Package.swift`
2. Import the relevant example file or copy the code
3. Ensure you have the required model files (see MODELS_README.md)
4. Run the example functions

Example:
```swift
import EmbedKit

// Run basic usage examples
try await BasicUsageExamples.demonstrateBasicEmbedding()

// Run pipeline examples  
try await PipelineIntegrationExample.runExample()
```

## Note on Models

These examples require CoreML model files to be present. See the [MODELS_README.md](../../../MODELS_README.md) for instructions on obtaining the necessary models.