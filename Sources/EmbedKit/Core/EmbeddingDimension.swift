//
//  EmbeddingDimension.swift
//  EmbedKit
//
//  Embedding-specific dimension types extending VectorCore's StaticDimension
//

import Foundation
import VectorCore

// MARK: - Embedding Dimension Protocol

/// Protocol marking dimensions suitable for embedding vectors.
///
/// `EmbeddingDimension` extends VectorCore's `StaticDimension` to provide
/// compile-time type safety for embedding operations. This protocol allows
/// us to constrain embedding-specific operations to appropriate dimension types
/// while leveraging all of VectorCore's optimized vector operations.
///
/// ## Design Philosophy
/// - Leverages VectorCore's existing dimension types (Dim384, Dim768, Dim1536)
/// - Provides embedding-specific semantics without duplicating infrastructure
/// - Enables compile-time dimension verification for embedding operations
/// - Supports standard embedding dimensions used in ML models
///
/// ## Example Usage
/// ```swift
/// func processEmbedding<D: EmbeddingDimension>(_ emb: Embedding<D>) {
///     // Compiler guarantees D is a valid embedding dimension
///     let similarity = emb.cosineSimilarity(to: other)
/// }
/// ```
public protocol EmbeddingDimension: StaticDimension {}

// MARK: - Standard Embedding Dimensions

/// 3-dimensional embedding space for testing and lightweight workflows.
///
/// Primarily used in unit tests and mock backends where small vectors
/// simplify correctness checks and speed up execution. Not intended for
/// production usage.
public struct Dim3: EmbeddingDimension {
    public static let value = 3
    public typealias Storage = DimensionStorage<Dim3, Float>
}

/// 384-dimensional embedding space.
///
/// Common dimension for:
/// - Sentence-BERT MiniLM models (all-MiniLM-L6-v2, all-MiniLM-L12-v2)
/// - Compact multilingual models
/// - Mobile-optimized embedding models
/// - Fast similarity search applications
///
/// ## Performance Characteristics
/// - Memory: 1,536 bytes per embedding (Float32)
/// - SIMD optimization: Full AVX/NEON vectorization
/// - Typical use case: High-throughput semantic search with 100M+ vectors
///
/// ## Mathematical Properties
/// - Sufficient expressiveness for most sentence-level semantic tasks
/// - Good balance between accuracy and computational efficiency
/// - Recommended minimum dimension for production sentence embeddings
public struct Dim384: EmbeddingDimension {
    public static let value = 384
    public typealias Storage = DimensionStorage<Dim384, Float>
}

/// 768-dimensional embedding space.
///
/// Standard dimension for:
/// - BERT base models (bert-base-uncased, bert-base-multilingual)
/// - DistilBERT and many transformer variants
/// - Document embeddings
/// - Standard NLP embedding tasks
///
/// ## Performance Characteristics
/// - Memory: 3,072 bytes per embedding (Float32)
/// - SIMD optimization: Optimized kernels in VectorCore (Vector768Optimized)
/// - Typical use case: Standard semantic search and text classification
///
/// ## Mathematical Properties
/// - Industry standard for transformer-based embeddings
/// - Well-balanced expressiveness for complex semantic relationships
/// - Extensive pre-trained model availability
///
/// - Note: VectorCore provides specialized `Vector768Optimized` for maximum performance
public struct Dim768: EmbeddingDimension {
    public static let value = 768
    public typealias Storage = DimensionStorage<Dim768, Float>
}

/// 1536-dimensional embedding space.
///
/// Large dimension for:
/// - OpenAI text-embedding-ada-002 and text-embedding-3-small
/// - GPT-style embeddings
/// - High-accuracy document retrieval
/// - Cross-modal embeddings (text-image)
///
/// ## Performance Characteristics
/// - Memory: 6,144 bytes per embedding (Float32)
/// - SIMD optimization: Optimized kernels in VectorCore (Vector1536Optimized)
/// - Typical use case: High-accuracy retrieval, large-scale RAG systems
///
/// ## Mathematical Properties
/// - Maximum expressiveness for nuanced semantic distinctions
/// - Captures fine-grained contextual information
/// - Suitable for multi-domain and cross-lingual applications
///
/// ## OpenAI Compatibility
/// - Direct compatibility with OpenAI embedding API responses
/// - No dimension reduction required for ada-002 embeddings
///
/// - Note: VectorCore provides specialized `Vector1536Optimized` for maximum performance
public struct Dim1536: EmbeddingDimension {
    public static let value = 1536
    public typealias Storage = DimensionStorage<Dim1536, Float>
}

// MARK: - Dimension Utilities

extension EmbeddingDimension {
    /// Human-readable name for the dimension.
    public static var dimensionName: String {
        "Dim\(value)"
    }

    /// Memory footprint per embedding in bytes (Float32).
    public static var bytesPerEmbedding: Int {
        value * MemoryLayout<Float>.stride
    }

    /// Approximate memory footprint per 1 million embeddings in MB.
    public static var megabytesPerMillion: Double {
        Double(bytesPerEmbedding * 1_000_000) / (1024 * 1024)
    }
}

// MARK: - Compile-Time Dimension Validation

/// Compile-time assertion that a dimension is suitable for embeddings.
///
/// This function provides compile-time verification that a dimension type
/// conforms to `EmbeddingDimension`. Use it to document dimension requirements
/// in generic contexts.
///
/// - Parameter _: The dimension type to validate
/// - Returns: The dimension value if valid
@inlinable
public func validateEmbeddingDimension<D: EmbeddingDimension>(_: D.Type = D.self) -> Int {
    D.value
}

// MARK: - Documentation Examples

/*
 ## Usage Examples

 ### Type-Safe Embedding Creation
 ```swift
 // Compile-time dimension verification
 let embedding = Embedding<Dim768>.zeros()

 // Type mismatch caught at compile time:
 // let wrong: Embedding<Dim768> = Embedding<Dim384>.zeros()  // ❌ Compile error
 ```

 ### Memory Planning
 ```swift
 print("Storing 10M Dim768 embeddings requires:")
 print("\(Dim768.megabytesPerMillion * 10) MB")
 // Output: ~29,296 MB (≈28.6 GB)
 ```

 ### Generic Embedding Operations
 ```swift
 func computeSimilarity<D: EmbeddingDimension>(
     query: Embedding<D>,
     candidates: [Embedding<D>]
 ) -> [Float] {
     candidates.map { query.cosineSimilarity(to: $0) }
 }
 ```

 ### Dimension Selection
 ```swift
 // Choose dimension based on use case:

 // High throughput, mobile deployment:
 typealias MobileEmbedding = Embedding<Dim384>

 // Standard NLP tasks, good accuracy/speed balance:
 typealias StandardEmbedding = Embedding<Dim768>

 // Maximum accuracy, OpenAI compatibility:
 typealias PremiumEmbedding = Embedding<Dim1536>
 ```
 */
