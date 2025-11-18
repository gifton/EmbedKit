//
//  DynamicEmbedding.swift
//  EmbedKit
//
//  Runtime-typed embeddings supporting multiple dimensions
//

import Foundation
import VectorCore

// MARK: - Dynamic Embedding

/// A runtime-typed embedding that can represent different dimensions.
///
/// `DynamicEmbedding` provides a type-erased wrapper for embeddings of different
/// dimensions, allowing heterogeneous collections and runtime dimension handling.
/// While less efficient than compile-time typed `Embedding<D>`, it's essential
/// for scenarios where dimensions are determined at runtime.
///
/// ## Design Philosophy
/// - **Runtime flexibility**: Support embeddings of varying dimensions in the same collection
/// - **Type safety**: Validates dimension compatibility at operation boundaries
/// - **API compatibility**: Provides similar interface to `Embedding<D>`
/// - **Conversion support**: Seamless conversion to/from compile-time typed embeddings
///
/// ## Use Cases
/// - Heterogeneous embedding collections from multiple models
/// - Dimension-agnostic APIs and protocols
/// - Dynamic model selection based on runtime configuration
/// - Serialization/deserialization where dimension is data-dependent
///
/// ## Performance Considerations
/// - Runtime dimension checks add overhead vs compile-time verification
/// - No SIMD specialization (uses generic paths)
/// - Heap allocation for storage (vs potential stack optimization for small dims)
/// - Prefer `Embedding<D>` when dimension is known at compile time
///
/// ## Example Usage
/// ```swift
/// // Heterogeneous collection
/// let embeddings: [DynamicEmbedding] = [
///     .dim384(Embedding<Dim384>.random()),
///     .dim768(Embedding<Dim768>.random()),
///     .dim1536(Embedding<Dim1536>.random())
/// ]
///
/// // Process each embedding
/// for emb in embeddings {
///     print("Dimension: \(emb.dimensions), Magnitude: \(emb.magnitude)")
/// }
/// ```
@frozen
public enum DynamicEmbedding: Sendable {

    // MARK: - Cases

    /// 384-dimensional embedding (Sentence-BERT MiniLM).
    case dim384(Embedding384)

    /// 768-dimensional embedding (BERT base).
    case dim768(Embedding768)

    /// 1536-dimensional embedding (OpenAI ada-002).
    case dim1536(Embedding1536)
    /// 3-dimensional embedding (testing/mocks)
    case dim3(Embedding<Dim3>)

    // MARK: - Properties

    /// The dimension of this embedding.
    ///
    /// Determined at runtime based on the stored embedding type.
    @inlinable
    public var dimensions: Int {
        switch self {
        case .dim3: return 3
        case .dim384: return 384
        case .dim768: return 768
        case .dim1536: return 1536
        }
    }

    /// The L2 norm (magnitude) of the embedding vector.
    @inlinable
    public var magnitude: Float {
        switch self {
        case .dim3(let emb): return emb.magnitude
        case .dim384(let emb): return emb.magnitude
        case .dim768(let emb): return emb.magnitude
        case .dim1536(let emb): return emb.magnitude
        }
    }

    /// The squared L2 norm.
    @inlinable
    public var magnitudeSquared: Float {
        switch self {
        case .dim3(let emb): return emb.magnitudeSquared
        case .dim384(let emb): return emb.magnitudeSquared
        case .dim768(let emb): return emb.magnitudeSquared
        case .dim1536(let emb): return emb.magnitudeSquared
        }
    }

    /// Check if all values are finite.
    @inlinable
    public var isFinite: Bool {
        switch self {
        case .dim3(let emb): return emb.isFinite
        case .dim384(let emb): return emb.isFinite
        case .dim768(let emb): return emb.isFinite
        case .dim1536(let emb): return emb.isFinite
        }
    }

    /// Check if this is the zero vector.
    @inlinable
    public var isZero: Bool {
        switch self {
        case .dim3(let emb): return emb.isZero
        case .dim384(let emb): return emb.isZero
        case .dim768(let emb): return emb.isZero
        case .dim1536(let emb): return emb.isZero
        }
    }

    // MARK: - Initialization

    /// Create a dynamic embedding from an array with dimension inference.
    ///
    /// The dimension type is selected based on the array length.
    ///
    /// - Parameter values: Array of Float values
    /// - Throws: `EmbeddingError.unsupportedDimension` if array length doesn't match supported dimensions
    ///
    /// ## Example
    /// ```swift
    /// let emb384 = try DynamicEmbedding(values: Array(repeating: 0, count: 384))
    /// let emb768 = try DynamicEmbedding(values: Array(repeating: 0, count: 768))
    /// ```
    public init(values: [Float]) throws {
        switch values.count {
        case 3:
            let emb = try Embedding<Dim3>(values)
            self = .dim3(emb)
        case 384:
            let emb = try Embedding<Dim384>(values)
            self = .dim384(emb)
        case 768:
            let emb = try Embedding<Dim768>(values)
            self = .dim768(emb)
        case 1536:
            let emb = try Embedding<Dim1536>(values)
            self = .dim1536(emb)
        default:
            throw EmbeddingError.unsupportedDimension(values.count)
        }
    }

    // MARK: - Factory Methods

    /// Create a zero embedding of the specified dimension.
    ///
    /// - Parameter dimension: The dimension (must be 384, 768, or 1536)
    /// - Throws: `EmbeddingError.unsupportedDimension` if dimension is not supported
    public static func zeros(dimension: Int) throws -> Self {
        switch dimension {
        case 384: return .dim384(Embedding384.zeros())
        case 768: return .dim768(Embedding768.zeros())
        case 1536: return .dim1536(Embedding1536.zeros())
        default: throw EmbeddingError.unsupportedDimension(dimension)
        }
    }

    /// Create an embedding with all ones of the specified dimension.
    ///
    /// - Parameter dimension: The dimension (must be 384, 768, or 1536)
    /// - Throws: `EmbeddingError.unsupportedDimension` if dimension is not supported
    public static func ones(dimension: Int) throws -> Self {
        switch dimension {
        case 384: return .dim384(Embedding384.ones())
        case 768: return .dim768(Embedding768.ones())
        case 1536: return .dim1536(Embedding1536.ones())
        default: throw EmbeddingError.unsupportedDimension(dimension)
        }
    }

    /// Create a random embedding of the specified dimension.
    ///
    /// - Parameters:
    ///   - dimension: The dimension (must be 384, 768, or 1536)
    ///   - range: The range of random values (default: 0...1)
    /// - Throws: `EmbeddingError.unsupportedDimension` if dimension is not supported
    public static func random(dimension: Int, in range: ClosedRange<Float> = 0...1) throws -> Self {
        switch dimension {
        case 384: return .dim384(Embedding384.random(in: range))
        case 768: return .dim768(Embedding768.random(in: range))
        case 1536: return .dim1536(Embedding1536.random(in: range))
        default: throw EmbeddingError.unsupportedDimension(dimension)
        }
    }

    // MARK: - Normalization

    /// Normalize the embedding to unit length.
    ///
    /// - Returns: A new normalized embedding
    /// - Throws: `VectorError.invalidOperation` if the vector is zero
    public func normalized() throws -> Self {
        switch self {
        case .dim3(let emb):
            let normalized = try emb.normalized().get()
            return .dim3(normalized)
        case .dim384(let emb):
            let normalized = try emb.normalized().get()
            return .dim384(normalized)
        case .dim768(let emb):
            let normalized = try emb.normalized().get()
            return .dim768(normalized)
        case .dim1536(let emb):
            let normalized = try emb.normalized().get()
            return .dim1536(normalized)
        }
    }

    // MARK: - Similarity and Distance Metrics

    /// Compute cosine similarity with another embedding.
    ///
    /// - Parameter other: The embedding to compare with
    /// - Throws: `EmbeddingError.dimensionMismatch` if dimensions don't match
    /// - Returns: Similarity in range [-1, 1]
    public func cosineSimilarity(to other: Self) throws -> Float {
        guard dimensions == other.dimensions else {
            throw EmbeddingError.dimensionMismatch(expected: dimensions, actual: other.dimensions)
        }

        switch (self, other) {
        case (.dim3(let a), .dim3(let b)):
            return a.cosineSimilarity(to: b)
        case (.dim384(let a), .dim384(let b)):
            return a.cosineSimilarity(to: b)
        case (.dim768(let a), .dim768(let b)):
            return a.cosineSimilarity(to: b)
        case (.dim1536(let a), .dim1536(let b)):
            return a.cosineSimilarity(to: b)
        default:
            // This should never happen due to dimension check above
            throw EmbeddingError.dimensionMismatch(expected: dimensions, actual: other.dimensions)
        }
    }

    /// Compute cosine distance (1 - cosine similarity).
    ///
    /// - Parameter other: The embedding to compare with
    /// - Throws: `EmbeddingError.dimensionMismatch` if dimensions don't match
    /// - Returns: Distance in range [0, 2]
    public func cosineDistance(to other: Self) throws -> Float {
        let similarity = try cosineSimilarity(to: other)
        return 1.0 - similarity
    }

    /// Compute Euclidean (L2) distance.
    ///
    /// - Parameter other: The embedding to compare with
    /// - Throws: `EmbeddingError.dimensionMismatch` if dimensions don't match
    /// - Returns: Distance ≥ 0
    public func euclideanDistance(to other: Self) throws -> Float {
        guard dimensions == other.dimensions else {
            throw EmbeddingError.dimensionMismatch(expected: dimensions, actual: other.dimensions)
        }

        switch (self, other) {
        case (.dim3(let a), .dim3(let b)):
            return a.euclideanDistance(to: b)
        case (.dim384(let a), .dim384(let b)):
            return a.euclideanDistance(to: b)
        case (.dim768(let a), .dim768(let b)):
            return a.euclideanDistance(to: b)
        case (.dim1536(let a), .dim1536(let b)):
            return a.euclideanDistance(to: b)
        default:
            throw EmbeddingError.dimensionMismatch(expected: dimensions, actual: other.dimensions)
        }
    }

    /// Compute dot product.
    ///
    /// - Parameter other: The embedding to compute dot product with
    /// - Throws: `EmbeddingError.dimensionMismatch` if dimensions don't match
    /// - Returns: The dot product value
    public func dotProduct(_ other: Self) throws -> Float {
        guard dimensions == other.dimensions else {
            throw EmbeddingError.dimensionMismatch(expected: dimensions, actual: other.dimensions)
        }

        switch (self, other) {
        case (.dim3(let a), .dim3(let b)):
            return a.dotProduct(b)
        case (.dim384(let a), .dim384(let b)):
            return a.dotProduct(b)
        case (.dim768(let a), .dim768(let b)):
            return a.dotProduct(b)
        case (.dim1536(let a), .dim1536(let b)):
            return a.dotProduct(b)
        default:
            throw EmbeddingError.dimensionMismatch(expected: dimensions, actual: other.dimensions)
        }
    }

    // MARK: - Typed Accessors

    /// Access as 384-dimensional embedding.
    ///
    /// - Returns: The underlying embedding if dimension matches
    public var as384: Embedding384? {
        guard case .dim384(let emb) = self else { return nil }
        return emb
    }

    /// Access as 768-dimensional embedding.
    ///
    /// - Returns: The underlying embedding if dimension matches
    public var as768: Embedding768? {
        guard case .dim768(let emb) = self else { return nil }
        return emb
    }

    /// Access as 1536-dimensional embedding.
    ///
    /// - Returns: The underlying embedding if dimension matches
    public var as1536: Embedding1536? {
        guard case .dim1536(let emb) = self else { return nil }
        return emb
    }

    // MARK: - Array Conversion

    /// Convert embedding to array of floats.
    public func toArray() -> [Float] {
        switch self {
        case .dim3(let emb): return emb.toArray()
        case .dim384(let emb): return emb.toArray()
        case .dim768(let emb): return emb.toArray()
        case .dim1536(let emb): return emb.toArray()
        }
    }

    // MARK: - Dimension Testing

    /// Check if this embedding has the specified dimension.
    ///
    /// - Parameter dimension: The dimension to check
    /// - Returns: true if dimensions match
    public func hasDimension(_ dimension: Int) -> Bool {
        dimensions == dimension
    }

    /// Supported embedding dimensions.
    public static var supportedDimensions: [Int] {
        [384, 768, 1536]
    }
}

// MARK: - Equatable & Hashable

extension DynamicEmbedding: Equatable {
    public static func == (lhs: Self, rhs: Self) -> Bool {
        switch (lhs, rhs) {
        case (.dim3(let a), .dim3(let b)): return a == b
        case (.dim384(let a), .dim384(let b)): return a == b
        case (.dim768(let a), .dim768(let b)): return a == b
        case (.dim1536(let a), .dim1536(let b)): return a == b
        default: return false
        }
    }
}

extension DynamicEmbedding: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(dimensions)
        switch self {
        case .dim3(let emb): hasher.combine(emb)
        case .dim384(let emb): hasher.combine(emb)
        case .dim768(let emb): hasher.combine(emb)
        case .dim1536(let emb): hasher.combine(emb)
        }
    }
}

// MARK: - Codable

extension DynamicEmbedding: Codable {
    private enum CodingKeys: String, CodingKey {
        case dimension
        case values
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let dimension = try container.decode(Int.self, forKey: .dimension)
        let values = try container.decode([Float].self, forKey: .values)

        guard values.count == dimension else {
            throw DecodingError.dataCorruptedError(
                forKey: .values,
                in: container,
                debugDescription: "Values count (\(values.count)) doesn't match dimension (\(dimension))"
            )
        }

        try self.init(values: values)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(dimensions, forKey: .dimension)
        try container.encode(toArray(), forKey: .values)
    }
}

// MARK: - CustomDebugStringConvertible

extension DynamicEmbedding: CustomDebugStringConvertible {
    public var debugDescription: String {
        switch self {
        case .dim3(let emb): return "DynamicEmbedding(\(emb.debugDescription))"
        case .dim384(let emb): return "DynamicEmbedding(\(emb.debugDescription))"
        case .dim768(let emb): return "DynamicEmbedding(\(emb.debugDescription))"
        case .dim1536(let emb): return "DynamicEmbedding(\(emb.debugDescription))"
        }
    }
}

// MARK: - Embedding Error

/// Errors specific to embedding operations.
public enum EmbeddingError: Error, Sendable {
    /// The dimension is not supported by DynamicEmbedding.
    case unsupportedDimension(Int)

    /// Dimension mismatch between embeddings.
    case dimensionMismatch(expected: Int, actual: Int)

    /// Invalid embedding data.
    case invalidData(String)
}

extension EmbeddingError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .unsupportedDimension(let dim):
            return "Unsupported embedding dimension: \(dim). Supported: \(DynamicEmbedding.supportedDimensions.map(String.init).joined(separator: ", "))"
        case .dimensionMismatch(let expected, let actual):
            return "Embedding dimension mismatch: expected \(expected), got \(actual)"
        case .invalidData(let message):
            return "Invalid embedding data: \(message)"
        }
    }
}

// MARK: - Documentation Examples

/*
 ## Usage Examples

 ### Heterogeneous Collections
 ```swift
 // Store embeddings of different dimensions
 var embeddings: [DynamicEmbedding] = []

 // Add different dimension embeddings
 embeddings.append(.dim384(Embedding384.random()))
 embeddings.append(.dim768(Embedding768.random()))
 embeddings.append(.dim1536(Embedding1536.random()))

 // Process each
 for emb in embeddings {
     print("Dimension: \(emb.dimensions)")
     print("Magnitude: \(emb.magnitude)")
 }
 ```

 ### Runtime Dimension Selection
 ```swift
 func createEmbedding(forModel model: String) throws -> DynamicEmbedding {
     switch model {
     case "minilm":
         return .dim384(Embedding384.zeros())
     case "bert":
         return .dim768(Embedding768.zeros())
     case "openai":
         return .dim1536(Embedding1536.zeros())
     default:
         throw EmbeddingError.invalidData("Unknown model: \(model)")
     }
 }
 ```

 ### Safe Operations with Runtime Checks
 ```swift
 let emb1 = try DynamicEmbedding.random(dimension: 768)
 let emb2 = try DynamicEmbedding.random(dimension: 768)

 // Safe similarity computation with runtime validation
 let similarity = try emb1.cosineSimilarity(to: emb2)

 // Dimension mismatch is caught at runtime
 let emb3 = try DynamicEmbedding.random(dimension: 384)
 let invalid = try emb1.cosineSimilarity(to: emb3)  // ❌ Throws EmbeddingError
 ```

 ### Conversion to Typed Embeddings
 ```swift
 let dynamic = try DynamicEmbedding.random(dimension: 768)

 // Type-safe conversion
 if let typed = dynamic.as768 {
     // Now have compile-time type safety
     let normalized = try typed.normalized().get()
     processEmbedding(normalized)  // Function requiring Embedding<Dim768>
 }
 ```

 ### Serialization
 ```swift
 // Encode
 let embedding = try DynamicEmbedding.random(dimension: 768)
 let data = try JSONEncoder().encode(embedding)

 // Decode (dimension determined from data)
 let decoded = try JSONDecoder().decode(DynamicEmbedding.self, from: data)
 print("Loaded \(decoded.dimensions)-dimensional embedding")
 ```
 */
