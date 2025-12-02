// EmbedKit - Tensor Result Extractor
//
// Helpers for extracting results from GPU buffers into Swift types.
// Provides type-safe extraction with shape validation.
//
// Metal 4.0 (iOS 26+ / macOS 26+)

import Foundation

#if canImport(Metal)
@preconcurrency import Metal

// MARK: - Tensor Result Extractor

/// Utility for extracting results from managed tensors into Swift types.
///
/// `TensorResultExtractor` provides type-safe methods for reading GPU buffer
/// contents and converting them to Swift arrays, with shape validation and
/// error handling.
///
/// ## Usage
/// ```swift
/// let extractor = TensorResultExtractor()
///
/// // Extract as 2D array
/// let embeddings: [[Float]] = try extractor.extract2D(
///     from: tensor,
///     rows: 32,
///     columns: 384
/// )
///
/// // Extract as flat array
/// let flat: [Float] = try extractor.extractFlat(from: tensor)
/// ```
public struct TensorResultExtractor: Sendable {

    // MARK: - Initialization

    public init() {}

    // MARK: - Flat Extraction

    /// Extract tensor contents as a flat Float array.
    ///
    /// - Parameter tensor: The tensor to extract from
    /// - Returns: Flat array of all tensor values
    public func extractFlat(from tensor: ManagedTensor) -> [Float] {
        let count = tensor.buffer.length / MemoryLayout<Float>.stride
        let ptr = tensor.buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    /// Extract a portion of tensor contents as a flat array.
    ///
    /// - Parameters:
    ///   - tensor: The tensor to extract from
    ///   - offset: Starting offset in elements
    ///   - count: Number of elements to extract
    /// - Returns: Flat array of extracted values
    /// - Throws: `EmbedKitError` if range is invalid
    public func extractFlat(
        from tensor: ManagedTensor,
        offset: Int,
        count: Int
    ) throws -> [Float] {
        let totalCount = tensor.buffer.length / MemoryLayout<Float>.stride

        guard offset >= 0, count >= 0, offset + count <= totalCount else {
            throw EmbedKitError.invalidConfiguration(
                "Invalid extraction range: offset=\(offset), count=\(count), total=\(totalCount)"
            )
        }

        let ptr = tensor.buffer.contents().bindMemory(to: Float.self, capacity: totalCount)
        return Array(UnsafeBufferPointer(start: ptr + offset, count: count))
    }

    // MARK: - 2D Extraction

    /// Extract tensor contents as a 2D array.
    ///
    /// - Parameters:
    ///   - tensor: The tensor to extract from
    ///   - rows: Number of rows
    ///   - columns: Number of columns
    /// - Returns: 2D array [rows][columns]
    /// - Throws: `EmbedKitError` if shape doesn't match buffer size
    public func extract2D(
        from tensor: ManagedTensor,
        rows: Int,
        columns: Int
    ) throws -> [[Float]] {
        let expectedCount = rows * columns
        let actualCount = tensor.buffer.length / MemoryLayout<Float>.stride

        guard expectedCount <= actualCount else {
            throw EmbedKitError.invalidConfiguration(
                "Shape mismatch: expected \(expectedCount) elements (\(rows)x\(columns)), buffer has \(actualCount)"
            )
        }

        let ptr = tensor.buffer.contents().bindMemory(to: Float.self, capacity: actualCount)

        var result: [[Float]] = []
        result.reserveCapacity(rows)

        for row in 0..<rows {
            let rowStart = row * columns
            result.append(Array(UnsafeBufferPointer(start: ptr + rowStart, count: columns)))
        }

        return result
    }

    /// Extract embeddings from a tensor using its shape information.
    ///
    /// - Parameter tensor: The tensor to extract from (must have embedding shape)
    /// - Returns: 2D array of embeddings [batchSize][dimensions]
    /// - Throws: `EmbedKitError` if tensor shape is incompatible
    public func extractEmbeddings(from tensor: ManagedTensor) throws -> [[Float]] {
        switch tensor.shape {
        case .embedding(let batchSize, let dimensions):
            return try extract2D(from: tensor, rows: batchSize, columns: dimensions)

        case .tokenEmbedding, .similarityMatrix, .buffer:
            throw EmbedKitError.invalidConfiguration(
                "Expected embedding shape, got \(tensor.shape)"
            )
        }
    }

    // MARK: - 3D Extraction

    /// Extract tensor contents as a 3D array.
    ///
    /// - Parameters:
    ///   - tensor: The tensor to extract from
    ///   - dim0: First dimension size
    ///   - dim1: Second dimension size
    ///   - dim2: Third dimension size
    /// - Returns: 3D array [dim0][dim1][dim2]
    /// - Throws: `EmbedKitError` if shape doesn't match buffer size
    public func extract3D(
        from tensor: ManagedTensor,
        dim0: Int,
        dim1: Int,
        dim2: Int
    ) throws -> [[[Float]]] {
        let expectedCount = dim0 * dim1 * dim2
        let actualCount = tensor.buffer.length / MemoryLayout<Float>.stride

        guard expectedCount <= actualCount else {
            throw EmbedKitError.invalidConfiguration(
                "Shape mismatch: expected \(expectedCount) elements, buffer has \(actualCount)"
            )
        }

        let ptr = tensor.buffer.contents().bindMemory(to: Float.self, capacity: actualCount)

        var result: [[[Float]]] = []
        result.reserveCapacity(dim0)

        for i in 0..<dim0 {
            var slice2D: [[Float]] = []
            slice2D.reserveCapacity(dim1)

            for j in 0..<dim1 {
                let start = (i * dim1 + j) * dim2
                slice2D.append(Array(UnsafeBufferPointer(start: ptr + start, count: dim2)))
            }
            result.append(slice2D)
        }

        return result
    }

    /// Extract token embeddings from a tensor using its shape information.
    ///
    /// - Parameter tensor: The tensor to extract from (must have tokenEmbedding shape)
    /// - Returns: 3D array [batchSize][sequenceLength][dimensions]
    /// - Throws: `EmbedKitError` if tensor shape is incompatible
    public func extractTokenEmbeddings(from tensor: ManagedTensor) throws -> [[[Float]]] {
        switch tensor.shape {
        case .tokenEmbedding(let batchSize, let sequenceLength, let dimensions):
            return try extract3D(from: tensor, dim0: batchSize, dim1: sequenceLength, dim2: dimensions)

        case .embedding, .similarityMatrix, .buffer:
            throw EmbedKitError.invalidConfiguration(
                "Expected tokenEmbedding shape, got \(tensor.shape)"
            )
        }
    }

    // MARK: - Similarity Matrix Extraction

    /// Extract a similarity matrix from a tensor.
    ///
    /// - Parameter tensor: The tensor to extract from (must have similarityMatrix shape)
    /// - Returns: 2D similarity matrix [queryCount][keyCount]
    /// - Throws: `EmbedKitError` if tensor shape is incompatible
    public func extractSimilarityMatrix(from tensor: ManagedTensor) throws -> [[Float]] {
        switch tensor.shape {
        case .similarityMatrix(let queryCount, let keyCount):
            return try extract2D(from: tensor, rows: queryCount, columns: keyCount)

        case .embedding, .tokenEmbedding, .buffer:
            throw EmbedKitError.invalidConfiguration(
                "Expected similarityMatrix shape, got \(tensor.shape)"
            )
        }
    }

    // MARK: - Single Vector Extraction

    /// Extract a single vector from a 2D tensor.
    ///
    /// - Parameters:
    ///   - tensor: The tensor to extract from
    ///   - index: Row index
    ///   - dimensions: Vector dimensions
    /// - Returns: Single vector [dimensions]
    /// - Throws: `EmbedKitError` if index is out of range
    public func extractVector(
        from tensor: ManagedTensor,
        at index: Int,
        dimensions: Int
    ) throws -> [Float] {
        let totalCount = tensor.buffer.length / MemoryLayout<Float>.stride
        let offset = index * dimensions

        guard offset + dimensions <= totalCount else {
            throw EmbedKitError.invalidConfiguration(
                "Vector at index \(index) out of range (total elements: \(totalCount), dimensions: \(dimensions))"
            )
        }

        let ptr = tensor.buffer.contents().bindMemory(to: Float.self, capacity: totalCount)
        return Array(UnsafeBufferPointer(start: ptr + offset, count: dimensions))
    }

    /// Extract the first embedding from a tensor.
    ///
    /// - Parameter tensor: The tensor to extract from (must have embedding shape)
    /// - Returns: First embedding vector
    /// - Throws: `EmbedKitError` if tensor is empty or has wrong shape
    public func extractFirstEmbedding(from tensor: ManagedTensor) throws -> [Float] {
        switch tensor.shape {
        case .embedding(let batchSize, let dimensions):
            guard batchSize > 0 else {
                throw EmbedKitError.invalidConfiguration("Tensor has no embeddings")
            }
            return try extractVector(from: tensor, at: 0, dimensions: dimensions)

        case .tokenEmbedding, .similarityMatrix, .buffer:
            throw EmbedKitError.invalidConfiguration(
                "Expected embedding shape, got \(tensor.shape)"
            )
        }
    }

    // MARK: - Scalar Extraction

    /// Extract a single scalar value from a tensor.
    ///
    /// - Parameters:
    ///   - tensor: The tensor to extract from
    ///   - index: Element index
    /// - Returns: The scalar value
    /// - Throws: `EmbedKitError` if index is out of range
    public func extractScalar(from tensor: ManagedTensor, at index: Int) throws -> Float {
        let count = tensor.buffer.length / MemoryLayout<Float>.stride

        guard index >= 0, index < count else {
            throw EmbedKitError.invalidConfiguration(
                "Index \(index) out of range (tensor has \(count) elements)"
            )
        }

        let ptr = tensor.buffer.contents().bindMemory(to: Float.self, capacity: count)
        return ptr[index]
    }

    // MARK: - Statistics Extraction

    /// Compute statistics for tensor contents.
    ///
    /// - Parameter tensor: The tensor to analyze
    /// - Returns: Statistics about the tensor values
    public func extractStatistics(from tensor: ManagedTensor) -> TensorStatistics {
        let count = tensor.buffer.length / MemoryLayout<Float>.stride
        guard count > 0 else {
            return TensorStatistics(count: 0, min: 0, max: 0, mean: 0, standardDeviation: 0, l2Norm: 0)
        }

        let ptr = tensor.buffer.contents().bindMemory(to: Float.self, capacity: count)

        var minVal: Float = .greatestFiniteMagnitude
        var maxVal: Float = -.greatestFiniteMagnitude
        var sum: Double = 0
        var sumSquares: Double = 0

        for i in 0..<count {
            let val = ptr[i]
            minVal = min(minVal, val)
            maxVal = max(maxVal, val)
            sum += Double(val)
            sumSquares += Double(val) * Double(val)
        }

        let mean = sum / Double(count)
        let variance = (sumSquares / Double(count)) - (mean * mean)
        let standardDeviation = sqrt(max(0, variance))
        let l2Norm = sqrt(sumSquares)

        return TensorStatistics(
            count: count,
            min: minVal,
            max: maxVal,
            mean: Float(mean),
            standardDeviation: Float(standardDeviation),
            l2Norm: Float(l2Norm)
        )
    }

    // MARK: - Validation

    /// Validate that a tensor contains normalized vectors.
    ///
    /// - Parameters:
    ///   - tensor: The tensor to validate
    ///   - dimensions: Vector dimensions
    ///   - tolerance: Tolerance for L2 norm comparison (default: 1e-4)
    /// - Returns: Whether all vectors are approximately unit normalized
    public func validateNormalized(
        tensor: ManagedTensor,
        dimensions: Int,
        tolerance: Float = 1e-4
    ) -> Bool {
        let count = tensor.buffer.length / MemoryLayout<Float>.stride
        let vectorCount = count / dimensions

        guard vectorCount > 0, count % dimensions == 0 else {
            return false
        }

        let ptr = tensor.buffer.contents().bindMemory(to: Float.self, capacity: count)

        for v in 0..<vectorCount {
            let offset = v * dimensions
            var sumSquares: Float = 0

            for d in 0..<dimensions {
                let val = ptr[offset + d]
                sumSquares += val * val
            }

            let norm = sqrt(sumSquares)
            if abs(norm - 1.0) > tolerance {
                return false
            }
        }

        return true
    }

    /// Check if tensor contains any NaN or infinite values.
    ///
    /// - Parameter tensor: The tensor to check
    /// - Returns: Information about invalid values
    public func checkValidity(tensor: ManagedTensor) -> ValidityCheck {
        let count = tensor.buffer.length / MemoryLayout<Float>.stride
        let ptr = tensor.buffer.contents().bindMemory(to: Float.self, capacity: count)

        var nanCount = 0
        var infCount = 0
        var negInfCount = 0

        for i in 0..<count {
            let val = ptr[i]
            if val.isNaN {
                nanCount += 1
            } else if val.isInfinite {
                if val > 0 {
                    infCount += 1
                } else {
                    negInfCount += 1
                }
            }
        }

        return ValidityCheck(
            totalElements: count,
            nanCount: nanCount,
            positiveInfinityCount: infCount,
            negativeInfinityCount: negInfCount
        )
    }
}

// MARK: - Supporting Types

/// Statistics about tensor values.
public struct TensorStatistics: Sendable {
    /// Total number of elements
    public let count: Int

    /// Minimum value
    public let min: Float

    /// Maximum value
    public let max: Float

    /// Mean value
    public let mean: Float

    /// Standard deviation
    public let standardDeviation: Float

    /// L2 norm of all values
    public let l2Norm: Float

    /// Value range (max - min)
    public var range: Float { max - min }
}

/// Results of validity check.
public struct ValidityCheck: Sendable {
    /// Total number of elements
    public let totalElements: Int

    /// Number of NaN values
    public let nanCount: Int

    /// Number of positive infinity values
    public let positiveInfinityCount: Int

    /// Number of negative infinity values
    public let negativeInfinityCount: Int

    /// Whether tensor is fully valid (no NaN/Inf)
    public var isValid: Bool {
        nanCount == 0 && positiveInfinityCount == 0 && negativeInfinityCount == 0
    }

    /// Total invalid values
    public var invalidCount: Int {
        nanCount + positiveInfinityCount + negativeInfinityCount
    }
}

// MARK: - ManagedTensor Convenience Extensions

extension ManagedTensor {

    /// Extract tensor contents using the default extractor.
    ///
    /// - Returns: Flat array of all values
    public func toArray() -> [Float] {
        TensorResultExtractor().extractFlat(from: self)
    }

    /// Extract tensor as 2D array based on shape.
    ///
    /// - Returns: 2D array if shape is compatible
    /// - Throws: `EmbedKitError` if shape is not 2D
    public func to2DArray() throws -> [[Float]] {
        let extractor = TensorResultExtractor()

        switch shape {
        case .embedding(let batchSize, let dimensions):
            return try extractor.extract2D(from: self, rows: batchSize, columns: dimensions)

        case .similarityMatrix(let queryCount, let keyCount):
            return try extractor.extract2D(from: self, rows: queryCount, columns: keyCount)

        case .tokenEmbedding:
            throw EmbedKitError.invalidConfiguration("Token embedding is 3D, use to3DArray()")

        case .buffer(let length):
            // Treat as single-row 2D array
            return try extractor.extract2D(from: self, rows: 1, columns: length)
        }
    }

    /// Extract tensor as 3D array based on shape.
    ///
    /// - Returns: 3D array if shape is token embedding
    /// - Throws: `EmbedKitError` if shape is not 3D
    public func to3DArray() throws -> [[[Float]]] {
        let extractor = TensorResultExtractor()

        switch shape {
        case .tokenEmbedding(let batchSize, let sequenceLength, let dimensions):
            return try extractor.extract3D(from: self, dim0: batchSize, dim1: sequenceLength, dim2: dimensions)

        case .embedding, .similarityMatrix, .buffer:
            throw EmbedKitError.invalidConfiguration("Shape \(shape) is not 3D")
        }
    }

    /// Get statistics about tensor values.
    public func statistics() -> TensorStatistics {
        TensorResultExtractor().extractStatistics(from: self)
    }

    /// Check tensor validity.
    public func checkValidity() -> ValidityCheck {
        TensorResultExtractor().checkValidity(tensor: self)
    }
}

// MARK: - Batch Result Extraction

/// Helper for extracting results from multiple tensors.
public struct BatchResultExtractor: Sendable {

    private let extractor = TensorResultExtractor()

    public init() {}

    /// Extract embeddings from multiple tensors.
    ///
    /// - Parameter tensors: Array of embedding tensors
    /// - Returns: Combined 2D array of all embeddings
    public func extractEmbeddings(from tensors: [ManagedTensor]) throws -> [[Float]] {
        var results: [[Float]] = []

        for tensor in tensors {
            let embeddings = try extractor.extractEmbeddings(from: tensor)
            results.append(contentsOf: embeddings)
        }

        return results
    }

    /// Extract and concatenate flat arrays from multiple tensors.
    ///
    /// - Parameter tensors: Array of tensors
    /// - Returns: Single concatenated array
    public func extractConcatenated(from tensors: [ManagedTensor]) -> [Float] {
        var results: [Float] = []

        for tensor in tensors {
            results.append(contentsOf: extractor.extractFlat(from: tensor))
        }

        return results
    }

    /// Extract vectors at the same index from multiple tensors.
    ///
    /// - Parameters:
    ///   - tensors: Array of tensors
    ///   - index: Vector index in each tensor
    ///   - dimensions: Vector dimensions
    /// - Returns: Array of extracted vectors
    public func extractVectors(
        from tensors: [ManagedTensor],
        at index: Int,
        dimensions: Int
    ) throws -> [[Float]] {
        var results: [[Float]] = []

        for tensor in tensors {
            let vector = try extractor.extractVector(from: tensor, at: index, dimensions: dimensions)
            results.append(vector)
        }

        return results
    }
}

#endif
