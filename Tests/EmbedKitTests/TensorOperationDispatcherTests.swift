// EmbedKit - Tensor Operation Dispatcher Tests
//
// Tests for tensor operation dispatching and result extraction.

import Testing
import Foundation
@testable import EmbedKit

#if canImport(Metal)
import Metal
#endif

// MARK: - TensorOperationType Tests

@Suite("TensorOperationType")
struct TensorOperationTypeTests {

    @Test("All operation types are available")
    func allOperationTypesAvailable() {
        let allTypes = TensorOperationType.allCases
        #expect(allTypes.count >= 10)
        #expect(allTypes.contains(.normalize))
        #expect(allTypes.contains(.meanPool))
        #expect(allTypes.contains(.maxPool))
        #expect(allTypes.contains(.fusedPoolNormalize))
        #expect(allTypes.contains(.cosineSimilarity))
    }
}

// MARK: - TensorOperationConfig Tests

@Suite("TensorOperationConfig")
struct TensorOperationConfigTests {

    @Test("Normalize config factory")
    func normalizeConfigFactory() {
        let config = TensorOperationConfig.normalizeConfig()

        #expect(config.operation == .normalize)
        #expect(config.normalize == true)
    }

    @Test("Mean pool normalize config factory")
    func meanPoolNormalizeConfigFactory() {
        let config = TensorOperationConfig.meanPoolNormalize()

        #expect(config.operation == .fusedPoolNormalize)
        #expect(config.poolingStrategy == .mean)
        #expect(config.normalize == true)
    }

    @Test("Max pool normalize config factory")
    func maxPoolNormalizeConfigFactory() {
        let config = TensorOperationConfig.maxPoolNormalize()

        #expect(config.operation == .fusedPoolNormalize)
        #expect(config.poolingStrategy == .max)
        #expect(config.normalize == true)
    }

    @Test("CLS pool normalize config factory")
    func clsPoolNormalizeConfigFactory() {
        let config = TensorOperationConfig.clsPoolNormalize()

        #expect(config.operation == .fusedPoolNormalize)
        #expect(config.poolingStrategy == .cls)
        #expect(config.normalize == true)
    }

    @Test("Attention pool config with weights")
    func attentionPoolConfigWithWeights() {
        let weights: [Float] = [0.1, 0.2, 0.3, 0.4]
        let config = TensorOperationConfig.attentionPoolNormalize(weights: weights)

        #expect(config.operation == .attentionPool)
        #expect(config.poolingStrategy == .attention)
        #expect(config.weights?.count == 4)
    }

    @Test("Cosine similarity config")
    func cosineSimilarityConfig() {
        let config = TensorOperationConfig.cosineSimilarity()

        #expect(config.operation == .cosineSimilarity)
        #expect(config.normalize == false)
    }

    @Test("Config with mask")
    func configWithMask() {
        let mask: [Int32] = [1, 1, 1, 0, 0]
        let config = TensorOperationConfig.meanPoolNormalize(mask: mask)

        #expect(config.mask?.count == 5)
        #expect(config.mask?[0] == 1)
        #expect(config.mask?[4] == 0)
    }
}

// MARK: - TensorOperationDispatcher Tests

@Suite("TensorOperationDispatcher", .serialized)
struct TensorOperationDispatcherTests {

    #if canImport(Metal)
    @Test("Dispatcher initializes correctly")
    func initializesCorrectly() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorOperationTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let accelerator = await MetalAccelerator()

        let dispatcher = await TensorOperationDispatcher(
            accelerator: accelerator,
            storageManager: storageManager
        )

        let stats = await dispatcher.getStatistics()
        #expect(stats.totalOperations == 0)
        #expect(stats.gpuOperations == 0)
        #expect(stats.cpuOperations == 0)
    }

    @Test("Executes normalization operation")
    func executesNormalization() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorOperationTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let accelerator = await MetalAccelerator()
        let dispatcher = await TensorOperationDispatcher(
            accelerator: accelerator,
            storageManager: storageManager
        )

        // Create input tensor with test data
        let batchSize = 4
        let dimensions = 64
        let input = try await storageManager.createEmbeddingTensor(
            batchSize: batchSize,
            dimensions: dimensions,
            label: "input"
        )

        // Fill with test data
        let inputPtr = input.buffer.contents().bindMemory(to: Float.self, capacity: batchSize * dimensions)
        for i in 0..<(batchSize * dimensions) {
            inputPtr[i] = Float(i % 10) + 1.0  // Values 1-10
        }

        // Create output tensor
        let output = try await storageManager.createEmbeddingTensor(
            batchSize: batchSize,
            dimensions: dimensions,
            label: "output"
        )

        // Execute normalization
        let result = try await dispatcher.execute(
            input: input,
            output: output,
            config: .normalizeConfig(),
            inputShape: .embedding(batchSize: batchSize, dimensions: dimensions)
        )

        #expect(result.operation == .normalize)
        #expect(result.executionTime > 0)

        // Verify output is normalized
        let extractor = TensorResultExtractor()
        let isNormalized = extractor.validateNormalized(tensor: output, dimensions: dimensions, tolerance: 1e-3)
        #expect(isNormalized)

        // Cleanup
        await storageManager.releaseAll()
    }

    @Test("Executes fused pool normalize operation")
    func executesFusedPoolNormalize() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorOperationTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let accelerator = await MetalAccelerator()
        let dispatcher = await TensorOperationDispatcher(
            accelerator: accelerator,
            storageManager: storageManager
        )

        // Create token embeddings
        let batchSize = 2
        let sequenceLength = 8
        let dimensions = 32
        let input = try await storageManager.createTokenEmbeddingTensor(
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            dimensions: dimensions,
            label: "token_input"
        )

        // Fill with test data
        let inputPtr = input.buffer.contents().bindMemory(
            to: Float.self,
            capacity: batchSize * sequenceLength * dimensions
        )
        for i in 0..<(batchSize * sequenceLength * dimensions) {
            inputPtr[i] = Float.random(in: -1...1)
        }

        // Create output tensor for pooled embeddings
        let output = try await storageManager.createEmbeddingTensor(
            batchSize: batchSize,
            dimensions: dimensions,
            label: "pooled_output"
        )

        // Execute fused pool + normalize
        let result = try await dispatcher.execute(
            input: input,
            output: output,
            config: .meanPoolNormalize(),
            inputShape: .tokenEmbedding(batchSize: batchSize, sequenceLength: sequenceLength, dimensions: dimensions)
        )

        #expect(result.operation == .fusedPoolNormalize)
        #expect(result.executionTime > 0)

        // Verify output is normalized
        let extractor = TensorResultExtractor()
        let isNormalized = extractor.validateNormalized(tensor: output, dimensions: dimensions, tolerance: 1e-3)
        #expect(isNormalized)

        // Cleanup
        await storageManager.releaseAll()
    }

    @Test("Executes similarity operation")
    func executesSimilarity() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorOperationTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let accelerator = await MetalAccelerator()
        let dispatcher = await TensorOperationDispatcher(
            accelerator: accelerator,
            storageManager: storageManager
        )

        // Create query and key tensors
        let queryCount = 4
        let keyCount = 8
        let dimensions = 16

        let queries = try await storageManager.createEmbeddingTensor(
            batchSize: queryCount,
            dimensions: dimensions,
            label: "queries"
        )

        let keys = try await storageManager.createEmbeddingTensor(
            batchSize: keyCount,
            dimensions: dimensions,
            label: "keys"
        )

        // Fill with normalized random data
        let qPtr = queries.buffer.contents().bindMemory(to: Float.self, capacity: queryCount * dimensions)
        let kPtr = keys.buffer.contents().bindMemory(to: Float.self, capacity: keyCount * dimensions)

        for q in 0..<queryCount {
            var sumSq: Float = 0
            for d in 0..<dimensions {
                qPtr[q * dimensions + d] = Float.random(in: -1...1)
                sumSq += qPtr[q * dimensions + d] * qPtr[q * dimensions + d]
            }
            let norm = sqrt(sumSq)
            for d in 0..<dimensions {
                qPtr[q * dimensions + d] /= norm
            }
        }

        for k in 0..<keyCount {
            var sumSq: Float = 0
            for d in 0..<dimensions {
                kPtr[k * dimensions + d] = Float.random(in: -1...1)
                sumSq += kPtr[k * dimensions + d] * kPtr[k * dimensions + d]
            }
            let norm = sqrt(sumSq)
            for d in 0..<dimensions {
                kPtr[k * dimensions + d] /= norm
            }
        }

        // Create output tensor
        let output = try await storageManager.createSimilarityTensor(
            queryCount: queryCount,
            keyCount: keyCount,
            label: "similarity"
        )

        // Execute similarity
        let result = try await dispatcher.executeSimilarity(
            queries: queries,
            keys: keys,
            output: output,
            queryCount: queryCount,
            keyCount: keyCount,
            dimensions: dimensions,
            normalized: true
        )

        #expect(result.executionTime > 0)

        // Verify output values are in valid range [-1, 1]
        let extractor = TensorResultExtractor()
        let matrix = try extractor.extractSimilarityMatrix(from: output)

        #expect(matrix.count == queryCount)
        #expect(matrix[0].count == keyCount)

        for row in matrix {
            for val in row {
                #expect(val >= -1.1 && val <= 1.1)  // Allow small tolerance
            }
        }

        // Cleanup
        await storageManager.releaseAll()
    }

    @Test("Statistics are tracked correctly")
    func statisticsTracked() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorOperationTestError.skipped("Metal not available")
        }

        let storageManager = TensorStorageManager(device: device)
        let accelerator = await MetalAccelerator()
        let dispatcher = await TensorOperationDispatcher(
            accelerator: accelerator,
            storageManager: storageManager
        )

        // Create small tensors (will use CPU fallback)
        let input = try await storageManager.createEmbeddingTensor(
            batchSize: 2,
            dimensions: 4,
            label: "small_input"
        )

        let output = try await storageManager.createEmbeddingTensor(
            batchSize: 2,
            dimensions: 4,
            label: "small_output"
        )

        // Fill input
        let ptr = input.buffer.contents().bindMemory(to: Float.self, capacity: 8)
        for i in 0..<8 { ptr[i] = Float(i + 1) }

        // Execute a few operations
        _ = try await dispatcher.execute(
            input: input,
            output: output,
            config: .normalizeConfig(),
            inputShape: .embedding(batchSize: 2, dimensions: 4)
        )

        _ = try await dispatcher.execute(
            input: input,
            output: output,
            config: .normalizeConfig(),
            inputShape: .embedding(batchSize: 2, dimensions: 4)
        )

        let stats = await dispatcher.getStatistics()
        #expect(stats.totalOperations == 2)
        #expect(stats.cpuOperations + stats.gpuOperations == 2)

        // Reset
        await dispatcher.resetStatistics()
        let resetStats = await dispatcher.getStatistics()
        #expect(resetStats.totalOperations == 0)

        // Cleanup
        await storageManager.releaseAll()
    }

    @Test("zz_cleanup - Release shared resources after tests")
    func zz_cleanupResources() async {
        await cleanupMetalTestResources()
    }
    #endif
}

// MARK: - InputShape Tests

@Suite("InputShape")
struct InputShapeTests {

    @Test("Embedding shape factory")
    func embeddingShapeFactory() {
        let shape = TensorOperationDispatcher.InputShape.embedding(batchSize: 32, dimensions: 384)

        #expect(shape.batchSize == 32)
        #expect(shape.sequenceLength == 1)
        #expect(shape.dimensions == 384)
        #expect(shape.elementCount == 32 * 384)
    }

    @Test("Token embedding shape factory")
    func tokenEmbeddingShapeFactory() {
        let shape = TensorOperationDispatcher.InputShape.tokenEmbedding(
            batchSize: 4,
            sequenceLength: 128,
            dimensions: 384
        )

        #expect(shape.batchSize == 4)
        #expect(shape.sequenceLength == 128)
        #expect(shape.dimensions == 384)
        #expect(shape.elementCount == 4 * 128 * 384)
    }
}

// MARK: - TensorResultExtractor Tests

@Suite("TensorResultExtractor")
struct TensorResultExtractorTests {

    #if canImport(Metal)
    @Test("Extracts flat array")
    func extractsFlatArray() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorOperationTestError.skipped("Metal not available")
        }

        let count = 100
        guard let buffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride) else {
            throw TensorOperationTestError.skipped("Failed to create buffer")
        }

        // Fill buffer
        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            ptr[i] = Float(i)
        }

        let tensor = ManagedTensor(
            buffer: buffer,
            shape: .buffer(length: count),
            label: "test"
        )

        let extractor = TensorResultExtractor()
        let result = extractor.extractFlat(from: tensor)

        #expect(result.count == count)
        #expect(result[0] == 0)
        #expect(result[50] == 50)
        #expect(result[99] == 99)
    }

    @Test("Extracts partial flat array")
    func extractsPartialFlatArray() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorOperationTestError.skipped("Metal not available")
        }

        let count = 100
        guard let buffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride) else {
            throw TensorOperationTestError.skipped("Failed to create buffer")
        }

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            ptr[i] = Float(i)
        }

        let tensor = ManagedTensor(
            buffer: buffer,
            shape: .buffer(length: count),
            label: "test"
        )

        let extractor = TensorResultExtractor()
        let result = try extractor.extractFlat(from: tensor, offset: 10, count: 20)

        #expect(result.count == 20)
        #expect(result[0] == 10)
        #expect(result[19] == 29)
    }

    @Test("Extracts 2D array")
    func extracts2DArray() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorOperationTestError.skipped("Metal not available")
        }

        let rows = 4
        let cols = 8
        let count = rows * cols
        guard let buffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride) else {
            throw TensorOperationTestError.skipped("Failed to create buffer")
        }

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            ptr[i] = Float(i)
        }

        let tensor = ManagedTensor(
            buffer: buffer,
            shape: .embedding(batchSize: rows, dimensions: cols),
            label: "test"
        )

        let extractor = TensorResultExtractor()
        let result = try extractor.extract2D(from: tensor, rows: rows, columns: cols)

        #expect(result.count == rows)
        #expect(result[0].count == cols)
        #expect(result[0][0] == 0)
        #expect(result[0][7] == 7)
        #expect(result[3][0] == 24)
        #expect(result[3][7] == 31)
    }

    @Test("Extracts 3D array")
    func extracts3DArray() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorOperationTestError.skipped("Metal not available")
        }

        let dim0 = 2
        let dim1 = 3
        let dim2 = 4
        let count = dim0 * dim1 * dim2
        guard let buffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride) else {
            throw TensorOperationTestError.skipped("Failed to create buffer")
        }

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            ptr[i] = Float(i)
        }

        let tensor = ManagedTensor(
            buffer: buffer,
            shape: .tokenEmbedding(batchSize: dim0, sequenceLength: dim1, dimensions: dim2),
            label: "test"
        )

        let extractor = TensorResultExtractor()
        let result = try extractor.extract3D(from: tensor, dim0: dim0, dim1: dim1, dim2: dim2)

        #expect(result.count == dim0)
        #expect(result[0].count == dim1)
        #expect(result[0][0].count == dim2)
        #expect(result[0][0][0] == 0)
        #expect(result[1][2][3] == 23)  // 1*12 + 2*4 + 3 = 23
    }

    @Test("Extracts single vector")
    func extractsSingleVector() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorOperationTestError.skipped("Metal not available")
        }

        let rows = 5
        let dims = 10
        let count = rows * dims
        guard let buffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride) else {
            throw TensorOperationTestError.skipped("Failed to create buffer")
        }

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            ptr[i] = Float(i)
        }

        let tensor = ManagedTensor(
            buffer: buffer,
            shape: .embedding(batchSize: rows, dimensions: dims),
            label: "test"
        )

        let extractor = TensorResultExtractor()
        let vector = try extractor.extractVector(from: tensor, at: 2, dimensions: dims)

        #expect(vector.count == dims)
        #expect(vector[0] == 20)  // 2 * 10
        #expect(vector[9] == 29)
    }

    @Test("Extracts scalar")
    func extractsScalar() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorOperationTestError.skipped("Metal not available")
        }

        let count = 50
        guard let buffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride) else {
            throw TensorOperationTestError.skipped("Failed to create buffer")
        }

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            ptr[i] = Float(i) * 2.0
        }

        let tensor = ManagedTensor(
            buffer: buffer,
            shape: .buffer(length: count),
            label: "test"
        )

        let extractor = TensorResultExtractor()
        let scalar = try extractor.extractScalar(from: tensor, at: 25)

        #expect(scalar == 50.0)
    }

    @Test("Computes statistics")
    func computesStatistics() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorOperationTestError.skipped("Metal not available")
        }

        let count = 100
        guard let buffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride) else {
            throw TensorOperationTestError.skipped("Failed to create buffer")
        }

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            ptr[i] = Float(i)  // 0 to 99
        }

        let tensor = ManagedTensor(
            buffer: buffer,
            shape: .buffer(length: count),
            label: "test"
        )

        let extractor = TensorResultExtractor()
        let stats = extractor.extractStatistics(from: tensor)

        #expect(stats.count == 100)
        #expect(stats.min == 0)
        #expect(stats.max == 99)
        #expect(abs(stats.mean - 49.5) < 0.1)
        #expect(stats.range == 99)
    }

    @Test("Validates normalized vectors")
    func validatesNormalizedVectors() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorOperationTestError.skipped("Metal not available")
        }

        let batchSize = 3
        let dims = 4
        let count = batchSize * dims
        guard let buffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride) else {
            throw TensorOperationTestError.skipped("Failed to create buffer")
        }

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)

        // Create normalized vectors
        for b in 0..<batchSize {
            // Simple unit vector: [1, 0, 0, 0], [0, 1, 0, 0], etc.
            for d in 0..<dims {
                ptr[b * dims + d] = (d == b % dims) ? 1.0 : 0.0
            }
        }

        let tensor = ManagedTensor(
            buffer: buffer,
            shape: .embedding(batchSize: batchSize, dimensions: dims),
            label: "test"
        )

        let extractor = TensorResultExtractor()
        let isNormalized = extractor.validateNormalized(tensor: tensor, dimensions: dims)

        #expect(isNormalized)
    }

    @Test("Checks validity detects NaN")
    func checksValidityDetectsNaN() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorOperationTestError.skipped("Metal not available")
        }

        let count = 10
        guard let buffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride) else {
            throw TensorOperationTestError.skipped("Failed to create buffer")
        }

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            ptr[i] = Float(i)
        }
        ptr[5] = Float.nan

        let tensor = ManagedTensor(
            buffer: buffer,
            shape: .buffer(length: count),
            label: "test"
        )

        let extractor = TensorResultExtractor()
        let validity = extractor.checkValidity(tensor: tensor)

        #expect(validity.isValid == false)
        #expect(validity.nanCount == 1)
        #expect(validity.totalElements == 10)
    }

    @Test("Checks validity detects infinity")
    func checksValidityDetectsInfinity() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorOperationTestError.skipped("Metal not available")
        }

        let count = 10
        guard let buffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride) else {
            throw TensorOperationTestError.skipped("Failed to create buffer")
        }

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            ptr[i] = Float(i)
        }
        ptr[3] = Float.infinity
        ptr[7] = -Float.infinity

        let tensor = ManagedTensor(
            buffer: buffer,
            shape: .buffer(length: count),
            label: "test"
        )

        let extractor = TensorResultExtractor()
        let validity = extractor.checkValidity(tensor: tensor)

        #expect(validity.isValid == false)
        #expect(validity.positiveInfinityCount == 1)
        #expect(validity.negativeInfinityCount == 1)
        #expect(validity.invalidCount == 2)
    }
    #endif
}

// MARK: - ManagedTensor Extraction Extensions Tests

@Suite("ManagedTensorExtractionExtensions")
struct ManagedTensorExtractionExtensionsTests {

    #if canImport(Metal)
    @Test("toArray extension works")
    func toArrayExtensionWorks() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorOperationTestError.skipped("Metal not available")
        }

        let count = 20
        guard let buffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride) else {
            throw TensorOperationTestError.skipped("Failed to create buffer")
        }

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            ptr[i] = Float(i)
        }

        let tensor = ManagedTensor(
            buffer: buffer,
            shape: .buffer(length: count),
            label: "test"
        )

        let array = tensor.toArray()
        #expect(array.count == count)
        #expect(array[0] == 0)
        #expect(array[19] == 19)
    }

    @Test("to2DArray extension works")
    func to2DArrayExtensionWorks() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorOperationTestError.skipped("Metal not available")
        }

        let rows = 3
        let cols = 5
        let count = rows * cols
        guard let buffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride) else {
            throw TensorOperationTestError.skipped("Failed to create buffer")
        }

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            ptr[i] = Float(i)
        }

        let tensor = ManagedTensor(
            buffer: buffer,
            shape: .embedding(batchSize: rows, dimensions: cols),
            label: "test"
        )

        let array2D = try tensor.to2DArray()
        #expect(array2D.count == rows)
        #expect(array2D[0].count == cols)
    }

    @Test("statistics extension works")
    func statisticsExtensionWorks() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorOperationTestError.skipped("Metal not available")
        }

        let count = 50
        guard let buffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride) else {
            throw TensorOperationTestError.skipped("Failed to create buffer")
        }

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            ptr[i] = Float(i)
        }

        let tensor = ManagedTensor(
            buffer: buffer,
            shape: .buffer(length: count),
            label: "test"
        )

        let stats = tensor.statistics()
        #expect(stats.count == 50)
        #expect(stats.min == 0)
        #expect(stats.max == 49)
    }

    @Test("checkValidity extension works")
    func checkValidityExtensionWorks() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorOperationTestError.skipped("Metal not available")
        }

        let count = 10
        guard let buffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride) else {
            throw TensorOperationTestError.skipped("Failed to create buffer")
        }

        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            ptr[i] = Float(i)
        }

        let tensor = ManagedTensor(
            buffer: buffer,
            shape: .buffer(length: count),
            label: "test"
        )

        let validity = tensor.checkValidity()
        #expect(validity.isValid == true)
    }
    #endif
}

// MARK: - BatchResultExtractor Tests

@Suite("BatchResultExtractor")
struct BatchResultExtractorTests {

    #if canImport(Metal)
    @Test("Extracts from multiple tensors")
    func extractsFromMultipleTensors() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorOperationTestError.skipped("Metal not available")
        }

        let batchSize = 2
        let dims = 4
        let count = batchSize * dims

        var tensors: [ManagedTensor] = []

        for t in 0..<3 {
            guard let buffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride) else {
                throw TensorOperationTestError.skipped("Failed to create buffer")
            }

            let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
            for i in 0..<count {
                ptr[i] = Float(t * 100 + i)
            }

            tensors.append(ManagedTensor(
                buffer: buffer,
                shape: .embedding(batchSize: batchSize, dimensions: dims),
                label: "tensor_\(t)"
            ))
        }

        let extractor = BatchResultExtractor()
        let embeddings = try extractor.extractEmbeddings(from: tensors)

        // Should have 3 * 2 = 6 embeddings
        #expect(embeddings.count == 6)
        #expect(embeddings[0][0] == 0)     // First tensor, first embedding
        #expect(embeddings[2][0] == 100)   // Second tensor, first embedding
        #expect(embeddings[4][0] == 200)   // Third tensor, first embedding
    }

    @Test("Extracts concatenated")
    func extractsConcatenated() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TensorOperationTestError.skipped("Metal not available")
        }

        var tensors: [ManagedTensor] = []

        for t in 0..<2 {
            let count = 5
            guard let buffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride) else {
                throw TensorOperationTestError.skipped("Failed to create buffer")
            }

            let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
            for i in 0..<count {
                ptr[i] = Float(t * 10 + i)
            }

            tensors.append(ManagedTensor(
                buffer: buffer,
                shape: .buffer(length: count),
                label: "tensor_\(t)"
            ))
        }

        let extractor = BatchResultExtractor()
        let concatenated = extractor.extractConcatenated(from: tensors)

        #expect(concatenated.count == 10)
        #expect(concatenated[0] == 0)
        #expect(concatenated[5] == 10)
    }
    #endif
}

// MARK: - Test Helper

enum TensorOperationTestError: Error {
    case skipped(String)
}
