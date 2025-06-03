import Testing
import Foundation
@testable import EmbedKit

@Suite("Metal Accelerator Tests")
struct MetalAcceleratorTests {
    
    @Test("Metal accelerator initialization")
    func metalAcceleratorInitialization() async throws {
        // Check if Metal is available
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available on this device")
        }
        
        // Accelerator was successfully created
    }
    
    @Test("L2 normalization correctness")
    func l2NormalizationCorrectness() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available on this device")
        }
        
        // Test data
        let vectors: [[Float]] = [
            [3.0, 4.0, 0.0],  // Should normalize to [0.6, 0.8, 0.0]
            [1.0, 0.0, 0.0],  // Should normalize to [1.0, 0.0, 0.0]
            [0.0, 0.0, 0.0]   // Zero vector should remain zero
        ]
        
        let normalized = try await accelerator.normalizeVectors(vectors)
        
        // Check first vector
        let expected1: [Float] = [0.6, 0.8, 0.0]
        for i in 0..<3 {
            #expect(abs(normalized[0][i] - expected1[i]) < 0.001)
        }
        
        // Check second vector
        let expected2: [Float] = [1.0, 0.0, 0.0]
        for i in 0..<3 {
            #expect(abs(normalized[1][i] - expected2[i]) < 0.001)
        }
        
        // Check third vector (zero vector)
        let expected3: [Float] = [0.0, 0.0, 0.0]
        for i in 0..<3 {
            #expect(abs(normalized[2][i] - expected3[i]) < 0.001)
        }
    }
    
    @Test("Mean pooling correctness")
    func meanPoolingCorrectness() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available on this device")
        }
        
        // Test data: 3 tokens, 4 dimensions
        let tokenEmbeddings: [[Float]] = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ]
        
        let pooled = try await accelerator.poolEmbeddings(tokenEmbeddings, strategy: .mean)
        
        // Expected: mean of each dimension
        let expected: [Float] = [5.0, 6.0, 7.0, 8.0]  // (1+5+9)/3, (2+6+10)/3, etc.
        
        #expect(pooled.count == 4)
        for i in 0..<4 {
            #expect(abs(pooled[i] - expected[i]) < 0.001)
        }
    }
    
    @Test("Mean pooling with attention mask")
    func meanPoolingWithMask() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available on this device")
        }
        
        // Test data: 3 tokens, 2 dimensions
        let tokenEmbeddings: [[Float]] = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ]
        
        // Mask: only use first two tokens
        let attentionMask = [1, 1, 0]
        
        let pooled = try await accelerator.poolEmbeddings(tokenEmbeddings, strategy: .mean, attentionMask: attentionMask)
        
        // Expected: mean of first two tokens only
        let expected: [Float] = [2.0, 3.0]  // (1+3)/2, (2+4)/2
        
        #expect(pooled.count == 2)
        for i in 0..<2 {
            #expect(abs(pooled[i] - expected[i]) < 0.001)
        }
    }
    
    @Test("Max pooling correctness")
    func maxPoolingCorrectness() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available on this device")
        }
        
        // Test data: 3 tokens, 4 dimensions
        let tokenEmbeddings: [[Float]] = [
            [1.0, 10.0, 3.0, 2.0],
            [5.0, 2.0, 7.0, 8.0],
            [3.0, 6.0, 1.0, 12.0]
        ]
        
        let pooled = try await accelerator.poolEmbeddings(tokenEmbeddings, strategy: .max)
        
        // Expected: max of each dimension
        let expected: [Float] = [5.0, 10.0, 7.0, 12.0]
        
        #expect(pooled.count == 4)
        for i in 0..<4 {
            #expect(abs(pooled[i] - expected[i]) < 0.001)
        }
    }
    
    @Test("CLS pooling correctness")
    func clsPoolingCorrectness() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available on this device")
        }
        
        // Test data: 3 tokens, 4 dimensions
        let tokenEmbeddings: [[Float]] = [
            [1.0, 2.0, 3.0, 4.0],  // CLS token
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ]
        
        let pooled = try await accelerator.poolEmbeddings(tokenEmbeddings, strategy: .cls)
        
        // Expected: first token (CLS)
        let expected: [Float] = [1.0, 2.0, 3.0, 4.0]
        
        #expect(pooled.count == 4)
        for i in 0..<4 {
            #expect(abs(pooled[i] - expected[i]) < 0.001)
        }
    }
    
    @Test("Attention-weighted pooling correctness")
    func attentionWeightedPoolingCorrectness() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available on this device")
        }
        
        // Test data: 3 tokens, 2 dimensions
        let tokenEmbeddings: [[Float]] = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ]
        
        // Attention weights: give more weight to middle token
        let attentionWeights: [Float] = [0.1, 0.8, 0.1]
        
        let pooled = try await accelerator.attentionWeightedPooling(tokenEmbeddings, attentionWeights: attentionWeights)
        
        // Expected: weighted average
        let expected: [Float] = [
            0.1 * 1.0 + 0.8 * 3.0 + 0.1 * 5.0,  // 3.0
            0.1 * 2.0 + 0.8 * 4.0 + 0.1 * 6.0   // 4.0
        ]
        
        #expect(pooled.count == 2)
        for i in 0..<2 {
            #expect(abs(pooled[i] - expected[i]) < 0.001)
        }
    }
    
    @Test("Cosine similarity matrix correctness")
    func cosineSimilarityMatrixCorrectness() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available on this device")
        }
        
        // Test data: orthogonal vectors
        let queries: [[Float]] = [
            [1.0, 0.0, 0.0],  // Unit vector along x-axis
            [0.0, 1.0, 0.0]   // Unit vector along y-axis
        ]
        
        let keys: [[Float]] = [
            [1.0, 0.0, 0.0],  // Same as first query
            [0.0, 0.0, 1.0]   // Unit vector along z-axis
        ]
        
        let similarities = try await accelerator.cosineSimilarityMatrix(queries: queries, keys: keys)
        
        #expect(similarities.count == 2)
        #expect(similarities[0].count == 2)
        #expect(similarities[1].count == 2)
        
        // First query vs first key: identical vectors = 1.0
        #expect(abs(similarities[0][0] - 1.0) < 0.001)
        
        // First query vs second key: orthogonal vectors = 0.0
        #expect(abs(similarities[0][1] - 0.0) < 0.001)
        
        // Second query vs first key: orthogonal vectors = 0.0
        #expect(abs(similarities[1][0] - 0.0) < 0.001)
        
        // Second query vs second key: orthogonal vectors = 0.0
        #expect(abs(similarities[1][1] - 0.0) < 0.001)
    }
    
    @Test("Metal performance vs CPU fallback")
    func metalPerformanceVsCPU() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available on this device")
        }
        
        // Generate larger test data
        let dimensions = 768
        let batchSize = 32
        
        var vectors: [[Float]] = []
        for _ in 0..<batchSize {
            var vector = [Float](repeating: 0, count: dimensions)
            for j in 0..<dimensions {
                vector[j] = Float.random(in: -1...1)
            }
            vectors.append(vector)
        }
        
        // Measure Metal performance
        let metalStart = CFAbsoluteTimeGetCurrent()
        let metalResult = try await accelerator.normalizeVectors(vectors)
        let metalTime = CFAbsoluteTimeGetCurrent() - metalStart
        
        // Simple CPU implementation for comparison
        let cpuStart = CFAbsoluteTimeGetCurrent()
        var cpuResult: [[Float]] = []
        for vector in vectors {
            let norm = sqrt(vector.reduce(0) { $0 + $1 * $1 })
            let normalized = norm > 0 ? vector.map { $0 / norm } : vector
            cpuResult.append(normalized)
        }
        let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart
        
        // Verify results are similar
        #expect(metalResult.count == cpuResult.count)
        for i in 0..<min(metalResult.count, cpuResult.count) {
            for j in 0..<min(metalResult[i].count, cpuResult[i].count) {
                #expect(abs(metalResult[i][j] - cpuResult[i][j]) < 0.01)
            }
        }
        
        print("Metal time: \(metalTime)s, CPU time: \(cpuTime)s, speedup: \(cpuTime/metalTime)x")
        
        // For this test, we don't enforce that Metal is faster since it depends on the device
        // But we verify that both produce correct results
    }
    
    @Test("Memory pressure handling")
    func memoryPressureHandling() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available on this device")
        }
        
        // Test that memory pressure handling doesn't crash
        await accelerator.handleMemoryPressure()
        
        // Verify that operations still work after memory pressure handling
        let vectors: [[Float]] = [[1.0, 2.0, 3.0]]
        let normalized = try await accelerator.normalizeVectors(vectors)
        
        #expect(normalized.count == 1)
        #expect(normalized[0].count == 3)
    }
    
    @Test("Error handling for invalid inputs")
    func errorHandlingForInvalidInputs() async throws {
        guard let accelerator = MetalAccelerator.shared else {
            throw XCTSkip("Metal not available on this device")
        }
        
        // Test empty input
        do {
            _ = try await accelerator.normalizeVectors([])
            // Empty input should be handled gracefully (return empty result)
        } catch {
            // Or throw an appropriate error
        }
        
        // Test dimension mismatch in cosine similarity
        let queries: [[Float]] = [[1.0, 2.0]]
        let keys: [[Float]] = [[1.0, 2.0, 3.0]]  // Different dimension
        
        do {
            _ = try await accelerator.cosineSimilarityMatrix(queries: queries, keys: keys)
            #expect(Bool(false), "Should have thrown dimension mismatch error")
        } catch {
            // Expected to throw an error
        }
        
        // Test attention weights mismatch
        let tokenEmbeddings: [[Float]] = [[1.0, 2.0], [3.0, 4.0]]
        let wrongWeights: [Float] = [0.5]  // Wrong count
        
        do {
            _ = try await accelerator.attentionWeightedPooling(tokenEmbeddings, attentionWeights: wrongWeights)
            #expect(Bool(false), "Should have thrown weight count mismatch error")
        } catch {
            // Expected to throw an error
        }
    }
}

// Helper for skipping tests when Metal is not available
struct XCTSkip: Error {
    let message: String
    
    init(_ message: String) {
        self.message = message
    }
}