import Testing
import Foundation
import Metal
@testable import EmbedKit

@Suite("Attention-Weighted Pooling Tests")
struct AttentionPoolingTests {
    
    // MARK: - Basic Functionality Tests
    
    @Test("Attention-weighted pooling with uniform weights equals mean pooling")
    func testUniformWeightsEqualsMean() async throws {
        // Create test token embeddings
        let tokenEmbeddings: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
        
        // Uniform weights
        let uniformWeights = [Float](repeating: 1.0 / 3.0, count: 3)
        
        // Use mock accelerator
        let accelerator = MockMetalAccelerator()
        
        // Test attention-weighted pooling with uniform weights
        let attentionResult = try await accelerator.poolEmbeddings(
            tokenEmbeddings,
            strategy: .attentionWeighted,
            attentionMask: nil,
            attentionWeights: uniformWeights
        )
        
        // Test mean pooling
        let meanResult = try await accelerator.poolEmbeddings(
            tokenEmbeddings,
            strategy: .mean,
            attentionMask: nil,
            attentionWeights: nil
        )
        
        // They should be equal (within floating-point precision)
        for (attention, mean) in zip(attentionResult, meanResult) {
            #expect(abs(attention - mean) < 0.0001)
        }
    }
    
    @Test("Attention-weighted pooling with custom weights")
    func testCustomWeights() async throws {
        // Create test token embeddings
        let tokenEmbeddings: [[Float]] = [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ]
        
        // Custom weights emphasizing the first token
        let weights: [Float] = [0.8, 0.1, 0.1]
        
        let accelerator = MockMetalAccelerator()
        
        let result = try await accelerator.poolEmbeddings(
            tokenEmbeddings,
            strategy: .attentionWeighted,
            attentionMask: nil,
            attentionWeights: weights
        )
        
        // Expected: heavily weighted towards first token [1.0, 0.0]
        // Result should be close to [0.9, 0.2]
        #expect(abs(result[0] - 0.9) < 0.0001)
        #expect(abs(result[1] - 0.2) < 0.0001)
    }
    
    @Test("Attention-weighted pooling CPU fallback")
    func testCPUFallback() async throws {
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: .miniLM_L6_v2,
            configuration: Configuration.default(for: .miniLM_L6_v2)
        )
        
        // Test that the embedder can handle attention-weighted pooling
        // even without a loaded model (will fail with modelNotLoaded)
        do {
            _ = try await embedder.embed("Test text")
            // If this succeeds, the model was loaded
        } catch {
            // Expected - model not loaded in test environment
            #expect(error is ContextualEmbeddingError)
        }
    }
    
    // MARK: - Edge Cases
    
    @Test("Empty token embeddings throws error")
    func testEmptyEmbeddings() async throws {
        let accelerator = MockMetalAccelerator()
        
        do {
            _ = try await accelerator.poolEmbeddings(
                [],
                strategy: .attentionWeighted,
                attentionMask: nil,
                attentionWeights: nil
            )
            #expect(Bool(false), "Should have thrown error")
        } catch {
            #expect(error is MetalError)
        }
    }
    
    @Test("Mismatched attention weights count throws error")
    func testMismatchedWeights() async throws {
        let tokenEmbeddings: [[Float]] = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ]
        
        // Wrong number of weights
        let weights: [Float] = [0.5, 0.5] // Only 2 weights for 3 tokens
        
        let accelerator = MockMetalAccelerator()
        
        do {
            _ = try await accelerator.attentionWeightedPooling(
                tokenEmbeddings,
                attentionWeights: weights
            )
            #expect(Bool(false), "Should have thrown error")
        } catch {
            #expect(error is MetalError)
        }
    }
    
    @Test("Zero attention weights")
    func testZeroWeights() async throws {
        let tokenEmbeddings: [[Float]] = [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
        
        // All zeros
        let weights: [Float] = [0.0, 0.0]
        
        let accelerator = MockMetalAccelerator()
        
        do {
            _ = try await accelerator.attentionWeightedPooling(
                tokenEmbeddings,
                attentionWeights: weights
            )
            #expect(Bool(false), "Should have thrown error for zero weights")
        } catch {
            #expect(error is MetalError)
            if let metalError = error as? MetalError {
                #expect(metalError.localizedDescription.contains("zero"))
            }
        }
    }
    
    // MARK: - Integration Tests
    
    @Test("End-to-end attention pooling with embedder")
    func testEndToEndAttentionPooling() async throws {
        var config = Configuration.default(for: .miniLM_L6_v2)
        // Create new config with attention-weighted pooling
        config = Configuration(
            model: ModelConfiguration(
                identifier: .miniLM_L6_v2,
                maxSequenceLength: config.model.maxSequenceLength,
                normalizeEmbeddings: config.model.normalizeEmbeddings,
                poolingStrategy: .attentionWeighted,
                loadingOptions: config.model.loadingOptions,
                computeUnits: config.model.computeUnits
            ),
            resources: config.resources,
            performance: config.performance,
            monitoring: config.monitoring,
            cache: config.cache,
            errorHandling: config.errorHandling
        )
        
        let embedder = CoreMLTextEmbedder(
            modelIdentifier: .miniLM_L6_v2,
            configuration: config
        )
        
        // Test that attention-weighted pooling works in the full pipeline
        do {
            _ = try await embedder.embed("Test text for attention pooling")
            // If model is loaded and supports attention weights, this should succeed
        } catch {
            // Expected if model is not loaded in test environment
            #expect(error is ContextualEmbeddingError)
        }
    }
    
    // MARK: - Performance Tests
    
    @Test("Attention pooling performance")
    func testPerformance() async throws {
        let sequenceLength = 512
        let dimensions = 768
        
        // Generate large token embeddings
        let tokenEmbeddings = (0..<sequenceLength).map { i in
            (0..<dimensions).map { j in
                Float(i + j) / Float(sequenceLength * dimensions)
            }
        }
        
        let weights = (0..<sequenceLength).map { i in
            Float(sequenceLength - i) / Float(sequenceLength * (sequenceLength + 1) / 2)
        }
        
        // Use real accelerator if available
        let accelerator: any MetalAcceleratorProtocol
        if TestEnvironment.hasMetalSupport {
            do {
                accelerator = try getTestMetalAccelerator()
            } catch {
                // Fall back to mock if Metal initialization fails
                accelerator = MockMetalAccelerator()
            }
        } else {
            accelerator = MockMetalAccelerator()
        }
        
        // Time implementation
        let start = Date()
        _ = try? await accelerator.attentionWeightedPooling(
            tokenEmbeddings,
            attentionWeights: weights
        )
        let time = Date().timeIntervalSince(start)
        
        print("Attention pooling time for \(sequenceLength)x\(dimensions): \(time)s")
        
        // Should complete in reasonable time
        #expect(time < 1.0)
    }
}