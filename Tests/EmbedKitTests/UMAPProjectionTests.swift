// Tests for UMAP Projection (GPU-accelerated dimensionality reduction)
import Testing
import Foundation
@testable import EmbedKit

// MARK: - UMAP Configuration Tests

@Suite("UMAPConfiguration - Presets")
struct UMAPConfigurationPresetTests {

    @Test("Default configuration values")
    func defaultConfiguration() {
        let config = UMAPConfiguration.default
        #expect(config.targetDimension == 2)
        #expect(config.neighbors == 15)
        #expect(config.minDistance == 0.1)
        #expect(config.iterations == 200)
        #expect(config.learningRate == 1.0)
        #expect(config.negativeSampleRate == 5)
        #expect(config.spread == 1.0)
    }

    @Test("visualization2D preset")
    func visualization2DPreset() {
        let config = UMAPConfiguration.visualization2D()
        #expect(config.targetDimension == 2)
        #expect(config.neighbors == 15)
        #expect(config.minDistance == 0.1)
        #expect(config.iterations == 200)
    }

    @Test("visualization3D preset")
    func visualization3DPreset() {
        let config = UMAPConfiguration.visualization3D()
        #expect(config.targetDimension == 3)
        #expect(config.neighbors == 15)
        #expect(config.minDistance == 0.1)
        #expect(config.iterations == 250)
    }

    @Test("quickPreview preset")
    func quickPreviewPreset() {
        let config = UMAPConfiguration.quickPreview()
        #expect(config.targetDimension == 2)
        #expect(config.neighbors == 10)
        #expect(config.iterations == 100)
    }

    @Test("quickPreview with custom dimension")
    func quickPreviewCustomDimension() {
        let config = UMAPConfiguration.quickPreview(dimension: 3)
        #expect(config.targetDimension == 3)
    }

    @Test("highQuality preset")
    func highQualityPreset() {
        let config = UMAPConfiguration.highQuality()
        #expect(config.targetDimension == 2)
        #expect(config.neighbors == 30)
        #expect(config.iterations == 500)
    }

    @Test("clusterEmphasis preset")
    func clusterEmphasisPreset() {
        let config = UMAPConfiguration.clusterEmphasis()
        #expect(config.targetDimension == 2)
        #expect(config.neighbors == 10)
        #expect(config.minDistance == 0.05)
    }

    @Test("globalStructure preset")
    func globalStructurePreset() {
        let config = UMAPConfiguration.globalStructure()
        #expect(config.targetDimension == 2)
        #expect(config.neighbors == 50)
        #expect(config.minDistance == 0.25)
    }
}

// MARK: - UMAP Configuration Validation Tests

@Suite("UMAPConfiguration - Validation")
struct UMAPConfigurationValidationTests {

    @Test("Valid configuration passes validation")
    func validConfigurationPasses() throws {
        let config = UMAPConfiguration.default
        try config.validate()
    }

    @Test("Invalid target dimension throws")
    func invalidTargetDimensionThrows() {
        let config = UMAPConfiguration(targetDimension: 0)
        #expect(throws: UMAPConfigurationError.self) {
            try config.validate()
        }
    }

    @Test("Invalid neighbors throws")
    func invalidNeighborsThrows() {
        let config = UMAPConfiguration(neighbors: 1)
        #expect(throws: UMAPConfigurationError.self) {
            try config.validate()
        }
    }

    @Test("Invalid minDistance throws (negative)")
    func invalidMinDistanceNegativeThrows() {
        let config = UMAPConfiguration(minDistance: -0.1)
        #expect(throws: UMAPConfigurationError.self) {
            try config.validate()
        }
    }

    @Test("Invalid minDistance throws (greater than 1)")
    func invalidMinDistanceGreaterThanOneThrows() {
        let config = UMAPConfiguration(minDistance: 1.5)
        #expect(throws: UMAPConfigurationError.self) {
            try config.validate()
        }
    }

    @Test("Invalid iterations throws")
    func invalidIterationsThrows() {
        let config = UMAPConfiguration(iterations: 0)
        #expect(throws: UMAPConfigurationError.self) {
            try config.validate()
        }
    }

    @Test("Invalid learning rate throws")
    func invalidLearningRateThrows() {
        let config = UMAPConfiguration(learningRate: 0)
        #expect(throws: UMAPConfigurationError.self) {
            try config.validate()
        }
    }

    @Test("Invalid negative sample rate throws")
    func invalidNegativeSampleRateThrows() {
        let config = UMAPConfiguration(negativeSampleRate: -1)
        #expect(throws: UMAPConfigurationError.self) {
            try config.validate()
        }
    }

    @Test("Invalid spread throws")
    func invalidSpreadThrows() {
        let config = UMAPConfiguration(spread: 0)
        #expect(throws: UMAPConfigurationError.self) {
            try config.validate()
        }
    }
}

// MARK: - UMAP Configuration Codable Tests

@Suite("UMAPConfiguration - Codable")
struct UMAPConfigurationCodableTests {

    @Test("Configuration encodes and decodes correctly")
    func encodeDecodeRoundTrip() throws {
        let original = UMAPConfiguration(
            targetDimension: 3,
            neighbors: 20,
            minDistance: 0.2,
            iterations: 300,
            learningRate: 0.5,
            negativeSampleRate: 10,
            spread: 1.5
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(original)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(UMAPConfiguration.self, from: data)

        #expect(decoded == original)
    }
}

// MARK: - UMAP Projection Tests

@Suite("AccelerationManager - UMAP Projection")
struct UMAPProjectionTests {

    /// Generate random embeddings for testing
    private func generateRandomEmbeddings(count: Int, dimension: Int) -> [[Float]] {
        (0..<count).map { _ in
            (0..<dimension).map { _ in Float.random(in: -1...1) }
        }
    }

    @Test("UMAP projection returns correct 2D output dimensions")
    func projection2DOutputDimensions() async throws {
        let manager = try await AccelerationManager.create()
        let embeddings = generateRandomEmbeddings(count: 50, dimension: 128)

        let projected = try await manager.umapProject(
            embeddings: embeddings,
            config: .quickPreview(dimension: 2)
        )

        #expect(projected.count == 50)
        for point in projected {
            #expect(point.count == 2)
        }
    }

    @Test("UMAP projection returns correct 3D output dimensions")
    func projection3DOutputDimensions() async throws {
        let manager = try await AccelerationManager.create()
        let embeddings = generateRandomEmbeddings(count: 50, dimension: 128)

        let projected = try await manager.umapProject(
            embeddings: embeddings,
            config: .quickPreview(dimension: 3)
        )

        #expect(projected.count == 50)
        for point in projected {
            #expect(point.count == 3)
        }
    }

    @Test("UMAP with small batch (minimum viable)")
    func projectionSmallBatch() async throws {
        let manager = try await AccelerationManager.create()
        // Minimum: neighbors + 1 = 11 points for quickPreview (k=10)
        let embeddings = generateRandomEmbeddings(count: 15, dimension: 64)

        let projected = try await manager.umapProject(
            embeddings: embeddings,
            config: .quickPreview()
        )

        #expect(projected.count == 15)
    }

    @Test("UMAP with medium batch")
    func projectionMediumBatch() async throws {
        let manager = try await AccelerationManager.create()
        let embeddings = generateRandomEmbeddings(count: 100, dimension: 256)

        let projected = try await manager.umapProject(
            embeddings: embeddings,
            config: .quickPreview()
        )

        #expect(projected.count == 100)
    }

    @Test("Empty input returns empty array")
    func emptyInputReturnsEmpty() async throws {
        let manager = try await AccelerationManager.create()
        let embeddings: [[Float]] = []

        let projected = try await manager.umapProject(
            embeddings: embeddings,
            config: .default
        )

        #expect(projected.isEmpty)
    }

    @Test("Dimension mismatch throws error")
    func dimensionMismatchThrows() async throws {
        let manager = try await AccelerationManager.create()

        let embeddings: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0],  // Wrong dimension
            [7.0, 8.0, 9.0]
        ]

        await #expect(throws: AccelerationError.self) {
            try await manager.umapProject(embeddings: embeddings, config: .default)
        }
    }

    @Test("Insufficient points for neighbors throws error")
    func insufficientPointsThrows() async throws {
        let manager = try await AccelerationManager.create()

        // Only 5 points but default config needs 15 neighbors + 1 = 16 points
        let embeddings = generateRandomEmbeddings(count: 5, dimension: 64)

        await #expect(throws: AccelerationError.self) {
            try await manager.umapProject(embeddings: embeddings, config: .default)
        }
    }

    @Test("NaN in embeddings throws error")
    func nanInEmbeddingsThrows() async throws {
        let manager = try await AccelerationManager.create()

        let embeddings: [[Float]] = [
            [1.0, 2.0, 3.0],
            [Float.nan, 5.0, 6.0],  // Contains NaN
            [7.0, 8.0, 9.0]
        ]

        await #expect(throws: AccelerationError.self) {
            try await manager.umapProject(embeddings: embeddings, config: .quickPreview())
        }
    }

    @Test("Infinity in embeddings throws error")
    func infinityInEmbeddingsThrows() async throws {
        let manager = try await AccelerationManager.create()

        let embeddings: [[Float]] = [
            [1.0, 2.0, 3.0],
            [Float.infinity, 5.0, 6.0],  // Contains infinity
            [7.0, 8.0, 9.0]
        ]

        await #expect(throws: AccelerationError.self) {
            try await manager.umapProject(embeddings: embeddings, config: .quickPreview())
        }
    }

    @Test("Custom configuration works")
    func customConfigurationWorks() async throws {
        let manager = try await AccelerationManager.create()
        let embeddings = generateRandomEmbeddings(count: 30, dimension: 64)

        let config = UMAPConfiguration(
            targetDimension: 2,
            neighbors: 10,
            minDistance: 0.2,
            iterations: 50,
            learningRate: 0.8,
            negativeSampleRate: 3,
            spread: 0.8
        )

        let projected = try await manager.umapProject(
            embeddings: embeddings,
            config: config
        )

        #expect(projected.count == 30)
        for point in projected {
            #expect(point.count == 2)
        }
    }

    @Test("Projection output contains finite values")
    func projectionOutputIsFinite() async throws {
        let manager = try await AccelerationManager.create()
        let embeddings = generateRandomEmbeddings(count: 30, dimension: 64)

        let projected = try await manager.umapProject(
            embeddings: embeddings,
            config: .quickPreview()
        )

        for point in projected {
            for value in point {
                #expect(value.isFinite, "Projection output contains non-finite value: \(value)")
            }
        }
    }

    @Test("GPU operations are tracked for UMAP")
    func gpuOperationsTracked() async throws {
        let manager = try await AccelerationManager.create()
        await manager.resetStatistics()

        let embeddings = generateRandomEmbeddings(count: 20, dimension: 64)

        _ = try await manager.umapProject(
            embeddings: embeddings,
            config: .quickPreview()
        )

        let stats = await manager.statistics()
        #expect(stats.gpuOperations > 0)
        #expect(stats.gpuTimeTotal > 0)
    }
}

// MARK: - Convenience API UMAP Tests

@Suite("ModelManager - UMAP Convenience API")
struct ModelManagerUMAPTests {

    /// Generate mock embeddings for testing
    private func generateMockEmbeddings(count: Int, dimension: Int = 384) -> [Embedding] {
        (0..<count).map { _ in
            let vector = (0..<dimension).map { _ in Float.random(in: -1...1) }
            return Embedding(
                vector: vector,
                metadata: EmbeddingMetadata(
                    modelID: ModelID(provider: "test", name: "mock", version: "1.0"),
                    tokenCount: 10,
                    processingTime: 0.001,
                    normalized: true,
                    poolingStrategy: .mean,
                    custom: [:]
                )
            )
        }
    }

    @Test("projectEmbeddings returns correct 2D dimensions")
    func projectEmbeddings2D() async throws {
        let manager = ModelManager()
        let embeddings = generateMockEmbeddings(count: 30)

        let projected = try await manager.projectEmbeddings(embeddings, dimensions: 2)

        #expect(projected.count == 30)
        for point in projected {
            #expect(point.count == 2)
        }
    }

    @Test("projectEmbeddings returns correct 3D dimensions")
    func projectEmbeddings3D() async throws {
        let manager = ModelManager()
        let embeddings = generateMockEmbeddings(count: 30)

        let projected = try await manager.projectEmbeddings(embeddings, dimensions: 3)

        #expect(projected.count == 30)
        for point in projected {
            #expect(point.count == 3)
        }
    }

    @Test("projectEmbeddings with empty input returns empty")
    func projectEmbeddingsEmptyInput() async throws {
        let manager = ModelManager()
        let embeddings: [Embedding] = []

        let projected = try await manager.projectEmbeddings(embeddings, dimensions: 2)

        #expect(projected.isEmpty)
    }

    @Test("projectEmbeddings with custom dimension uses custom config")
    func projectEmbeddingsCustomDimension() async throws {
        let manager = ModelManager()
        let embeddings = generateMockEmbeddings(count: 30)

        // Use dimension 4 which should create a custom UMAPConfiguration
        let projected = try await manager.projectEmbeddings(embeddings, dimensions: 4)

        #expect(projected.count == 30)
        for point in projected {
            #expect(point.count == 4)
        }
    }
}

// MARK: - Configuration Factory Tests

@Suite("EmbeddingConfiguration - Visualization")
struct EmbeddingConfigurationVisualizationTests {

    @Test("forVisualization creates appropriate config")
    func forVisualizationConfig() {
        let config = EmbeddingConfiguration.forVisualization()

        #expect(config.maxTokens == 128)
        #expect(config.truncationStrategy == .end)
        #expect(config.poolingStrategy == .mean)
        #expect(config.normalizeOutput == true)
    }

    @Test("forVisualization with custom maxLength")
    func forVisualizationCustomMaxLength() {
        let config = EmbeddingConfiguration.forVisualization(maxLength: 256)

        #expect(config.maxTokens == 256)
    }

    @Test("UseCase.visualization exists and works with forMiniLM")
    func useCaseVisualizationWithMiniLM() {
        let config = EmbeddingConfiguration.forMiniLM(useCase: .visualization)

        #expect(config.maxTokens == 128)
        #expect(config.poolingStrategy == .mean)
    }

    @Test("UseCase.visualization exists and works with forBERT")
    func useCaseVisualizationWithBERT() {
        let config = EmbeddingConfiguration.forBERT(useCase: .visualization)

        #expect(config.maxTokens == 128)
        #expect(config.poolingStrategy == .mean)
    }
}
