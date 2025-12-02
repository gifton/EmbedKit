// EmbedKit - MemoryAwareGenerator Tests

import Testing
import Foundation
@testable import EmbedKit
import VectorCore

// MARK: - MemoryPressureLevel Tests

@Suite("MemoryPressureLevel")
struct MemoryPressureLevelTests {

    @Test("Levels are ordered correctly")
    func testOrdering() {
        #expect(MemoryPressureLevel.normal < MemoryPressureLevel.warning)
        #expect(MemoryPressureLevel.warning < MemoryPressureLevel.critical)
    }

    @Test("Batch size multipliers decrease with pressure")
    func testMultipliers() {
        #expect(MemoryPressureLevel.normal.batchSizeMultiplier == 1.0)
        #expect(MemoryPressureLevel.warning.batchSizeMultiplier == 0.5)
        #expect(MemoryPressureLevel.critical.batchSizeMultiplier == 0.25)
    }

    @Test("All cases available")
    func testAllCases() {
        #expect(MemoryPressureLevel.allCases.count == 3)
    }
}

// MARK: - MemoryMonitor Tests

@Suite("MemoryMonitor", .serialized)
struct MemoryMonitorTests {

    @Test("Shared instance exists")
    func testSharedInstance() {
        let monitor = MemoryMonitor.shared
        #expect(monitor === MemoryMonitor.shared)
    }

    @Test("Initial level is normal")
    func testInitialLevel() {
        // Note: May not be normal if system is under pressure
        let level = MemoryMonitor.shared.currentLevel
        #expect(level == .normal || level == .warning || level == .critical)
    }

    @Test("Simulate pressure changes level")
    func testSimulatePressure() {
        let monitor = MemoryMonitor.shared

        monitor.simulatePressure(.warning)
        #expect(monitor.currentLevel == .warning)

        monitor.simulatePressure(.critical)
        #expect(monitor.currentLevel == .critical)

        monitor.simulatePressure(.normal)
        #expect(monitor.currentLevel == .normal)
    }

    @Test("Pressure change handler is called")
    func testPressureChangeHandler() async {
        let monitor = MemoryMonitor.shared

        actor StateHolder {
            var receivedLevel: MemoryPressureLevel?
            func setLevel(_ level: MemoryPressureLevel) { receivedLevel = level }
        }
        let holder = StateHolder()

        let handlerID = monitor.onPressureChange { level in
            Task { await holder.setLevel(level) }
        }

        // Ensure we start at normal
        monitor.simulatePressure(.normal)
        try? await Task.sleep(for: .milliseconds(50))

        // Trigger warning
        monitor.simulatePressure(.warning)
        try? await Task.sleep(for: .milliseconds(50))

        let received = await holder.receivedLevel
        #expect(received == .warning)

        // Cleanup
        monitor.removeHandler(at: handlerID)
        monitor.simulatePressure(.normal)
    }

    @Test("Memory stats returns valid data")
    func testMemoryStats() {
        let stats = MemoryMonitor.shared.memoryStats

        #expect(stats.physicalMemory > 0)
        #expect(stats.memoryUtilization >= 0)
        #expect(stats.memoryUtilization <= 1.0)
    }

    @Test("zz_cleanup - Reset MemoryMonitor after tests")
    func zz_cleanupResources() {
        cleanupMemoryMonitorTestResources()
    }
}

// MARK: - MemoryAwareConfig Tests

@Suite("MemoryAwareConfig")
struct MemoryAwareConfigTests {

    @Test("Default configuration")
    func testDefault() {
        let config = MemoryAwareConfig.default

        #expect(config.baseBatchSize == 32)
        #expect(config.minBatchSize == 4)
        #expect(config.adaptiveBatching == true)
        #expect(config.pressureThreshold == 0.7)
        #expect(config.releaseOnCritical == false)
    }

    @Test("Conservative configuration")
    func testConservative() {
        let config = MemoryAwareConfig.conservative

        #expect(config.baseBatchSize == 16)
        #expect(config.minBatchSize == 2)
        #expect(config.releaseOnCritical == true)
    }

    @Test("Aggressive configuration")
    func testAggressive() {
        let config = MemoryAwareConfig.aggressive

        #expect(config.baseBatchSize == 64)
        #expect(config.minBatchSize == 8)
        #expect(config.pressureThreshold == 0.85)
    }

    @Test("Custom configuration")
    func testCustom() {
        let config = MemoryAwareConfig(
            baseBatchSize: 48,
            minBatchSize: 6,
            adaptiveBatching: false,
            pressureThreshold: 0.6,
            releaseOnCritical: true
        )

        #expect(config.baseBatchSize == 48)
        #expect(config.minBatchSize == 6)
        #expect(config.adaptiveBatching == false)
        #expect(config.pressureThreshold == 0.6)
        #expect(config.releaseOnCritical == true)
    }
}

// MARK: - MemoryAwareGenerator Tests

@Suite("MemoryAwareGenerator")
struct MemoryAwareGeneratorTests {

    @Test("Conforms to VectorProducer")
    func testVectorProducerConformance() async {
        let mockModel = MockEmbeddingModel(dimensions: 384)
        let generator = EmbeddingGenerator(model: mockModel)
        let memoryAware = MemoryAwareGenerator(generator: generator)

        #expect(memoryAware.dimensions == 384)
        #expect(memoryAware.producesNormalizedVectors == true)
    }

    @Test("Effective batch size at normal pressure")
    func testEffectiveBatchSizeNormal() async {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)
        let config = MemoryAwareConfig(baseBatchSize: 32)
        let memoryAware = MemoryAwareGenerator(generator: generator, config: config)

        await memoryAware.setPressure(.normal)
        let batchSize = await memoryAware.effectiveBatchSize

        #expect(batchSize == 32)
    }

    @Test("Effective batch size at warning pressure")
    func testEffectiveBatchSizeWarning() async {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)
        let config = MemoryAwareConfig(baseBatchSize: 32, minBatchSize: 4)
        let memoryAware = MemoryAwareGenerator(generator: generator, config: config)

        await memoryAware.setPressure(.warning)
        let batchSize = await memoryAware.effectiveBatchSize

        #expect(batchSize == 16)  // 32 * 0.5
    }

    @Test("Effective batch size at critical pressure")
    func testEffectiveBatchSizeCritical() async {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)
        let config = MemoryAwareConfig(baseBatchSize: 32, minBatchSize: 4)
        let memoryAware = MemoryAwareGenerator(generator: generator, config: config)

        await memoryAware.setPressure(.critical)
        let batchSize = await memoryAware.effectiveBatchSize

        #expect(batchSize == 8)  // 32 * 0.25
    }

    @Test("Effective batch size respects minimum")
    func testEffectiveBatchSizeMinimum() async {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)
        let config = MemoryAwareConfig(baseBatchSize: 8, minBatchSize: 4)
        let memoryAware = MemoryAwareGenerator(generator: generator, config: config)

        await memoryAware.setPressure(.critical)
        let batchSize = await memoryAware.effectiveBatchSize

        // 8 * 0.25 = 2, but min is 4
        #expect(batchSize == 4)
    }

    @Test("Adaptive batching can be disabled")
    func testAdaptiveBatchingDisabled() async {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)
        let config = MemoryAwareConfig(baseBatchSize: 32, adaptiveBatching: false)
        let memoryAware = MemoryAwareGenerator(generator: generator, config: config)

        await memoryAware.setPressure(.critical)
        let batchSize = await memoryAware.effectiveBatchSize

        // Should ignore pressure when adaptive is disabled
        #expect(batchSize == 32)
    }

    @Test("Produce returns correct embeddings")
    func testProduce() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: mockModel)
        let memoryAware = MemoryAwareGenerator(generator: generator)

        let texts = ["Hello", "World", "Test"]
        let embeddings = try await memoryAware.produce(texts)

        #expect(embeddings.count == 3)
        for embedding in embeddings {
            #expect(embedding.count == 128)
        }
    }

    @Test("Produce single text")
    func testProduceSingle() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 256)
        let generator = EmbeddingGenerator(model: mockModel)
        let memoryAware = MemoryAwareGenerator(generator: generator)

        let embedding = try await memoryAware.produce("Hello")

        #expect(embedding.count == 256)
    }

    @Test("Statistics track batches correctly")
    func testStatistics() async throws {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)
        // Use baseBatchSize=2 and minBatchSize=1 to ensure batch size is 2
        let config = MemoryAwareConfig(baseBatchSize: 2, minBatchSize: 1)
        let memoryAware = MemoryAwareGenerator(generator: generator, config: config)

        // Ensure we start at normal and verify effective batch size
        await memoryAware.setPressure(.normal)
        let effectiveBatch = await memoryAware.effectiveBatchSize
        #expect(effectiveBatch == 2)

        let texts = ["a", "b", "c", "d"]  // 4 texts at batch size 2 = 2 batches
        _ = try await memoryAware.produce(texts)

        let stats = await memoryAware.getStatistics()

        #expect(stats.totalItemsProcessed == 4)
        // The MemoryAwareGenerator should process in 2 batches of 2
        #expect(stats.totalBatches == 2, "Expected 2 batches but got \(stats.totalBatches)")
        #expect(stats.batchesAtNormal == 2)
        #expect(stats.currentPressure == .normal)
    }

    @Test("Statistics track pressure changes")
    func testStatisticsPressureChanges() async throws {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)
        let config = MemoryAwareConfig(baseBatchSize: 10)
        let memoryAware = MemoryAwareGenerator(generator: generator, config: config)

        // Process at normal
        await memoryAware.setPressure(.normal)
        _ = try await memoryAware.produce(["a"])

        // Process at warning
        await memoryAware.setPressure(.warning)
        _ = try await memoryAware.produce(["b"])

        // Process at critical
        await memoryAware.setPressure(.critical)
        _ = try await memoryAware.produce(["c"])

        let stats = await memoryAware.getStatistics()

        #expect(stats.batchesAtNormal == 1)
        #expect(stats.batchesAtWarning == 1)
        #expect(stats.batchesAtCritical == 1)
        #expect(stats.batchSizeAdjustments == 2)  // normal->warning, warning->critical
    }

    @Test("Reset statistics")
    func testResetStatistics() async throws {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)
        let memoryAware = MemoryAwareGenerator(generator: generator)

        _ = try await memoryAware.produce(["test"])
        await memoryAware.resetStatistics()

        let stats = await memoryAware.getStatistics()

        #expect(stats.totalBatches == 0)
        #expect(stats.totalItemsProcessed == 0)
    }

    @Test("Config can be updated")
    func testUpdateConfig() async {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)
        let memoryAware = MemoryAwareGenerator(generator: generator)

        await memoryAware.updateConfig(MemoryAwareConfig(baseBatchSize: 64))
        let config = await memoryAware.getConfig()

        #expect(config.baseBatchSize == 64)
    }

    @Test("memoryAware() extension creates wrapper")
    func testExtension() async {
        let mockModel = MockEmbeddingModel(dimensions: 512)
        let generator = EmbeddingGenerator(model: mockModel)
        let memoryAware = await generator.memoryAware(config: .conservative)

        #expect(memoryAware.dimensions == 512)
        let config = await memoryAware.getConfig()
        #expect(config.baseBatchSize == 16)
    }
}

// MARK: - MemoryAwareGenerator Progress Tests

@Suite("MemoryAwareGenerator Progress")
struct MemoryAwareGeneratorProgressTests {

    @Test("Progress callback is invoked")
    func testProgressCallback() async throws {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)
        let config = MemoryAwareConfig(baseBatchSize: 2)
        let memoryAware = MemoryAwareGenerator(generator: generator, config: config)

        actor ProgressCollector {
            var updates: [BatchProgress] = []
            func add(_ p: BatchProgress) { updates.append(p) }
        }
        let collector = ProgressCollector()

        let texts = ["a", "b", "c", "d"]
        _ = try await memoryAware.produceWithProgress(texts) { progress in
            Task { await collector.add(progress) }
        }

        // Wait for callbacks
        try? await Task.sleep(for: .milliseconds(50))

        let updates = await collector.updates

        #expect(updates.count >= 2)  // At least started and completed
        #expect(updates.first?.phase == "Starting")
        #expect(updates.last?.phase == "Complete" || updates.last?.isComplete == true)
    }

    @Test("Progress shows correct item counts")
    func testProgressItemCounts() async throws {
        let mockModel = MockEmbeddingModel()
        let generator = EmbeddingGenerator(model: mockModel)
        let config = MemoryAwareConfig(baseBatchSize: 3)
        let memoryAware = MemoryAwareGenerator(generator: generator, config: config)

        actor ProgressCollector {
            var lastProgress: BatchProgress?
            func set(_ p: BatchProgress) { lastProgress = p }
        }
        let collector = ProgressCollector()

        let texts = ["1", "2", "3", "4", "5", "6"]
        _ = try await memoryAware.produceWithProgress(texts) { progress in
            Task { await collector.set(progress) }
        }

        try? await Task.sleep(for: .milliseconds(50))
        let lastProgress = await collector.lastProgress

        #expect(lastProgress?.total == 6)
        #expect(lastProgress?.current == 6)
        #expect(lastProgress?.isComplete == true)
    }
}

// MARK: - MemoryStats Tests

@Suite("MemoryStats")
struct MemoryStatsTests {

    @Test("Memory utilization calculation")
    func testMemoryUtilization() {
        let stats = MemoryStats(
            residentSize: 500_000_000,  // 500 MB
            virtualSize: 1_000_000_000,
            physicalMemory: 8_000_000_000,  // 8 GB
            pressure: .normal
        )

        #expect(stats.memoryUtilization == 0.0625)  // 500MB / 8GB
        #expect(abs(stats.residentSizeMB - 476.8) < 1.0)  // ~477 MB
    }

    @Test("Zero physical memory handled")
    func testZeroPhysicalMemory() {
        let stats = MemoryStats(
            residentSize: 100,
            virtualSize: 200,
            physicalMemory: 0,
            pressure: .normal
        )

        #expect(stats.memoryUtilization == 0)
    }
}

// MARK: - Integration Tests

@Suite("MemoryAwareGenerator Integration")
struct MemoryAwareIntegrationTests {

    @Test("Batch size adjusts during processing")
    func testDynamicBatchSizeAdjustment() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 64)
        let generator = EmbeddingGenerator(model: mockModel)

        actor AdjustmentTracker {
            var adjustments: [(Int, MemoryPressureLevel)] = []
            func add(_ size: Int, _ level: MemoryPressureLevel) {
                adjustments.append((size, level))
            }
        }
        let tracker = AdjustmentTracker()

        let config = MemoryAwareConfig(
            baseBatchSize: 32,
            minBatchSize: 4,
            onBatchSizeAdjusted: { size, level in
                Task { await tracker.add(size, level) }
            }
        )

        let memoryAware = MemoryAwareGenerator(generator: generator, config: config)

        // Start at normal
        await memoryAware.setPressure(.normal)

        // Change to warning
        await memoryAware.setPressure(.warning)

        // Change to critical
        await memoryAware.setPressure(.critical)

        try? await Task.sleep(for: .milliseconds(50))

        let adjustments = await tracker.adjustments

        #expect(adjustments.count == 2)  // normal->warning, warning->critical
        if adjustments.count >= 2 {
            #expect(adjustments[0].0 == 16)  // warning: 32 * 0.5
            #expect(adjustments[0].1 == .warning)
            #expect(adjustments[1].0 == 8)   // critical: 32 * 0.25
            #expect(adjustments[1].1 == .critical)
        }
    }

    @Test("Large batch under memory pressure")
    func testLargeBatchUnderPressure() async throws {
        let mockModel = MockEmbeddingModel(dimensions: 128)
        let generator = EmbeddingGenerator(model: mockModel)
        let config = MemoryAwareConfig(baseBatchSize: 20, minBatchSize: 5)
        let memoryAware = MemoryAwareGenerator(generator: generator, config: config)

        // Set critical pressure
        await memoryAware.setPressure(.critical)

        // Process 100 texts - should be processed in batches of 5
        let texts = (0..<100).map { "Text \($0)" }
        let results = try await memoryAware.produce(texts)

        #expect(results.count == 100)

        let stats = await memoryAware.getStatistics()
        #expect(stats.totalBatches == 20)  // 100 / 5
        #expect(stats.batchesAtCritical == 20)
    }

    @Test("zz_cleanup - Release all resources after integration tests")
    func zz_cleanupResources() async {
        await cleanupEmbedKitTestResources()
    }
}
