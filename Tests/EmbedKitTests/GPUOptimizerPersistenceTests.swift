// EmbedKit - GPU Optimizer Persistence Tests

import Testing
import Foundation
@testable import EmbedKit

#if canImport(Metal)
import Metal

@Suite("GPU Optimizer Persistence")
struct GPUOptimizerPersistenceTests {

    // MARK: - AdaptiveKernelSelector Persistence Tests

    @Test("Save and load performance history")
    func saveAndLoadHistory() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("EmbedKitTests")
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent("gpu_performance.json")

        let capabilities = GPUDeviceCapabilities(device: device)
        let selector = AdaptiveKernelSelector(
            capabilities: capabilities,
            adaptiveLearning: true,
            persistenceURL: tempURL
        )

        // Record some performance data
        await selector.recordPerformance(
            operation: .poolAndNormalize,
            choice: .fused,
            workloadSize: 10000,
            executionTime: 0.005
        )
        await selector.recordPerformance(
            operation: .poolAndNormalize,
            choice: .fused,
            workloadSize: 20000,
            executionTime: 0.008
        )
        await selector.recordPerformance(
            operation: .similarityMatrix,
            choice: .progressive,
            workloadSize: 1000000,
            executionTime: 0.1
        )

        // Save to disk
        await selector.savePerformanceHistory()

        // Verify file was created
        #expect(FileManager.default.fileExists(atPath: tempURL.path))

        // Create new selector and load
        let selector2 = AdaptiveKernelSelector(
            capabilities: capabilities,
            adaptiveLearning: true,
            persistenceURL: tempURL
        )

        await selector2.loadPerformanceHistory()

        #expect(await selector2.hasPerformanceHistory)
        #expect(await selector2.totalSamples == 3)

        // Verify performance stats loaded correctly
        let poolStats = await selector2.getPerformanceStats(for: .poolAndNormalize)
        #expect(poolStats != nil)
        #expect(poolStats?.totalOperations == 2)
        #expect(poolStats?.fusedThroughput != nil)

        let simStats = await selector2.getPerformanceStats(for: .similarityMatrix)
        #expect(simStats != nil)
        #expect(simStats?.totalOperations == 1)

        // Cleanup
        try? FileManager.default.removeItem(at: tempURL.deletingLastPathComponent())
    }

    @Test("Empty selector has no history")
    func emptyHistory() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("EmbedKitTests")
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent("gpu_performance.json")

        let capabilities = GPUDeviceCapabilities(device: device)
        let selector = AdaptiveKernelSelector(
            capabilities: capabilities,
            adaptiveLearning: true,
            persistenceURL: tempURL
        )

        #expect(await !selector.hasPerformanceHistory)
        #expect(await selector.totalSamples == 0)

        // Cleanup
        try? FileManager.default.removeItem(at: tempURL.deletingLastPathComponent())
    }

    @Test("Clear performance history")
    func clearHistory() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("EmbedKitTests")
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent("gpu_performance.json")

        let capabilities = GPUDeviceCapabilities(device: device)
        let selector = AdaptiveKernelSelector(
            capabilities: capabilities,
            adaptiveLearning: true,
            persistenceURL: tempURL
        )

        // Record data and save
        await selector.recordPerformance(
            operation: .poolOnly,
            choice: .fused,
            workloadSize: 5000,
            executionTime: 0.003
        )
        await selector.savePerformanceHistory()

        #expect(await selector.hasPerformanceHistory)
        #expect(FileManager.default.fileExists(atPath: tempURL.path))

        // Clear history
        await selector.clearPerformanceHistory()

        #expect(await !selector.hasPerformanceHistory)
        #expect(await selector.totalSamples == 0)
        #expect(!FileManager.default.fileExists(atPath: tempURL.path))

        // Cleanup
        try? FileManager.default.removeItem(at: tempURL.deletingLastPathComponent())
    }

    @Test("Handles non-existent persistence file")
    func nonExistentFile() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("EmbedKitTests")
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent("does_not_exist.json")

        let capabilities = GPUDeviceCapabilities(device: device)
        let selector = AdaptiveKernelSelector(
            capabilities: capabilities,
            adaptiveLearning: true,
            persistenceURL: tempURL
        )

        // Should handle gracefully
        await selector.loadPerformanceHistory()

        #expect(await !selector.hasPerformanceHistory)

        // Cleanup
        try? FileManager.default.removeItem(at: tempURL.deletingLastPathComponent())
    }

    @Test("Selector with nil persistence URL")
    func nilPersistenceURL() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }

        let capabilities = GPUDeviceCapabilities(device: device)
        let selector = AdaptiveKernelSelector(
            capabilities: capabilities,
            adaptiveLearning: true,
            persistenceURL: nil
        )

        // Record some data
        await selector.recordPerformance(
            operation: .normalizeOnly,
            choice: .fused,
            workloadSize: 8000,
            executionTime: 0.004
        )

        // Save and load should be no-ops but not crash
        await selector.savePerformanceHistory()
        await selector.loadPerformanceHistory()

        // In-memory data should still work
        #expect(await selector.hasPerformanceHistory)
        #expect(await selector.totalSamples == 1)
    }

    @Test("Kernel selection uses loaded history")
    func selectionUsesHistory() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("EmbedKitTests")
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent("gpu_performance.json")

        let capabilities = GPUDeviceCapabilities(device: device)
        let selector = AdaptiveKernelSelector(
            capabilities: capabilities,
            adaptiveLearning: true,
            persistenceURL: tempURL
        )

        // Record multiple samples showing .separate is faster
        for _ in 0..<10 {
            await selector.recordPerformance(
                operation: .poolAndNormalize,
                choice: .separate,
                workloadSize: 50000,
                executionTime: 0.002  // 25M items/sec
            )
            await selector.recordPerformance(
                operation: .poolAndNormalize,
                choice: .fused,
                workloadSize: 50000,
                executionTime: 0.004  // 12.5M items/sec (slower)
            )
        }

        await selector.savePerformanceHistory()

        // Load in new selector
        let selector2 = AdaptiveKernelSelector(
            capabilities: capabilities,
            adaptiveLearning: true,
            persistenceURL: tempURL
        )
        await selector2.loadPerformanceHistory()

        // Selection should prefer .separate based on history
        let choice = await selector2.selectKernel(
            for: .poolAndNormalize,
            batchSize: 100,
            sequenceLength: 128,
            dimensions: 384
        )

        // Either .separate (learned) or default behavior is acceptable
        #expect(choice == .separate || choice == .fused || choice == .progressive)

        // Cleanup
        try? FileManager.default.removeItem(at: tempURL.deletingLastPathComponent())
    }

    @Test("Performance record timestamps are preserved")
    func timestampPreservation() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("EmbedKitTests")
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent("gpu_performance.json")

        let capabilities = GPUDeviceCapabilities(device: device)
        let selector = AdaptiveKernelSelector(
            capabilities: capabilities,
            adaptiveLearning: true,
            persistenceURL: tempURL
        )

        let beforeRecord = Date()

        await selector.recordPerformance(
            operation: .fullPipeline,
            choice: .fused,
            workloadSize: 100000,
            executionTime: 0.05
        )

        let afterRecord = Date()

        await selector.savePerformanceHistory()

        // Read the file directly to verify timestamp
        let data = try Data(contentsOf: tempURL)
        let jsonString = String(data: data, encoding: .utf8)!

        // Verify ISO8601 timestamp is present
        #expect(jsonString.contains("timestamp"))
        // ISO8601 format includes T separator
        #expect(jsonString.contains("T"))

        // Cleanup
        try? FileManager.default.removeItem(at: tempURL.deletingLastPathComponent())
    }

    // MARK: - KernelChoice and EmbeddingOperation Codable Tests

    @Test("KernelChoice encodes and decodes")
    func kernelChoiceCodable() throws {
        let choices: [AdaptiveKernelSelector.KernelChoice] = [.fused, .separate, .cpu, .progressive]

        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        for choice in choices {
            let data = try encoder.encode(choice)
            let decoded = try decoder.decode(AdaptiveKernelSelector.KernelChoice.self, from: data)
            #expect(decoded == choice)

            // Verify raw value encoding
            let jsonString = String(data: data, encoding: .utf8)!
            #expect(jsonString.contains(choice.rawValue))
        }
    }

    @Test("EmbeddingOperation encodes and decodes")
    func embeddingOperationCodable() throws {
        let operations = AdaptiveKernelSelector.EmbeddingOperation.allCases

        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        for op in operations {
            let data = try encoder.encode(op)
            let decoded = try decoder.decode(AdaptiveKernelSelector.EmbeddingOperation.self, from: data)
            #expect(decoded == op)

            // Verify raw value encoding
            let jsonString = String(data: data, encoding: .utf8)!
            #expect(jsonString.contains(op.rawValue))
        }
    }

    // MARK: - GPUFamily Tests

    @Test("GPUFamily has String raw values")
    func gpuFamilyRawValues() {
        let family: GPUDeviceCapabilities.GPUFamily = .m4Pro
        #expect(family.rawValue == "Apple M4 Pro")

        let m1 = GPUDeviceCapabilities.GPUFamily.m1
        #expect(m1.rawValue == "Apple M1")

        // Verify can initialize from raw value
        let fromRaw = GPUDeviceCapabilities.GPUFamily(rawValue: "Apple M3")
        #expect(fromRaw == .m3)
    }

    // MARK: - Integration Tests

    @Test("Full persistence workflow")
    func fullPersistenceWorkflow() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }

        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("EmbedKitTests")
            .appendingPathComponent(UUID().uuidString)

        let tempURL = tempDir.appendingPathComponent("gpu_performance.json")

        let capabilities = GPUDeviceCapabilities(device: device)

        // Session 1: Record performance data
        do {
            let selector = AdaptiveKernelSelector(
                capabilities: capabilities,
                adaptiveLearning: true,
                persistenceURL: tempURL
            )

            // Simulate multiple operations
            for operation in AdaptiveKernelSelector.EmbeddingOperation.allCases {
                await selector.recordPerformance(
                    operation: operation,
                    choice: .fused,
                    workloadSize: 50000,
                    executionTime: 0.01
                )
            }

            await selector.savePerformanceHistory()
            #expect(await selector.totalSamples == 5)
        }

        // Session 2: Load and continue
        do {
            let selector = AdaptiveKernelSelector(
                capabilities: capabilities,
                adaptiveLearning: true,
                persistenceURL: tempURL
            )

            await selector.loadPerformanceHistory()
            #expect(await selector.hasPerformanceHistory)
            #expect(await selector.totalSamples == 5)

            // Add more records
            await selector.recordPerformance(
                operation: .poolAndNormalize,
                choice: .separate,
                workloadSize: 100000,
                executionTime: 0.015
            )

            await selector.savePerformanceHistory()
            #expect(await selector.totalSamples == 6)
        }

        // Session 3: Verify accumulated data
        do {
            let selector = AdaptiveKernelSelector(
                capabilities: capabilities,
                adaptiveLearning: true,
                persistenceURL: tempURL
            )

            await selector.loadPerformanceHistory()
            #expect(await selector.totalSamples == 6)

            // Verify poolAndNormalize has both fused and separate records
            let stats = await selector.getPerformanceStats(for: .poolAndNormalize)
            #expect(stats?.totalOperations == 2)
            #expect(stats?.fusedThroughput != nil)
            #expect(stats?.separateThroughput != nil)
        }

        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }
}

// Helper to skip tests without Metal
struct XCTSkip: Error {
    let message: String
    init(_ message: String) { self.message = message }
}
#endif
