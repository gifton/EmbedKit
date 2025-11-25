// EmbedKit - GPU Optimizer Tests
//
// Tests for Phase 4 GPU optimization infrastructure:
// - GPUDeviceCapabilities
// - ThreadgroupOptimizer
// - AdaptiveKernelSelector
// - ProgressiveSimilarityComputer
// - BufferResidencyManager

import Testing
import Foundation
@testable import EmbedKit

#if canImport(Metal)
import Metal
#endif

// MARK: - GPU Device Capabilities Tests

@Suite("GPU Device Capabilities")
struct GPUDeviceCapabilitiesTests {

    #if canImport(Metal)
    @Test("Detects device capabilities")
    func detectsCapabilities() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalNotAvailable
        }

        let capabilities = GPUDeviceCapabilities(device: device)

        // Should have valid values
        #expect(capabilities.maxThreadsPerThreadgroup > 0)
        #expect(capabilities.maxThreadgroupMemory > 0)
        #expect(capabilities.maxBufferLength > 0)
        #expect(!capabilities.deviceName.isEmpty)
    }

    @Test("Identifies GPU family from device name")
    func identifiesGPUFamily() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalNotAvailable
        }

        let capabilities = GPUDeviceCapabilities(device: device)

        // Family should not be unknown for Apple devices
        if device.name.lowercased().contains("apple") ||
           device.name.lowercased().contains("m1") ||
           device.name.lowercased().contains("m2") ||
           device.name.lowercased().contains("m3") ||
           device.name.lowercased().contains("m4") {
            #expect(capabilities.family != .unknown)
        }

        // SIMD width should be valid
        #expect(capabilities.family.optimalSimdWidth > 0)
        #expect(capabilities.family.recommendedMaxThreads > 0)
    }

    @Test("Recommends threadgroup width based on dimensions")
    func recommendsThreadgroupWidth() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalNotAvailable
        }

        let capabilities = GPUDeviceCapabilities(device: device)

        // Small dimensions
        let small = capabilities.recommendedThreadgroupWidth(forDimensions: 32)
        #expect(small > 0)
        #expect(small <= 32)

        // Medium dimensions
        let medium = capabilities.recommendedThreadgroupWidth(forDimensions: 384)
        #expect(medium > 0)
        #expect(medium <= 384)

        // Large dimensions
        let large = capabilities.recommendedThreadgroupWidth(forDimensions: 1024)
        #expect(large > 0)
        #expect(large <= capabilities.maxThreadsPerThreadgroup)
    }
    #endif
}

// MARK: - Threadgroup Optimizer Tests

@Suite("Threadgroup Optimizer")
struct ThreadgroupOptimizerTests {

    #if canImport(Metal)
    @Test("Calculates optimal threadgroup for pooling")
    func optimalPoolingThreadgroup() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalNotAvailable
        }

        let capabilities = GPUDeviceCapabilities(device: device)
        let optimizer = ThreadgroupOptimizer(capabilities: capabilities)

        let (threadgroup, grid) = optimizer.optimalThreadgroup(
            for: .pooling,
            batchSize: 16,
            sequenceLength: 128,
            dimensions: 384
        )

        // Threadgroup should be valid
        #expect(threadgroup.width > 0)
        #expect(threadgroup.height >= 1)
        #expect(threadgroup.depth >= 1)

        // Grid should cover all work
        #expect(grid.width > 0)
        #expect(grid.height > 0)
    }

    @Test("Calculates optimal threadgroup for normalization")
    func optimalNormalizationThreadgroup() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalNotAvailable
        }

        let capabilities = GPUDeviceCapabilities(device: device)
        let optimizer = ThreadgroupOptimizer(capabilities: capabilities)

        let (threadgroup, grid) = optimizer.optimalThreadgroup(
            for: .normalization,
            batchSize: 32,
            dimensions: 384
        )

        // Normalization uses one threadgroup per vector
        #expect(grid.height == 32)  // One per batch item
        #expect(threadgroup.width > 0)
    }

    @Test("Calculates optimal threadgroup for similarity")
    func optimalSimilarityThreadgroup() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalNotAvailable
        }

        let capabilities = GPUDeviceCapabilities(device: device)
        let optimizer = ThreadgroupOptimizer(capabilities: capabilities)

        let (threadgroup, grid) = optimizer.optimalThreadgroup(
            for: .similarity,
            batchSize: 100,
            sequenceLength: 100,
            dimensions: 384
        )

        // 2D threadgroup for similarity matrix
        #expect(threadgroup.width > 0)
        #expect(threadgroup.height > 0)

        // Grid should cover the similarity matrix
        #expect(grid.width > 0)
        #expect(grid.height > 0)
    }

    @Test("Calculates optimal threadgroup for fused operations")
    func optimalFusedPoolNormThreadgroup() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalNotAvailable
        }

        let capabilities = GPUDeviceCapabilities(device: device)
        let optimizer = ThreadgroupOptimizer(capabilities: capabilities)

        let (threadgroup, grid) = optimizer.optimalThreadgroup(
            for: .fusedPoolNorm,
            batchSize: 8,
            sequenceLength: 128,
            dimensions: 384
        )

        // Fused ops use one threadgroup per sequence
        #expect(grid.height == 8)  // One per batch
        #expect(threadgroup.width > 0)
    }
    #endif
}

// MARK: - Adaptive Kernel Selector Tests

@Suite("Adaptive Kernel Selector")
struct AdaptiveKernelSelectorTests {

    #if canImport(Metal)
    @Test("Selects CPU for small workloads")
    func selectsCPUForSmallWorkloads() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalNotAvailable
        }

        let capabilities = GPUDeviceCapabilities(device: device)
        let selector = AdaptiveKernelSelector(capabilities: capabilities, adaptiveLearning: false)

        // Very small workload should use CPU
        let choice = await selector.selectKernel(
            for: .poolAndNormalize,
            batchSize: 1,
            sequenceLength: 10,
            dimensions: 32
        )

        #expect(choice == .cpu)
    }

    @Test("Selects fused for medium workloads")
    func selectsFusedForMediumWorkloads() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalNotAvailable
        }

        let capabilities = GPUDeviceCapabilities(device: device)
        let selector = AdaptiveKernelSelector(capabilities: capabilities, adaptiveLearning: false)

        // Medium workload should use fused kernel
        let choice = await selector.selectKernel(
            for: .poolAndNormalize,
            batchSize: 16,
            sequenceLength: 128,
            dimensions: 384
        )

        #expect(choice == .fused)
    }

    @Test("Records and learns from performance")
    func recordsPerformance() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalNotAvailable
        }

        let capabilities = GPUDeviceCapabilities(device: device)
        let selector = AdaptiveKernelSelector(capabilities: capabilities, adaptiveLearning: true)

        // Record some performance data
        await selector.recordPerformance(
            operation: .poolAndNormalize,
            choice: .fused,
            workloadSize: 100_000,
            executionTime: 0.01
        )

        await selector.recordPerformance(
            operation: .poolAndNormalize,
            choice: .fused,
            workloadSize: 100_000,
            executionTime: 0.012
        )

        // Check stats
        let stats = await selector.getPerformanceStats(for: .poolAndNormalize)
        #expect(stats != nil)
        #expect(stats?.totalOperations == 2)
        #expect(stats?.fusedThroughput != nil)
    }

    @Test("Selects progressive for large similarity matrices")
    func selectsProgressiveForLargeMatrices() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalNotAvailable
        }

        let capabilities = GPUDeviceCapabilities(device: device)
        let selector = AdaptiveKernelSelector(capabilities: capabilities, adaptiveLearning: false)

        // Very large similarity matrix
        let choice = await selector.selectKernel(
            for: .similarityMatrix,
            batchSize: 10000,
            sequenceLength: 1,
            dimensions: 384
        )

        #expect(choice == .progressive)
    }
    #endif
}

// MARK: - Progressive Similarity Computer Tests

@Suite("Progressive Similarity Computer")
struct ProgressiveSimilarityComputerTests {

    @Test("Single tile for small matrices")
    func singleTileForSmallMatrices() {
        let computer = ProgressiveSimilarityComputer()

        let tiles = computer.calculateTiles(
            queryBatchSize: 100,
            keyBatchSize: 100
        )

        // Small matrix should be single tile
        #expect(tiles.count == 1)
        #expect(tiles[0].queryStart == 0)
        #expect(tiles[0].queryEnd == 100)
        #expect(tiles[0].keyStart == 0)
        #expect(tiles[0].keyEnd == 100)
    }

    @Test("Multiple tiles for large matrices")
    func multipleTilesForLargeMatrices() {
        let config = ProgressiveSimilarityComputer.Configuration(
            maxTileElements: 10_000,
            preferredTileSize: 100
        )
        let computer = ProgressiveSimilarityComputer(configuration: config)

        let tiles = computer.calculateTiles(
            queryBatchSize: 500,
            keyBatchSize: 500
        )

        // Should have multiple tiles
        #expect(tiles.count > 1)

        // Tiles should cover entire matrix
        var coveredElements = 0
        for tile in tiles {
            coveredElements += tile.elementCount
        }
        #expect(coveredElements == 500 * 500)

        // Tiles should not overlap and should be contiguous
        let sortedTiles = tiles.sorted { ($0.queryStart, $0.keyStart) < ($1.queryStart, $1.keyStart) }
        for tile in sortedTiles {
            #expect(tile.queryEnd <= 500)
            #expect(tile.keyEnd <= 500)
        }
    }

    @Test("Tile counts match element coverage")
    func tileCountsMatchCoverage() {
        let config = ProgressiveSimilarityComputer.Configuration(
            maxTileElements: 50_000,
            preferredTileSize: 200
        )
        let computer = ProgressiveSimilarityComputer(configuration: config)

        let tiles = computer.calculateTiles(
            queryBatchSize: 1000,
            keyBatchSize: 800
        )

        var totalElements = 0
        for tile in tiles {
            totalElements += tile.queryCount * tile.keyCount
        }

        #expect(totalElements == 1000 * 800)
    }

    #if canImport(Metal)
    @Test("Device-specific configuration")
    func deviceSpecificConfiguration() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalNotAvailable
        }

        let capabilities = GPUDeviceCapabilities(device: device)
        let config = ProgressiveSimilarityComputer.Configuration.forDevice(capabilities)

        // Config should have reasonable values
        #expect(config.maxTileElements > 0)
        #expect(config.preferredTileSize > 0)

        // M3/M4 should have larger tiles
        if capabilities.family.generation >= 3 {
            #expect(config.preferredTileSize >= 1536)
        }
    }
    #endif
}

// MARK: - Buffer Residency Manager Tests

#if canImport(Metal)
@Suite("Buffer Residency Manager")
struct BufferResidencyManagerTests {

    @Test("Statistics struct has correct properties")
    func statisticsStructProperties() {
        let stats = BufferResidencyManager.ResidencyStatistics(
            residentBufferCount: 5,
            totalResidentBytes: 1024 * 1024,
            averageAccessCount: 3.5,
            maxResidentBytes: 10 * 1024 * 1024
        )

        #expect(stats.residentBufferCount == 5)
        #expect(stats.totalResidentBytes == 1024 * 1024)
        #expect(stats.averageAccessCount == 3.5)
        #expect(stats.utilizationPercent == 10.0)  // 1MB / 10MB * 100
    }

    @Test("Utilization percentage calculation")
    func utilizationCalculation() {
        // 50% utilization
        let stats50 = BufferResidencyManager.ResidencyStatistics(
            residentBufferCount: 1,
            totalResidentBytes: 500,
            averageAccessCount: 1.0,
            maxResidentBytes: 1000
        )
        #expect(stats50.utilizationPercent == 50.0)

        // 0% utilization (empty)
        let stats0 = BufferResidencyManager.ResidencyStatistics(
            residentBufferCount: 0,
            totalResidentBytes: 0,
            averageAccessCount: 0.0,
            maxResidentBytes: 1000
        )
        #expect(stats0.utilizationPercent == 0.0)

        // Edge case: 0 max bytes
        let statsEdge = BufferResidencyManager.ResidencyStatistics(
            residentBufferCount: 0,
            totalResidentBytes: 0,
            averageAccessCount: 0.0,
            maxResidentBytes: 0
        )
        #expect(statsEdge.utilizationPercent == 0.0)
    }

    @Test("Manager initializes with device")
    func managerInitializes() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalNotAvailable
        }

        let manager = BufferResidencyManager(device: device, maxResidentMB: 100)

        // Initial state should be empty
        let stats = await manager.getStatistics()
        #expect(stats.residentBufferCount == 0)
        #expect(stats.totalResidentBytes == 0)
        #expect(stats.maxResidentBytes == 100 * 1024 * 1024)
    }
}
#endif

// MARK: - GPU Optimizer Integration Tests

#if canImport(Metal)
@Suite("GPU Optimizer Integration")
struct GPUOptimizerIntegrationTests {

    @Test("Creates optimizer from device")
    func createsOptimizerFromDevice() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalNotAvailable
        }

        let optimizer = GPUOptimizer(device: device)

        // Should have valid capabilities
        let caps = await optimizer.capabilities
        #expect(caps.maxThreadsPerThreadgroup > 0)
    }

    @Test("Gets dispatch parameters")
    func getsDispatchParameters() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalNotAvailable
        }

        let optimizer = GPUOptimizer(device: device)

        let params = await optimizer.getDispatchParameters(
            operation: .fusedPoolNorm,
            batchSize: 16,
            sequenceLength: 128,
            dimensions: 384
        )

        #expect(params.threadgroupSize.width > 0)
        #expect(params.gridSize.height == 16)  // One per batch
    }

    @Test("Optimizes operation with kernel selection")
    func optimizesOperation() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalNotAvailable
        }

        let optimizer = GPUOptimizer(device: device)

        let optimized = await optimizer.optimizeOperation(
            .poolAndNormalize,
            batchSize: 16,
            sequenceLength: 128,
            dimensions: 384
        )

        #expect(optimized.kernelChoice == .fused)
        #expect(optimized.threadgroupSize.width > 0)
        #expect(optimized.tiles == nil)  // No progressive tiles for this size
    }

    @Test("Returns progressive tiles for large workloads")
    func returnsProgressiveTiles() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw TestError.metalNotAvailable
        }

        let optimizer = GPUOptimizer(device: device)

        let optimized = await optimizer.optimizeOperation(
            .similarityMatrix,
            batchSize: 10000,
            sequenceLength: 1,
            dimensions: 384
        )

        #expect(optimized.kernelChoice == .progressive)
        #expect(optimized.tiles != nil)
        #expect(optimized.tiles!.count > 1)
    }
}
#endif

// MARK: - Test Helpers

enum TestError: Error {
    case metalNotAvailable
    case bufferCreationFailed
}
