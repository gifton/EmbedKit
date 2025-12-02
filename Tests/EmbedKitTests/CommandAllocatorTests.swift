// EmbedKit - Command Allocator Tests
//
// Tests for Metal 4 command allocator pool and memory management.
// Validates pool behavior, acquire/release cycles, and statistics.

import Testing
import Foundation
@testable import EmbedKit

#if canImport(Metal)
import Metal
#endif

// MARK: - Test Error

enum CommandAllocatorTestError: Error {
    case skipped(String)
}

// MARK: - SimpleCommandAllocator Tests

@Suite("SimpleCommandAllocator Unit")
struct SimpleCommandAllocatorUnitTests {

    @Test("Initializes with specified heap size")
    func initializesWithHeapSize() {
        let allocator = SimpleCommandAllocator(heapSize: 32 * 1024 * 1024)

        #expect(allocator.heapSize == 32 * 1024 * 1024)
        #expect(allocator.isInUse == false)
    }

    @Test("Initializes with default heap size")
    func initializesWithDefaultHeapSize() {
        let allocator = SimpleCommandAllocator()

        #expect(allocator.heapSize == 16 * 1024 * 1024)
    }

    @Test("Tracks in-use state")
    func tracksInUseState() {
        let allocator = SimpleCommandAllocator()

        #expect(allocator.isInUse == false)

        allocator.markInUse()
        #expect(allocator.isInUse == true)

        allocator.reset()
        #expect(allocator.isInUse == false)
    }

    @Test("Reset clears in-use state")
    func resetClearsInUseState() {
        let allocator = SimpleCommandAllocator()

        allocator.markInUse()
        allocator.markInUse() // Multiple calls
        #expect(allocator.isInUse == true)

        allocator.reset()
        #expect(allocator.isInUse == false)
    }
}

// MARK: - CommandAllocatorPool Configuration Tests

@Suite("CommandAllocatorPool - Configuration")
struct CommandAllocatorPoolConfigurationTests {

    @Test("Default configuration has expected values")
    func defaultConfigurationValues() {
        let config = CommandAllocatorPool.Configuration.default

        #expect(config.poolSize == 3)
        #expect(config.heapSizePerAllocator == 16 * 1024 * 1024)
    }

    @Test("Large batch configuration has expected values")
    func largeBatchConfigurationValues() {
        let config = CommandAllocatorPool.Configuration.largeBatch

        #expect(config.poolSize == 4)
        #expect(config.heapSizePerAllocator == 64 * 1024 * 1024)
    }

    @Test("Custom configuration respects minimum values")
    func customConfigurationMinimumsz() {
        let config = CommandAllocatorPool.Configuration(
            poolSize: 0,
            heapSizePerAllocator: 100
        )

        #expect(config.poolSize == 1) // Minimum 1
        #expect(config.heapSizePerAllocator == 1024 * 1024) // Minimum 1MB
    }

    @Test("Custom configuration accepts valid values")
    func customConfigurationValidValues() {
        let config = CommandAllocatorPool.Configuration(
            poolSize: 5,
            heapSizePerAllocator: 32 * 1024 * 1024
        )

        #expect(config.poolSize == 5)
        #expect(config.heapSizePerAllocator == 32 * 1024 * 1024)
    }
}

// MARK: - CommandAllocatorPool Basic Tests

@Suite("CommandAllocatorPool - Basic Operations")
struct CommandAllocatorPoolBasicTests {

    #if canImport(Metal)
    @Test("Initializes with default configuration")
    func initializesWithDefaultConfig() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CommandAllocatorTestError.skipped("Metal not available")
        }

        let pool = CommandAllocatorPool(device: device)
        let stats = await pool.getStatistics()

        #expect(stats.totalAllocators == 3)
        #expect(stats.availableAllocators == 3)
        #expect(stats.inUseAllocators == 0)
    }

    @Test("Initializes with custom configuration")
    func initializesWithCustomConfig() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CommandAllocatorTestError.skipped("Metal not available")
        }

        let config = CommandAllocatorPool.Configuration(
            poolSize: 5,
            heapSizePerAllocator: 8 * 1024 * 1024
        )
        let pool = CommandAllocatorPool(device: device, configuration: config)
        let stats = await pool.getStatistics()

        #expect(stats.totalAllocators == 5)
        #expect(stats.heapSizePerAllocator == 8 * 1024 * 1024)
    }

    @Test("Initializes with convenience initializer")
    func initializesWithConvenience() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CommandAllocatorTestError.skipped("Metal not available")
        }

        let pool = CommandAllocatorPool(device: device, poolSize: 4, heapSizeMB: 32)
        let stats = await pool.getStatistics()

        #expect(stats.totalAllocators == 4)
        #expect(stats.heapSizePerAllocator == 32 * 1024 * 1024)
    }
    #endif
}

// MARK: - CommandAllocatorPool Acquire/Release Tests

@Suite("CommandAllocatorPool - Acquire/Release")
struct CommandAllocatorPoolAcquireReleaseTests {

    #if canImport(Metal)
    @Test("Acquires allocator from pool")
    func acquiresAllocator() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CommandAllocatorTestError.skipped("Metal not available")
        }

        let pool = CommandAllocatorPool(device: device)

        let allocator = await pool.acquire()
        let stats = await pool.getStatistics()

        #expect(allocator.isInUse == true)
        #expect(stats.availableAllocators == 2)
        #expect(stats.inUseAllocators == 1)
    }

    @Test("Releases allocator back to pool")
    func releasesAllocator() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CommandAllocatorTestError.skipped("Metal not available")
        }

        let pool = CommandAllocatorPool(device: device)

        let allocator = await pool.acquire()
        #expect(await pool.inUseCount == 1)

        await pool.release(allocator)
        #expect(await pool.inUseCount == 0)
        #expect(await pool.availableCount == 3)
    }

    @Test("tryAcquire returns nil when pool exhausted")
    func tryAcquireReturnsNilWhenExhausted() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CommandAllocatorTestError.skipped("Metal not available")
        }

        let pool = CommandAllocatorPool(device: device, poolSize: 2, heapSizeMB: 8)

        // Acquire all
        let a1 = await pool.tryAcquire()
        let a2 = await pool.tryAcquire()
        let a3 = await pool.tryAcquire()

        #expect(a1 != nil)
        #expect(a2 != nil)
        #expect(a3 == nil) // Pool exhausted
    }

    @Test("Multiple acquire/release cycles work correctly")
    func multipleAcquireReleaseCycles() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CommandAllocatorTestError.skipped("Metal not available")
        }

        let pool = CommandAllocatorPool(device: device, poolSize: 2, heapSizeMB: 8)

        // First cycle
        let a1 = await pool.acquire()
        let a2 = await pool.acquire()
        #expect(await pool.availableCount == 0)

        await pool.release(a1)
        await pool.release(a2)
        #expect(await pool.availableCount == 2)

        // Second cycle
        let b1 = await pool.acquire()
        let b2 = await pool.acquire()
        #expect(await pool.availableCount == 0)

        await pool.release(b1)
        await pool.release(b2)
        #expect(await pool.availableCount == 2)
    }

    @Test("releaseAll returns all allocators")
    func releaseAllReturnsAll() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CommandAllocatorTestError.skipped("Metal not available")
        }

        let pool = CommandAllocatorPool(device: device, poolSize: 3, heapSizeMB: 8)

        _ = await pool.acquire()
        _ = await pool.acquire()
        _ = await pool.acquire()
        #expect(await pool.availableCount == 0)

        await pool.releaseAll()
        #expect(await pool.availableCount == 3)
    }
    #endif
}

// MARK: - CommandAllocatorPool Statistics Tests

@Suite("CommandAllocatorPool - Statistics")
struct CommandAllocatorPoolStatisticsTests {

    #if canImport(Metal)
    @Test("Statistics track acquisitions")
    func statisticsTrackAcquisitions() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CommandAllocatorTestError.skipped("Metal not available")
        }

        let pool = CommandAllocatorPool(device: device)

        _ = await pool.acquire()
        _ = await pool.acquire()

        let stats = await pool.getStatistics()
        #expect(stats.totalAcquisitions == 2)
    }

    @Test("Statistics track resets")
    func statisticsTrackResets() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CommandAllocatorTestError.skipped("Metal not available")
        }

        let pool = CommandAllocatorPool(device: device)

        let a1 = await pool.acquire()
        let a2 = await pool.acquire()
        await pool.release(a1)
        await pool.release(a2)

        let stats = await pool.getStatistics()
        #expect(stats.totalResets == 2)
    }

    @Test("Utilization percentage calculated correctly")
    func utilizationPercentageCalculation() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CommandAllocatorTestError.skipped("Metal not available")
        }

        let pool = CommandAllocatorPool(device: device, poolSize: 4, heapSizeMB: 8)

        _ = await pool.acquire()
        _ = await pool.acquire()

        let stats = await pool.getStatistics()
        #expect(stats.utilizationPercentage == 50.0)
    }

    @Test("Total heap size calculated correctly")
    func totalHeapSizeCalculation() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CommandAllocatorTestError.skipped("Metal not available")
        }

        let pool = CommandAllocatorPool(device: device, poolSize: 4, heapSizeMB: 8)

        let stats = await pool.getStatistics()
        #expect(stats.totalHeapSize == 4 * 8 * 1024 * 1024)
    }

    @Test("Statistics reflect available and in-use counts")
    func statisticsReflectCounts() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CommandAllocatorTestError.skipped("Metal not available")
        }

        let pool = CommandAllocatorPool(device: device, poolSize: 4, heapSizeMB: 8)

        let a1 = await pool.acquire()
        let stats1 = await pool.getStatistics()
        #expect(stats1.availableAllocators == 3)
        #expect(stats1.inUseAllocators == 1)

        await pool.release(a1)
        let stats2 = await pool.getStatistics()
        #expect(stats2.availableAllocators == 4)
        #expect(stats2.inUseAllocators == 0)
    }
    #endif
}

// MARK: - CommandAllocatorPool Concurrent Tests

@Suite("CommandAllocatorPool - Concurrency", .serialized)
struct CommandAllocatorPoolConcurrencyTests {

    #if canImport(Metal)
    @Test("Handles concurrent acquire operations")
    func handlesConcurrentAcquire() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CommandAllocatorTestError.skipped("Metal not available")
        }

        let pool = CommandAllocatorPool(device: device, poolSize: 10, heapSizeMB: 4)

        // Acquire concurrently from multiple tasks
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<10 {
                group.addTask {
                    let allocator = await pool.acquire()
                    // Simulate some work
                    try? await Task.sleep(for: .milliseconds(10))
                    await pool.release(allocator)
                }
            }
        }

        // All should be released
        let stats = await pool.getStatistics()
        #expect(stats.availableAllocators == 10)
        #expect(stats.inUseAllocators == 0)
    }

    @Test("Maintains consistency under concurrent access")
    func maintainsConsistencyUnderConcurrency() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CommandAllocatorTestError.skipped("Metal not available")
        }

        let pool = CommandAllocatorPool(device: device, poolSize: 5, heapSizeMB: 4)

        // Multiple concurrent acquire/release cycles
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<20 {
                group.addTask {
                    if let allocator = await pool.tryAcquire() {
                        try? await Task.sleep(for: .milliseconds(5))
                        await pool.release(allocator)
                    }
                }
            }
        }

        // Pool should be in consistent state
        let stats = await pool.getStatistics()
        #expect(stats.availableAllocators + stats.inUseAllocators == 5)
    }
    #endif
}

// MARK: - CommandAllocator Protocol Tests

@Suite("CommandAllocator Protocol")
struct CommandAllocatorProtocolTests {

    @Test("SimpleCommandAllocator conforms to CommandAllocator")
    func simpleAllocatorConformsToProtocol() {
        let allocator: any CommandAllocator = SimpleCommandAllocator(heapSize: 8 * 1024 * 1024)

        #expect(allocator.heapSize == 8 * 1024 * 1024)
        #expect(allocator.isInUse == false)

        allocator.reset()
        #expect(allocator.isInUse == false)
    }
}
