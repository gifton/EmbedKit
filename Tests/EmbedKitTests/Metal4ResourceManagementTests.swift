// EmbedKit - Metal 4 Resource Management Tests

import Testing
import Foundation
@testable import EmbedKit

#if canImport(Metal)
import Metal
#endif

// MARK: - Metal4ResidencySet Tests

@Suite("Metal4ResidencySet")
struct Metal4ResidencySetTests {

    #if canImport(Metal)
    @Test("Residency set initializes correctly")
    func initializesCorrectly() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4ResourceTestError.skipped("Metal not available")
        }

        let set = Metal4ResidencySet(device: device, identifier: "test", initialCapacity: 32)

        #expect(set.identifier == "test")
        #expect(set.allocationCount == 0)
        #expect(set.committed == false)
    }

    @Test("Add and remove buffers")
    func addAndRemoveBuffers() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4ResourceTestError.skipped("Metal not available")
        }

        let set = Metal4ResidencySet(device: device, identifier: "test")
        let buffer1 = device.makeBuffer(length: 1024, options: .storageModeShared)!
        let buffer2 = device.makeBuffer(length: 2048, options: .storageModeShared)!

        set.addBuffer(buffer1)
        #expect(set.allocationCount == 1)

        set.addBuffer(buffer2)
        #expect(set.allocationCount == 2)

        set.removeBuffer(buffer1)
        #expect(set.allocationCount == 1)
    }

    @Test("Add multiple buffers at once")
    func addMultipleBuffers() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4ResourceTestError.skipped("Metal not available")
        }

        let set = Metal4ResidencySet(device: device, identifier: "test")
        let buffers = (0..<5).compactMap { _ in
            device.makeBuffer(length: 1024, options: .storageModeShared)
        }

        set.addBuffers(buffers)
        #expect(set.allocationCount == 5)
    }

    @Test("Commit changes state")
    func commitChangesState() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4ResourceTestError.skipped("Metal not available")
        }

        let set = Metal4ResidencySet(device: device, identifier: "test")

        #expect(set.committed == false)
        set.commit()
        #expect(set.committed == true)
    }
    #endif
}

// MARK: - Metal4ResidencyManager Tests

@Suite("Metal4ResidencyManager")
struct Metal4ResidencyManagerTests {

    #if canImport(Metal)
    @Test("Manager initializes correctly")
    func initializesCorrectly() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4ResourceTestError.skipped("Metal not available")
        }

        let manager = Metal4ResidencyManager(device: device, maxResidentMB: 256)
        let stats = await manager.getStatistics()

        #expect(stats.totalSets == 0)
        #expect(stats.maxResidentBytes == 256 * 1024 * 1024)
    }

    @Test("Create and retrieve residency sets")
    func createAndRetrieveSets() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4ResourceTestError.skipped("Metal not available")
        }

        let manager = Metal4ResidencyManager(device: device)

        let set1 = try await manager.createResidencySet(named: "embeddings")
        let set2 = try await manager.createResidencySet(named: "cache")

        #expect(set1.identifier == "embeddings")
        #expect(set2.identifier == "cache")

        let stats = await manager.getStatistics()
        #expect(stats.totalSets == 2)
    }

    @Test("Duplicate set name throws error")
    func duplicateSetNameThrows() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4ResourceTestError.skipped("Metal not available")
        }

        let manager = Metal4ResidencyManager(device: device)
        _ = try await manager.createResidencySet(named: "test")

        await #expect(throws: EmbedKitError.self) {
            _ = try await manager.createResidencySet(named: "test")
        }
    }

    @Test("Add buffer to set")
    func addBufferToSet() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4ResourceTestError.skipped("Metal not available")
        }

        let manager = Metal4ResidencyManager(device: device)
        _ = try await manager.createResidencySet(named: "test")

        let buffer = device.makeBuffer(length: 1024, options: .storageModeShared)!
        try await manager.addBuffer(buffer, toSet: "test")

        let stats = await manager.getStatistics()
        #expect(stats.totalAllocations == 1)
    }

    @Test("Commit set")
    func commitSet() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4ResourceTestError.skipped("Metal not available")
        }

        let manager = Metal4ResidencyManager(device: device)
        let set = try await manager.createResidencySet(named: "test")

        #expect(set.committed == false)
        try await manager.commitSet(named: "test")
        #expect(set.committed == true)

        let stats = await manager.getStatistics()
        #expect(stats.committedSets == 1)
    }

    @Test("Remove residency set")
    func removeSet() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4ResourceTestError.skipped("Metal not available")
        }

        let manager = Metal4ResidencyManager(device: device)
        _ = try await manager.createResidencySet(named: "test")

        var stats = await manager.getStatistics()
        #expect(stats.totalSets == 1)

        await manager.removeResidencySet(named: "test")

        stats = await manager.getStatistics()
        #expect(stats.totalSets == 0)
    }
    #endif
}

// MARK: - Metal4ArgumentTable Tests

@Suite("Metal4ArgumentTable")
struct Metal4ArgumentTableTests {

    #if canImport(Metal)
    @Test("Argument table initializes with configuration")
    func initializesWithConfiguration() throws {
        let table = Metal4ArgumentTable(configuration: .init(
            maxBufferBindCount: 8,
            maxTextureBindCount: 2
        ))

        #expect(table.configuration.maxBufferBindCount == 8)
        #expect(table.configuration.maxTextureBindCount == 2)
        #expect(table.activeBufferCount == 0)
        #expect(table.activeTextureCount == 0)
    }

    @Test("Set and clear buffer bindings")
    func setAndClearBufferBindings() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4ResourceTestError.skipped("Metal not available")
        }

        let table = Metal4ArgumentTable(configuration: .default)
        let buffer = device.makeBuffer(length: 1024, options: .storageModeShared)!

        table.setBuffer(buffer, offset: 0, at: 0)
        #expect(table.activeBufferCount == 1)

        table.setBuffer(buffer, offset: 512, at: 1)
        #expect(table.activeBufferCount == 2)

        table.clearBuffer(at: 0)
        #expect(table.activeBufferCount == 1)
    }

    @Test("Clear all bindings")
    func clearAllBindings() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4ResourceTestError.skipped("Metal not available")
        }

        let table = Metal4ArgumentTable(configuration: .default)
        let buffer = device.makeBuffer(length: 1024, options: .storageModeShared)!

        table.setBuffer(buffer, at: 0)
        table.setBuffer(buffer, at: 1)
        table.setBuffer(buffer, at: 2)
        #expect(table.activeBufferCount == 3)

        table.clearAll()
        #expect(table.activeBufferCount == 0)
    }

    @Test("Apply bindings to encoder")
    func applyBindingsToEncoder() async throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else {
            throw Metal4ResourceTestError.skipped("Metal not available")
        }

        let table = Metal4ArgumentTable(configuration: .default)
        let buffer = device.makeBuffer(length: 1024, options: .storageModeShared)!

        table.setBuffer(buffer, at: 0)

        // Should not throw
        table.applyTo(encoder)

        encoder.endEncoding()
    }

    @Test("Ignore out of bounds bindings")
    func ignoreOutOfBoundsBindings() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Metal4ResourceTestError.skipped("Metal not available")
        }

        let table = Metal4ArgumentTable(configuration: .init(
            maxBufferBindCount: 2,
            maxTextureBindCount: 0
        ))
        let buffer = device.makeBuffer(length: 1024, options: .storageModeShared)!

        // These should be silently ignored
        table.setBuffer(buffer, at: 5)
        table.setBuffer(buffer, at: 100)

        #expect(table.activeBufferCount == 0)
    }
    #endif
}

// MARK: - Metal4ArgumentTableFactory Tests

@Suite("Metal4ArgumentTableFactory")
struct Metal4ArgumentTableFactoryTests {

    #if canImport(Metal)
    @Test("Create pooling table")
    func createPoolingTable() throws {
        let table = Metal4ArgumentTableFactory.createForPooling()

        #expect(table.configuration.maxBufferBindCount == 4)
        #expect(table.configuration.maxTextureBindCount == 0)
    }

    @Test("Create normalization table")
    func createNormalizationTable() throws {
        let table = Metal4ArgumentTableFactory.createForNormalization()

        #expect(table.configuration.maxBufferBindCount == 3)
        #expect(table.configuration.maxTextureBindCount == 0)
    }

    @Test("Create fused pool norm table")
    func createFusedPoolNormTable() throws {
        let table = Metal4ArgumentTableFactory.createForFusedPoolNorm()

        #expect(table.configuration.maxBufferBindCount == 4)
        #expect(table.configuration.maxTextureBindCount == 0)
    }

    @Test("Create similarity table")
    func createSimilarityTable() throws {
        let table = Metal4ArgumentTableFactory.createForSimilarity()

        #expect(table.configuration.maxBufferBindCount == 4)
        #expect(table.configuration.maxTextureBindCount == 0)
    }
    #endif
}

// MARK: - Configuration Tests

@Suite("Metal4ArgumentTableConfiguration")
struct ArgumentTableConfigurationTests {

    @Test("Default configuration values")
    func defaultConfigurationValues() throws {
        let config = Metal4ArgumentTable.Configuration.default

        #expect(config.maxBufferBindCount == 16)
        #expect(config.maxTextureBindCount == 4)
    }

    @Test("Embedding configuration values")
    func embeddingConfigurationValues() throws {
        let config = Metal4ArgumentTable.Configuration.embedding

        #expect(config.maxBufferBindCount == 8)
        #expect(config.maxTextureBindCount == 0)
    }

    @Test("Similarity configuration values")
    func similarityConfigurationValues() throws {
        let config = Metal4ArgumentTable.Configuration.similarity

        #expect(config.maxBufferBindCount == 4)
        #expect(config.maxTextureBindCount == 0)
    }

    @Test("Configuration enforces minimum values")
    func configurationEnforcesMinimums() throws {
        let config = Metal4ArgumentTable.Configuration(
            maxBufferBindCount: -5,
            maxTextureBindCount: -10
        )

        #expect(config.maxBufferBindCount >= 1)
        #expect(config.maxTextureBindCount >= 0)
    }
}

// MARK: - Test Helper

enum Metal4ResourceTestError: Error {
    case skipped(String)
}
