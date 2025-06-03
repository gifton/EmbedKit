import Foundation
import Testing
@testable import EmbedKit
import OSLog

@Suite("Storage Components Tests")
struct StorageTests {
    
    // MARK: - PersistentModelRegistry Tests
    
    @Test("Initialize PersistentModelRegistry with custom directory")
    func testInitializeModelRegistryCustomDirectory() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("EmbedKitTest_\(UUID())")
        let _ = try await PersistentModelRegistry(storageDirectory: tempDir)
        
        let dbPath = tempDir.appendingPathComponent("model_registry.sqlite").path
        #expect(FileManager.default.fileExists(atPath: dbPath))
        
        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }
    
    @Test("Save and load model version")
    func testSaveAndLoadModelVersion() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("EmbedKitTest_\(UUID())")
        let registry = try await PersistentModelRegistry(storageDirectory: tempDir)
        
        // Create test model file
        let modelFile = tempDir.appendingPathComponent("test_model.mlmodel")
        try "test model data".data(using: .utf8)!.write(to: modelFile)
        
        // Create version
        let version = ModelVersion(
            identifier: "test-model",
            version: "1.0.0",
            buildNumber: 100,
            createdAt: Date(),
            metadata: ["author": "test", "description": "Test model"]
        )
        
        // Save version
        try await registry.saveVersion(version, fileURL: modelFile, signature: "test-signature")
        
        // Load versions
        let loadedVersions = try await registry.loadVersions(for: "test-model")
        print("DEBUG: Loaded \(loadedVersions.count) versions")
        for (index, record) in loadedVersions.enumerated() {
            print("DEBUG: Record \(index): id=\(record.id), identifier=\(record.version.identifier), version=\(record.version.version)")
        }
        #expect(loadedVersions.count == 1)
        
        let loadedRecord = loadedVersions.first!
        #expect(loadedRecord.version.identifier == version.identifier)
        #expect(loadedRecord.version.version == version.version)
        #expect(loadedRecord.version.buildNumber == version.buildNumber)
        #expect(loadedRecord.filePath == modelFile.path)
        #expect(loadedRecord.signatureHash == "test-signature")
        #expect(loadedRecord.fileSize > 0)
        #expect(!loadedRecord.checksum.isEmpty)
        
        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }
    
    @Test("Set and get active version")
    func testSetAndGetActiveVersion() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("EmbedKitTest_\(UUID())")
        let registry = try await PersistentModelRegistry(storageDirectory: tempDir)
        
        // Create test model files
        let modelFile1 = tempDir.appendingPathComponent("model_v1.mlmodel")
        let modelFile2 = tempDir.appendingPathComponent("model_v2.mlmodel")
        try "model v1".data(using: .utf8)!.write(to: modelFile1)
        try "model v2".data(using: .utf8)!.write(to: modelFile2)
        
        // Create versions
        let version1 = ModelVersion(
            identifier: "test-model",
            version: "1.0.0",
            buildNumber: 100,
            createdAt: Date(),
            metadata: [:]
        )
        
        let version2 = ModelVersion(
            identifier: "test-model",
            version: "2.0.0",
            buildNumber: 200,
            createdAt: Date().addingTimeInterval(3600),
            metadata: [:]
        )
        
        // Save versions
        try await registry.saveVersion(version1, fileURL: modelFile1)
        try await registry.saveVersion(version2, fileURL: modelFile2)
        
        // Initially no active version
        let initialActive = try await registry.getActiveVersion(for: "test-model")
        #expect(initialActive == nil)
        
        // Set version 1 as active
        try await registry.setActiveVersion(version1)
        var activeRecord = try await registry.getActiveVersion(for: "test-model")
        #expect(activeRecord?.version.version == "1.0.0")
        #expect(activeRecord?.isActive == true)
        
        // Switch to version 2
        try await registry.setActiveVersion(version2)
        activeRecord = try await registry.getActiveVersion(for: "test-model")
        #expect(activeRecord?.version.version == "2.0.0")
        
        // Verify version 1 is no longer active
        let allVersions = try await registry.loadVersions(for: "test-model")
        let v1Record = allVersions.first { $0.version.version == "1.0.0" }
        #expect(v1Record?.isActive == false)
        
        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }
    
    @Test("Remove model version")
    func testRemoveModelVersion() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("EmbedKitTest_\(UUID())")
        let registry = try await PersistentModelRegistry(storageDirectory: tempDir)
        
        let modelFile = tempDir.appendingPathComponent("test_model.mlmodel")
        try "test model".data(using: .utf8)!.write(to: modelFile)
        
        let version = ModelVersion(
            identifier: "test-model",
            version: "1.0.0",
            buildNumber: 100,
            createdAt: Date(),
            metadata: [:]
        )
        
        // Save and verify
        try await registry.saveVersion(version, fileURL: modelFile)
        var versions = try await registry.loadVersions(for: "test-model")
        #expect(versions.count == 1)
        
        // Remove version
        try await registry.removeVersion(version)
        
        // Verify removal
        versions = try await registry.loadVersions(for: "test-model")
        #expect(versions.isEmpty)
        
        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }
    
    @Test("Get storage statistics")
    func testGetStorageStatistics() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("EmbedKitTest_\(UUID())")
        let registry = try await PersistentModelRegistry(storageDirectory: tempDir)
        
        // Add multiple models and versions
        for i in 1...3 {
            let modelFile = tempDir.appendingPathComponent("model_\(i).mlmodel")
            let data = String(repeating: "x", count: 1000 * i).data(using: .utf8)!
            try data.write(to: modelFile)
            
            for j in 1...2 {
                let version = ModelVersion(
                    identifier: "model-\(i)",
                    version: "\(j).0.0",
                    buildNumber: j * 100,
                    createdAt: Date(),
                    metadata: [:]
                )
                try await registry.saveVersion(version, fileURL: modelFile)
            }
        }
        
        let stats = try await registry.getStatistics()
        #expect(stats.totalModels == 3)
        #expect(stats.totalVersions == 6)
        #expect(stats.totalFileSize > 0)
        #expect(stats.databaseSize > 0)
        
        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }
    
    @Test("Cleanup orphaned records")
    func testCleanupOrphanedRecords() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("EmbedKitTest_\(UUID())")
        let registry = try await PersistentModelRegistry(storageDirectory: tempDir)
        
        // Create model files
        let modelFile1 = tempDir.appendingPathComponent("model1.mlmodel")
        let modelFile2 = tempDir.appendingPathComponent("model2.mlmodel")
        try "model 1".data(using: .utf8)!.write(to: modelFile1)
        try "model 2".data(using: .utf8)!.write(to: modelFile2)
        
        // Save versions
        let version1 = ModelVersion(identifier: "model-1", version: "1.0.0", buildNumber: 100, createdAt: Date(), metadata: [:])
        let version2 = ModelVersion(identifier: "model-2", version: "1.0.0", buildNumber: 100, createdAt: Date(), metadata: [:])
        
        try await registry.saveVersion(version1, fileURL: modelFile1)
        try await registry.saveVersion(version2, fileURL: modelFile2)
        
        // Delete one model file to create orphan
        try FileManager.default.removeItem(at: modelFile1)
        
        // Verify both records exist
        let model1Versions = try await registry.loadVersions(for: "model-1")
        let model2Versions = try await registry.loadVersions(for: "model-2")
        let allVersions = model1Versions + model2Versions
        #expect(allVersions.count == 2)
        
        // Run cleanup
        try await registry.cleanup()
        
        // Verify orphaned record is removed
        let remainingVersions = try await registry.loadVersions(for: "model-1")
        #expect(remainingVersions.isEmpty)
        
        let validVersions = try await registry.loadVersions(for: "model-2")
        #expect(validVersions.count == 1)
        
        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }
    
    @Test("Handle concurrent operations")
    func testConcurrentOperations() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("EmbedKitTest_\(UUID())")
        let registry = try await PersistentModelRegistry(storageDirectory: tempDir)
        
        // Create model file
        let modelFile = tempDir.appendingPathComponent("concurrent_model.mlmodel")
        try "concurrent test model".data(using: .utf8)!.write(to: modelFile)
        
        // Perform concurrent saves
        try await withThrowingTaskGroup(of: Void.self) { group in
            for i in 1...10 {
                group.addTask {
                    let version = ModelVersion(
                        identifier: "concurrent-model",
                        version: "1.0.\(i)",
                        buildNumber: 100 + i,
                        createdAt: Date(),
                        metadata: ["index": "\(i)"]
                    )
                    try await registry.saveVersion(version, fileURL: modelFile)
                }
            }
            
            try await group.waitForAll()
        }
        
        // Verify all versions were saved
        let versions = try await registry.loadVersions(for: "concurrent-model")
        #expect(versions.count == 10)
        
        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }
    
    // MARK: - PersistentModelVersionRegistry Tests
    
    @Test("Initialize PersistentModelVersionRegistry")
    func testInitializePersistentModelVersionRegistry() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("EmbedKitTest_\(UUID())")
        let registry = try await PersistentModelVersionRegistry(storageDirectory: tempDir)
        
        // Test initial state
        let stats = await registry.getStatistics()
        #expect(stats.storage.totalModels == 0)
        #expect(stats.cache.cachedModels == 0)
        
        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }
    
    @Test("Register and retrieve model version")
    func testRegisterAndRetrieveModelVersion() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("EmbedKitTest_\(UUID())")
        let registry = try await PersistentModelVersionRegistry(storageDirectory: tempDir)
        
        // Create model file
        let modelFile = tempDir.appendingPathComponent("test_model.mlmodel")
        try "test model data".data(using: .utf8)!.write(to: modelFile)
        
        // Register version
        let version = ModelVersion(
            identifier: "test-model",
            version: "1.0.0",
            buildNumber: 100,
            createdAt: Date(),
            metadata: ["key": "value"]
        )
        
        try await registry.register(version: version, modelURL: modelFile, signature: "test-sig")
        
        // Retrieve active version
        let activeVersion = await registry.getActiveVersion(for: "test-model")
        #expect(activeVersion?.identifier == version.identifier)
        #expect(activeVersion?.version == version.version)
        
        // Get all versions
        let allVersions = await registry.getVersions(for: "test-model")
        #expect(allVersions.count == 1)
        #expect(allVersions.first == version)
        
        // Get model URL
        let modelURL = await registry.getModelURL(for: version)
        #expect(modelURL?.path == modelFile.path)
        
        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }
    
    @Test("Model version auto-activation")
    func testModelVersionAutoActivation() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("EmbedKitTest_\(UUID())")
        let registry = try await PersistentModelVersionRegistry(storageDirectory: tempDir)
        
        // Create model files
        let modelFile1 = tempDir.appendingPathComponent("model_v1.mlmodel")
        let modelFile2 = tempDir.appendingPathComponent("model_v2.mlmodel")
        try "v1".data(using: .utf8)!.write(to: modelFile1)
        try "v2".data(using: .utf8)!.write(to: modelFile2)
        
        // Register older version
        let version1 = ModelVersion(
            identifier: "auto-model",
            version: "1.0.0",
            buildNumber: 100,
            createdAt: Date(),
            metadata: [:]
        )
        
        try await registry.register(version: version1, modelURL: modelFile1)
        
        // Verify v1 is active
        var activeVersion = await registry.getActiveVersion(for: "auto-model")
        #expect(activeVersion?.version == "1.0.0")
        
        // Register newer version
        let version2 = ModelVersion(
            identifier: "auto-model",
            version: "2.0.0",
            buildNumber: 200,
            createdAt: Date().addingTimeInterval(3600),
            metadata: [:]
        )
        
        try await registry.register(version: version2, modelURL: modelFile2)
        
        // Verify v2 is now active
        activeVersion = await registry.getActiveVersion(for: "auto-model")
        #expect(activeVersion?.version == "2.0.0")
        
        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }
    
    @Test("Verify model integrity")
    func testVerifyModelIntegrity() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("EmbedKitTest_\(UUID())")
        let registry = try await PersistentModelVersionRegistry(storageDirectory: tempDir)
        
        // Create model file
        let modelFile = tempDir.appendingPathComponent("integrity_test.mlmodel")
        let modelData = "test model data for integrity check".data(using: .utf8)!
        try modelData.write(to: modelFile)
        
        // Register version
        let version = ModelVersion(
            identifier: "integrity-model",
            version: "1.0.0",
            buildNumber: 100,
            createdAt: Date(),
            metadata: [:]
        )
        
        try await registry.register(version: version, modelURL: modelFile)
        
        // Verify integrity - should pass
        var result = try await registry.verifyModelIntegrity(for: version)
        #expect(result.isValid)
        #expect(result.issues.isEmpty)
        
        // Corrupt the file
        try "corrupted data".data(using: .utf8)!.write(to: modelFile)
        
        // Verify integrity - should fail
        result = try await registry.verifyModelIntegrity(for: version)
        #expect(!result.isValid)
        #expect(result.issues.contains { issue in
            if case .checksumMismatch = issue { return true }
            return false
        })
        
        // Delete the file
        try FileManager.default.removeItem(at: modelFile)
        
        // Verify integrity - should report file not found
        result = try await registry.verifyModelIntegrity(for: version)
        #expect(!result.isValid)
        #expect(result.issues.contains { issue in
            if case .fileNotFound = issue { return true }
            return false
        })
        
        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }
    
    @Test("Remove version and update active")
    func testRemoveVersionAndUpdateActive() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("EmbedKitTest_\(UUID())")
        let registry = try await PersistentModelVersionRegistry(storageDirectory: tempDir)
        
        // Create model files
        let modelFiles = (1...3).map { tempDir.appendingPathComponent("model_v\($0).mlmodel") }
        for (i, file) in modelFiles.enumerated() {
            try "model v\(i+1)".data(using: .utf8)!.write(to: file)
        }
        
        // Register multiple versions
        let versions = (1...3).map { i in
            ModelVersion(
                identifier: "remove-test",
                version: "\(i).0.0",
                buildNumber: i * 100,
                createdAt: Date().addingTimeInterval(Double(i * 3600)),
                metadata: [:]
            )
        }
        
        for (version, file) in zip(versions, modelFiles) {
            try await registry.register(version: version, modelURL: file)
        }
        
        // Set v2 as active
        try await registry.setActiveVersion(versions[1])
        
        // Remove active version (v2)
        try await registry.removeVersion(versions[1])
        
        // Verify v3 becomes active (newest remaining)
        let activeVersion = await registry.getActiveVersion(for: "remove-test")
        #expect(activeVersion?.version == "3.0.0")
        
        // Verify only 2 versions remain
        let remainingVersions = await registry.getVersions(for: "remove-test")
        #expect(remainingVersions.count == 2)
        #expect(!remainingVersions.contains(versions[1]))
        
        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }
    
    @Test("Export registry data")
    func testExportRegistryData() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("EmbedKitTest_\(UUID())")
        let registry = try await PersistentModelVersionRegistry(storageDirectory: tempDir)
        
        // Create and register models
        for i in 1...2 {
            let modelFile = tempDir.appendingPathComponent("model_\(i).mlmodel")
            try "model \(i)".data(using: .utf8)!.write(to: modelFile)
            
            let version = ModelVersion(
                identifier: "export-model-\(i)",
                version: "1.0.0",
                buildNumber: 100,
                createdAt: Date(),
                metadata: ["index": "\(i)"]
            )
            
            try await registry.register(version: version, modelURL: modelFile)
        }
        
        // Export data
        let exportData = try await registry.exportData()
        
        #expect(exportData.statistics.totalModels >= 2)
        #expect(exportData.statistics.totalVersions >= 2)
        #expect(!exportData.modelData.isEmpty)
        #expect(exportData.exportedAt.timeIntervalSinceNow < 1)
        
        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }
    
    @Test("Perform health check")
    func testPerformHealthCheck() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("EmbedKitTest_\(UUID())")
        let registry = try await PersistentModelVersionRegistry(storageDirectory: tempDir)
        
        // Initial health check - should be healthy
        var health = await registry.performHealthCheck()
        #expect(health.isHealthy)
        #expect(health.issues.isEmpty)
        
        // Create and register a model
        let modelFile = tempDir.appendingPathComponent("health_test.mlmodel")
        try "healthy model".data(using: .utf8)!.write(to: modelFile)
        
        let version = ModelVersion(
            identifier: "health-model",
            version: "1.0.0",
            buildNumber: 100,
            createdAt: Date(),
            metadata: [:]
        )
        
        try await registry.register(version: version, modelURL: modelFile)
        
        // Delete file to create orphaned record
        try FileManager.default.removeItem(at: modelFile)
        
        // Health check should detect orphaned record
        health = await registry.performHealthCheck()
        #expect(!health.isHealthy)
        #expect(!health.issues.isEmpty)
        #expect(health.issues.contains { issue in
            if case .orphanedRecord(_, _) = issue { return true }
            return false
        })
        
        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }
    
    @Test("Test model version metadata persistence")
    func testModelVersionMetadataPersistence() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("EmbedKitTest_\(UUID())")
        let registry = try await PersistentModelVersionRegistry(storageDirectory: tempDir)
        
        let modelFile = tempDir.appendingPathComponent("metadata_test.mlmodel")
        try "metadata test".data(using: .utf8)!.write(to: modelFile)
        
        // Create version with complex metadata
        let metadata = [
            "author": "Test Author",
            "description": "Test model with metadata",
            "framework": "CoreML",
            "inputDimensions": "512",
            "outputDimensions": "768",
            "trainingDataset": "TestDataset-v1"
        ]
        
        let version = ModelVersion(
            identifier: "metadata-model",
            version: "1.0.0",
            buildNumber: 100,
            createdAt: Date(),
            metadata: metadata
        )
        
        try await registry.register(version: version, modelURL: modelFile)
        
        // Retrieve and verify metadata
        let record = await registry.getVersionRecord(for: version)
        #expect(record != nil)
        #expect(record?.version.metadata.count == metadata.count)
        
        for (key, value) in metadata {
            #expect(record?.version.metadata[key] == value)
        }
        
        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }
    
    @Test("Error handling for invalid operations")
    func testErrorHandlingInvalidOperations() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("EmbedKitTest_\(UUID())")
        let registry = try await PersistentModelVersionRegistry(storageDirectory: tempDir)
        
        // Try to register with non-existent file
        let version = ModelVersion(
            identifier: "error-model",
            version: "1.0.0",
            buildNumber: 100,
            createdAt: Date(),
            metadata: [:]
        )
        
        let nonExistentFile = tempDir.appendingPathComponent("non_existent.mlmodel")
        
        await #expect(throws: Error.self) {
            try await registry.register(version: version, modelURL: nonExistentFile)
        }
        
        // Try to set active version that doesn't exist
        await #expect(throws: Error.self) {
            try await registry.setActiveVersion(version)
        }
        
        // Try to verify integrity of non-existent version
        await #expect(throws: Error.self) {
            try await registry.verifyModelIntegrity(for: version)
        }
        
        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }
}

