import XCTest
@testable import EmbedKit

// MARK: - Swift 6 Concurrency Tests

final class Swift6ConcurrencyTests: XCTestCase {
    
    // MARK: - ConsoleDownloadDelegate Tests
    
    func testConsoleDownloadDelegateActorIsolation() async throws {
        let delegate = ConsoleDownloadDelegate()
        
        // Test concurrent access
        await withTaskGroup(of: Void.self) { group in
            for i in 0..<10 {
                group.addTask {
                    let url = URL(string: "https://example.com/model\(i).mlmodel")!
                    await delegate.downloadDidStart(url: url)
                    await delegate.downloadDidProgress(
                        bytesWritten: 1024,
                        totalBytesWritten: Int64(i * 1024),
                        totalBytesExpected: 10240
                    )
                }
            }
        }
        
        // Should complete without data races
    }
    
    func testDetailedDownloadDelegateThrottling() async throws {
        let delegate = DetailedDownloadDelegate()
        let url = URL(string: "https://example.com/model.mlmodel")!
        
        await delegate.downloadDidStart(url: url)
        
        // Rapid progress updates
        for i in 1...100 {
            await delegate.downloadDidProgress(
                bytesWritten: 1024,
                totalBytesWritten: Int64(i * 1024),
                totalBytesExpected: 102400
            )
            // Should throttle updates to avoid spam
        }
        
        await delegate.downloadDidComplete(url: url, localURL: url)
    }
    
    // MARK: - PersistentModelRegistry Tests
    
    func testPersistentModelRegistryActorIsolation() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("embedkit-test-\(UUID().uuidString)")
        
        let registry = try await PersistentModelRegistry(storageDirectory: tempDir)
        
        // Test concurrent writes
        await withTaskGroup(of: Void.self) { group in
            for i in 0..<10 {
                group.addTask {
                    let version = ModelVersion(
                        identifier: .miniLM_L6_v2,
                        version: "1.0.\(i)",
                        buildNumber: i,
                        filePath: tempDir.appendingPathComponent("model\(i).mlmodel"),
                        fileSize: 1024 * 1024
                    )
                    try? await registry.addModelVersion(version)
                }
            }
        }
        
        // Verify all versions were added
        let versions = try await registry.getAllVersions(for: .miniLM_L6_v2)
        XCTAssertGreaterThanOrEqual(versions.count, 10)
        
        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }
    
    func testPersistentModelRegistryTransactions() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("embedkit-test-\(UUID().uuidString)")
        
        let registry = try await PersistentModelRegistry(storageDirectory: tempDir)
        
        // Add versions
        let version1 = ModelVersion(
            identifier: .miniLM_L6_v2,
            version: "1.0.0",
            buildNumber: 1,
            filePath: tempDir.appendingPathComponent("model1.mlmodel"),
            fileSize: 1024 * 1024
        )
        
        let version2 = ModelVersion(
            identifier: .miniLM_L6_v2,
            version: "1.0.1",
            buildNumber: 2,
            filePath: tempDir.appendingPathComponent("model2.mlmodel"),
            fileSize: 1024 * 1024
        )
        
        try await registry.addModelVersion(version1)
        try await registry.addModelVersion(version2)
        
        // Set active version
        try await registry.setActiveVersion(
            identifier: .miniLM_L6_v2,
            versionId: version2.id
        )
        
        // Verify active version
        let activeVersion = try await registry.getActiveModelVersion(for: .miniLM_L6_v2)
        XCTAssertEqual(activeVersion?.id, version2.id)
        
        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }
    
    // MARK: - TelemetryManager Tests
    
    func testTelemetryManagerSendable() async throws {
        let telemetry = TelemetryManager.shared.system
        
        // Update configuration
        await telemetry.updateConfiguration(.development)
        
        // Test concurrent event recording
        await withTaskGroup(of: Void.self) { group in
            for i in 0..<100 {
                group.addTask {
                    await telemetry.recordEvent("test_event", properties: [
                        "index": i,
                        "thread": Thread.current.description
                    ])
                }
            }
        }
        
        // Get events
        let events = await telemetry.getEvents()
        XCTAssertGreaterThan(events.count, 0)
    }
    
    func testTelemetryMetrics() async throws {
        let telemetry = TelemetryManager.shared.system
        await telemetry.updateConfiguration(.development)
        
        // Record metrics
        for i in 0..<10 {
            await telemetry.recordMetric("test_metric", value: Double(i), unit: .count)
        }
        
        // Test duration tracking
        let result = await telemetry.recordDuration("operation_duration") {
            try? await Task.sleep(nanoseconds: 100_000_000) // 0.1 seconds
            return 42
        }
        XCTAssertEqual(result, 42)
        
        // Get metrics summary
        let summary = await telemetry.getMetricsSummary()
        XCTAssertNotNil(summary["test_metric"])
        XCTAssertNotNil(summary["operation_duration"])
        
        if let testMetric = summary["test_metric"] {
            XCTAssertEqual(testMetric.count, 10)
            XCTAssertEqual(testMetric.average, 4.5, accuracy: 0.01)
        }
    }
}

// MARK: - Configuration Tests

final class EmbedKitConfigTests: XCTestCase {
    
    func testConfigBuilder() throws {
        let config = EmbedKitConfig.builder()
            .model(.miniLM_L6_v2)
            .maxSequenceLength(256)
            .batchSize(16)
            .metalAcceleration(true)
            .enableCache(true)
            .cacheSize(100 * 1024 * 1024)
            .build()
        
        XCTAssertEqual(config.modelIdentifier, .miniLM_L6_v2)
        XCTAssertEqual(config.maxSequenceLength, 256)
        XCTAssertEqual(config.batchSize, 16)
        XCTAssertTrue(config.useMetalAcceleration)
        XCTAssertTrue(config.cacheEnabled)
        XCTAssertEqual(config.maxCacheSize, 100 * 1024 * 1024)
    }
    
    func testPresetConfigurations() throws {
        // Test production preset
        let production = EmbedKitConfig.production()
        XCTAssertTrue(production.verifyModelSignatures)
        XCTAssertTrue(production.telemetryEnabled)
        XCTAssertEqual(production.telemetrySamplingRate, 0.1)
        
        // Test development preset
        let development = EmbedKitConfig.development()
        XCTAssertFalse(development.verifyModelSignatures)
        XCTAssertTrue(development.performanceLoggingEnabled)
        XCTAssertEqual(development.telemetrySamplingRate, 1.0)
        
        // Test high performance preset
        let highPerf = EmbedKitConfig.highPerformance()
        XCTAssertEqual(highPerf.batchSize, 64)
        XCTAssertEqual(highPerf.maxConcurrentOperations, 8)
        XCTAssertFalse(highPerf.telemetryEnabled)
        
        // Test memory constrained preset
        let memConstrained = EmbedKitConfig.memoryConstrained()
        XCTAssertEqual(memConstrained.batchSize, 8)
        XCTAssertFalse(memConstrained.useMetalAcceleration)
        XCTAssertFalse(memConstrained.cacheEnabled)
        XCTAssertNotNil(memConstrained.memoryLimit)
    }
    
    func testConfigValidation() throws {
        // Valid config
        let validConfig = EmbedKitConfig.builder()
            .maxSequenceLength(512)
            .batchSize(32)
            .build()
        
        XCTAssertNoThrow(try validConfig.validate())
        
        // Invalid sequence length
        let invalidSeqLength = EmbedKitConfig.builder()
            .maxSequenceLength(0)
            .build()
        
        XCTAssertThrowsError(try invalidSeqLength.validate()) { error in
            guard case ConfigurationError.validationFailed(let message) = error else {
                XCTFail("Wrong error type")
                return
            }
            XCTAssertTrue(message.contains("maxSequenceLength"))
        }
        
        // Invalid batch size
        let invalidBatchSize = EmbedKitConfig.builder()
            .batchSize(300)
            .build()
        
        XCTAssertThrowsError(try invalidBatchSize.validate()) { error in
            guard case ConfigurationError.validationFailed(let message) = error else {
                XCTFail("Wrong error type")
                return
            }
            XCTAssertTrue(message.contains("batchSize"))
        }
    }
    
    func testEnvironmentConfiguration() throws {
        // Set environment variables
        let env = ProcessInfo.processInfo.environment
        
        // This would normally be done in a test harness
        // setenv("EMBEDKIT_MODEL", "miniLM-L6-v2", 1)
        // setenv("EMBEDKIT_MAX_SEQUENCE_LENGTH", "256", 1)
        // setenv("EMBEDKIT_USE_METAL", "true", 1)
        
        // Test would verify environment parsing
        // let config = try EmbedKitConfig.fromEnvironment()
    }
}

// MARK: - Dependency Injection Tests

final class DependencyContainerTests: XCTestCase {
    
    override func setUp() async throws {
        // Clear container before each test
        await DependencyContainer.shared.clear()
    }
    
    func testBasicRegistrationAndResolution() async throws {
        let container = DependencyContainer.shared
        
        // Register a simple value
        await container.register(String.self, instance: "Hello, EmbedKit!")
        
        // Resolve the value
        let resolved = try await container.resolve(String.self)
        XCTAssertEqual(resolved, "Hello, EmbedKit!")
    }
    
    func testFactoryRegistration() async throws {
        let container = DependencyContainer.shared
        
        // Register a factory
        var creationCount = 0
        await container.register(Int.self) {
            creationCount += 1
            return creationCount
        }
        
        // Each resolution creates a new instance
        let value1 = try await container.resolve(Int.self)
        let value2 = try await container.resolve(Int.self)
        
        XCTAssertEqual(value1, 1)
        XCTAssertEqual(value2, 2)
    }
    
    func testSingletonRegistration() async throws {
        let container = DependencyContainer.shared
        
        // Register a singleton
        var creationCount = 0
        await container.registerSingleton(UUID.self) {
            creationCount += 1
            return UUID()
        }
        
        // Multiple resolutions return the same instance
        let uuid1 = try await container.resolve(UUID.self)
        let uuid2 = try await container.resolve(UUID.self)
        
        XCTAssertEqual(uuid1, uuid2)
        XCTAssertEqual(creationCount, 1)
    }
    
    func testProtocolRegistration() async throws {
        let container = DependencyContainer.shared
        
        // Register protocol implementation
        await container.register(
            any TokenizerProtocol.self,
            implementation: MockTokenizer.self
        ) {
            MockTokenizer()
        }
        
        // Resolve protocol
        let tokenizer = try await container.resolve((any TokenizerProtocol).self)
        XCTAssertNotNil(tokenizer)
        XCTAssertEqual(tokenizer.maxSequenceLength, 512)
    }
    
    func testDependencyBuilder() async throws {
        let container = await DependencyBuilder.builder()
            .register(String.self, instance: "Test")
            .register(Int.self) { 42 }
            .singleton(Date.self) { Date() }
            .build()
        
        let string = try await container.resolve(String.self)
        let int = try await container.resolve(Int.self)
        let date = try await container.resolve(Date.self)
        
        XCTAssertEqual(string, "Test")
        XCTAssertEqual(int, 42)
        XCTAssertNotNil(date)
    }
    
    func testOptionalResolution() async throws {
        let container = DependencyContainer.shared
        
        // Try to resolve unregistered type
        let optional = await container.resolveOptional(URL.self)
        XCTAssertNil(optional)
        
        // Register and resolve
        await container.register(URL.self, instance: URL(string: "https://example.com")!)
        let resolved = await container.resolveOptional(URL.self)
        XCTAssertNotNil(resolved)
    }
    
    func testPropertyWrappers() async throws {
        let container = DependencyContainer.shared
        await container.register(String.self, instance: "Injected Value")
        
        struct TestStruct {
            @Injected(String.self) var injectedString
            @OptionalInjected(Int.self) var optionalInt
        }
        
        let test = TestStruct()
        let string = try await test.injectedString
        let int = await test.optionalInt
        
        XCTAssertEqual(string, "Injected Value")
        XCTAssertNil(int)
    }
}

// MARK: - Component Metadata Tests

final class CommandMetadataTests: XCTestCase {
    
    func testMutableMetadata() async throws {
        let metadata = MutableMetadata(value: 0)
        
        // Test concurrent updates
        await withTaskGroup(of: Void.self) { group in
            for i in 1...10 {
                group.addTask {
                    await metadata.update(i)
                }
            }
        }
        
        let finalValue = await metadata.value
        XCTAssertGreaterThanOrEqual(finalValue, 1)
        XCTAssertLessThanOrEqual(finalValue, 10)
    }
    
    func testVersionedMetadata() async throws {
        let metadata = VersionedMetadata(value: "initial")
        
        // Make updates
        await metadata.update("version1")
        await metadata.update("version2")
        await metadata.update("version3")
        
        // Check version
        let version = await metadata.version
        XCTAssertEqual(version, 3)
        
        // Check history
        let history = await metadata.getHistory()
        XCTAssertEqual(history.count, 4) // initial + 3 updates
        
        // Check specific version
        let v1Value = await metadata.value(at: 1)
        XCTAssertEqual(v1Value, "version1")
    }
    
    func testCachedMetadata() async throws {
        var refreshCount = 0
        
        let metadata = CachedMetadata(
            value: "initial",
            ttl: 0.1  // 100ms TTL
        ) {
            refreshCount += 1
            return "refreshed-\(refreshCount)"
        }
        
        // Initial value
        let value1 = try await metadata.value
        XCTAssertEqual(value1, "initial")
        
        // Wait for expiration
        try await Task.sleep(nanoseconds: 200_000_000) // 200ms
        
        // Should trigger refresh
        let value2 = try await metadata.value
        XCTAssertEqual(value2, "refreshed-1")
        XCTAssertEqual(refreshCount, 1)
    }
    
    func testObservableMetadata() async throws {
        let metadata = ObservableMetadata(value: 0)
        
        var observedChanges: [(old: Int, new: Int)] = []
        let observerId = await metadata.addObserver { old, new in
            observedChanges.append((old: old, new: new))
        }
        
        // Make updates
        await metadata.update(1)
        await metadata.update(2)
        await metadata.update(3)
        
        // Small delay to ensure observers complete
        try await Task.sleep(nanoseconds: 10_000_000) // 10ms
        
        // Verify observations
        XCTAssertEqual(observedChanges.count, 3)
        XCTAssertEqual(observedChanges[0].old, 0)
        XCTAssertEqual(observedChanges[0].new, 1)
        XCTAssertEqual(observedChanges[2].old, 2)
        XCTAssertEqual(observedChanges[2].new, 3)
        
        // Remove observer
        await metadata.removeObserver(observerId)
    }
}

// MARK: - Mock Types

struct MockTokenizer: TokenizerProtocol {
    let maxSequenceLength = 512
    let vocabularySize = 30522
    
    func tokenize(_ text: String) async throws -> [Int] {
        // Simple mock tokenization
        return Array(repeating: 101, count: min(text.count, maxSequenceLength))
    }
    
    func tokenizeBatch(_ texts: [String]) async throws -> [[Int]] {
        try await withThrowingTaskGroup(of: [Int].self) { group in
            for text in texts {
                group.addTask {
                    try await self.tokenize(text)
                }
            }
            
            var results: [[Int]] = []
            for try await tokens in group {
                results.append(tokens)
            }
            return results
        }
    }
    
    func detokenize(_ tokens: [Int]) async throws -> String {
        // Simple mock detokenization
        return String(repeating: "word ", count: tokens.count)
    }
}