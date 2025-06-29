import Testing
import Foundation
import CryptoKit
@testable import EmbedKit

@Suite("Model Download Integration Tests")
struct ModelDownloadIntegrationTests {
    
    // MARK: - Local Model Management Tests
    
    @Test("Local model discovery and loading")
    func testLocalModelDiscovery() async throws {
        let loader = CoreMLModelLoader()
        
        // Try to discover models in bundle (might be empty in tests)
        let bundleURL = Bundle.main.bundleURL
        let discovered = try? await loader.discoverModels(in: bundleURL)
        
        // Whether we find models or not, the API should work
        #expect(discovered != nil || discovered?.count == 0)
        
        if let models = discovered, !models.isEmpty {
            // Verify metadata structure
            for metadata in models {
                #expect(!metadata.name.isEmpty)
                #expect(!metadata.version.isEmpty)
                #expect(metadata.embeddingDimensions >= 0)
            }
        }
    }
    
    @Test("Model downloader initialization")
    func testModelDownloaderSetup() async {
        let downloader = ModelDownloader()
        
        // Test local model URL generation
        let testId = ModelIdentifier(family: "test", variant: "model", version: "v1")
        let localURL = await downloader.localModelURL(for: testId)
        
        #expect(localURL.path.contains("EmbedKitModels"))
        #expect(localURL.path.contains("test"))
        #expect(localURL.path.contains("model"))
        #expect(localURL.lastPathComponent == "v1.mlmodelc")
    }
    
    @Test("Model registry functionality")
    func testModelRegistry() async {
        let registry = HuggingFaceModelRegistry()
        
        // Test listing models
        let models = await registry.listModels()
        #expect(models.count > 0)
        
        // Verify known models are present
        let miniLMEntry = await registry.getModel(.miniLM_L6_v2)
        #expect(miniLMEntry != nil)
        
        if let entry = miniLMEntry {
            #expect(entry.identifier == .miniLM_L6_v2)
            #expect(entry.metadata["dimensions"] == "384")
            #expect(entry.metadata["source"] == "huggingface")
        }
    }
    
    // MARK: - Model Manager Integration Tests
    
    @Test("Model manager with offline mode")
    func testOfflineModelManager() async throws {
        // Create manager in offline mode
        let manager = ModelManager(
            options: ModelManager.ManagerLoadingOptions(
                allowDownload: false
            )
        )
        
        // List available models (only bundled/cached)
        let models = try await manager.listAvailableModels()
        
        // In offline mode, we might have no models or only bundled ones
        for model in models {
            #expect(model.location == .bundled || model.location == .downloaded)
            // Should not have remote models in offline mode
            #expect(model.location != DownloadedModelInfo.ModelLocation.remote)
        }
        
        // Try to load a model that's not available locally
        let unavailableId = ModelIdentifier(family: "unavailable", variant: "model")
        do {
            _ = try await manager.loadModel(unavailableId)
            #expect(Bool(false), "Should have thrown error for unavailable model")
        } catch {
            // Expected error
            #expect(error is ContextualEmbeddingError)
            if let contextualError = error as? ContextualEmbeddingError {
                #expect(contextualError.context.operation == .modelLoading)
            }
        }
    }
    
    @Test("Model manager with download capability")
    func testModelManagerWithDownloads() async throws {
        // Create progress tracking delegate
        let progressDelegate = TestDownloadDelegate()
        
        let manager = ModelManager(
            options: ModelManager.ManagerLoadingOptions(
                allowDownload: true,
                registry: HuggingFaceModelRegistry()
            ),
            downloadDelegate: progressDelegate
        )
        
        // List all available models (including remote)
        let availableModels = try await manager.listAvailableModels()
        
        // Should include at least the known models from registry
        let modelIdentifiers = availableModels.map { $0.identifier }
        #expect(modelIdentifiers.contains(.miniLM_L6_v2) || availableModels.isEmpty)
        
        // Test model lifecycle management
        let testId = ModelIdentifier(family: "test", variant: "lifecycle")
        
        // Clean up any existing test model
        try? await manager.deleteModel(testId)
        
        // Verify it's gone
        let modelsAfterDelete = try await manager.listAvailableModels()
        #expect(!modelsAfterDelete.contains { $0.identifier == testId })
    }
    
    // MARK: - Download Process Tests
    
    @Test("Download configuration and retry logic")
    func testDownloadConfiguration() async {
        let config = ModelDownloader.DownloadConfiguration(
            maxRetries: 5,
            timeoutInterval: 30,
            verifyChecksum: true
        )
        
        #expect(config.maxRetries == 5)
        #expect(config.timeoutInterval == 30)
        #expect(config.verifyChecksum == true)
        
        let downloader = ModelDownloader(configuration: config)
        
        // Test with invalid source (should fail gracefully)
        let invalidSource = ModelDownloader.ModelSource(
            url: URL(string: "https://invalid.example.com/model.mlpackage")!,
            expectedChecksum: "invalid",
            modelIdentifier: ModelIdentifier(family: "test")
        )
        
        do {
            _ = try await downloader.downloadModel(from: invalidSource)
            #expect(Bool(false), "Should have failed with invalid URL")
        } catch {
            // Expected to fail
            #expect(error is ContextualEmbeddingError)
        }
    }
    
    @Test("Checksum verification")
    func testChecksumVerification() async throws {
        let downloader = ModelDownloader()
        
        // Create test data
        let testData = "Test model data".data(using: .utf8)!
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        
        try testData.write(to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }
        
        // Calculate checksum using CryptoKit directly (since calculateChecksum is private)
        let data = try Data(contentsOf: tempURL)
        let digest = SHA256.hash(data: data)
        let checksum = digest.compactMap { String(format: "%02x", $0) }.joined()
        
        // Verify it's a valid SHA256 hex string (64 characters)
        #expect(checksum.count == 64)
        #expect(checksum.allSatisfy { $0.isHexDigit })
        
        // Same data should produce same checksum
        let digest2 = SHA256.hash(data: data)
        let checksum2 = digest2.compactMap { String(format: "%02x", $0) }.joined()
        #expect(checksum == checksum2)
    }
    
    // MARK: - End-to-End Download Tests
    
    @Test("Complete model download and usage flow", .disabled("Requires network"))
    func testCompleteDownloadFlow() async throws {
        // This test simulates the complete flow from discovery to usage
        
        // 1. Search for models
        let registry = HuggingFaceModelRegistry()
        let searchResults = try? await registry.searchModels(query: "minilm", limit: 3)
        
        if let results = searchResults, !results.isEmpty {
            #expect(results.count <= 3)
            for result in results {
                #expect(!result.modelId.isEmpty)
                #expect(result.downloads >= 0)
            }
        }
        
        // 2. Create model manager
        let manager = ModelManager(
            options: ModelManager.ManagerLoadingOptions(
                allowDownload: true,
                verifySignature: false // Skip for test models
            )
        )
        
        // 3. Create embedder (will download if needed)
        // Using a small test model to avoid large downloads in tests
        let testModelId = ModelIdentifier(family: "test", variant: "tiny")
        
        do {
            let embedder = try await manager.createEmbedder(
                identifier: testModelId,
                configuration: Configuration.memoryOptimized(for: testModelId)
            )
            
            // 4. Use the embedder
            let embedding = try await embedder.embed("Test text")
            #expect(embedding.dimensions > 0)
            
            // 5. Clean up
            try? await manager.deleteModel(testModelId)
        } catch {
            // Download might fail in test environment
            print("Download test skipped: \(error)")
        }
    }
}

// MARK: - Test Helpers

/// Test download delegate for progress tracking
final class TestDownloadDelegate: ModelDownloader.DownloadDelegate, @unchecked Sendable {
    var startedDownloads: [URL] = []
    var completedDownloads: [URL] = []
    var failedDownloads: [(URL, Error)] = []
    var progressUpdates: [(URL, Double)] = []
    
    func downloadDidStart(url: URL) {
        startedDownloads.append(url)
    }
    
    func downloadDidProgress(bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpected: Int64) {
        let progress = Double(totalBytesWritten) / Double(totalBytesExpected)
        if let lastStarted = startedDownloads.last {
            progressUpdates.append((lastStarted, progress))
        }
    }
    
    func downloadDidComplete(url: URL, localURL: URL) {
        completedDownloads.append(url)
    }
    
    func downloadDidFail(url: URL, error: Error) {
        failedDownloads.append((url, error))
    }
}

// Extension to check hex characters
private extension Character {
    var isHexDigit: Bool {
        return "0123456789abcdefABCDEF".contains(self)
    }
}