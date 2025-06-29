import Testing
import Foundation
@testable import EmbedKit

@Suite("Model Download Tests")
struct ModelDownloadTests {
    
    @Test("Model source initialization")
    func testModelSourceInitialization() {
        let url = URL(string: "https://example.com/model.mlpackage")!
        let source = ModelDownloader.ModelSource(
            url: url,
            expectedChecksum: "abc123",
            expectedSize: 1024 * 1024,
            modelIdentifier: .miniLM_L6_v2
        )
        
        #expect(source.url == url)
        #expect(source.expectedChecksum == "abc123")
        #expect(source.expectedSize == 1024 * 1024)
        #expect(source.modelIdentifier == .miniLM_L6_v2)
    }
    
    @Test("Download configuration defaults")
    func testDownloadConfigurationDefaults() {
        let config = ModelDownloader.DownloadConfiguration()
        
        #expect(config.maxRetries == 3)
        #expect(config.timeoutInterval == 300)
        #expect(config.verifyChecksum == true)
    }
    
    @Test("Local model URL generation")
    func testLocalModelURLGeneration() async {
        let downloader = ModelDownloader()
        
        let url1 = await downloader.localModelURL(for: .miniLM_L6_v2)
        #expect(url1.pathComponents.contains("EmbedKitModels"))
        #expect(url1.pathComponents.contains("all-MiniLM-L6-v2"))
        #expect(url1.lastPathComponent == "1.mlmodelc")
        
        let customId = ModelIdentifier(family: "test", variant: "small", version: "v2")
        let url2 = await downloader.localModelURL(for: customId)
        #expect(url2.pathComponents.contains("test"))
        #expect(url2.pathComponents.contains("small"))
        #expect(url2.lastPathComponent == "v2.mlmodelc")
    }
    
    @Test("HuggingFace registry model listing")
    func testHuggingFaceRegistryListing() async {
        let registry = HuggingFaceModelRegistry()
        let models = await registry.listModels()
        
        #expect(models.count > 0)
        
        // Check known models are included
        let modelIds = models.map { $0.identifier }
        #expect(modelIds.contains(.miniLM_L6_v2))
        
        // Check metadata
        for model in models {
            #expect(model.metadata["source"] == "huggingface")
            #expect(model.metadata["dimensions"] != nil)
            #expect(model.metadata["max_length"] != nil)
        }
    }
    
    @Test("HuggingFace registry model lookup")
    func testHuggingFaceRegistryModelLookup() async {
        let registry = HuggingFaceModelRegistry()
        
        // Test known model
        let entry = await registry.getModel(.miniLM_L6_v2)
        #expect(entry != nil)
        #expect(entry?.identifier == .miniLM_L6_v2)
        #expect(entry?.downloadURL != nil)
        #expect(entry?.metadata["huggingface_id"] == "sentence-transformers/all-MiniLM-L6-v2")
        #expect(entry?.metadata["dimensions"] == "384")
        
        // Test unknown model
        let unknownId = ModelIdentifier(family: "unknown", variant: "model", version: "v1")
        let unknownEntry = await registry.getModel(unknownId)
        #expect(unknownEntry == nil)
    }
    
    @Test("Model manager initialization")
    func testModelManagerInitialization() async {
        let manager = ModelManager()
        
        // List available models (should at least have bundled ones if any)
        let models = try? await manager.listAvailableModels()
        #expect(models != nil)
    }
    
    @Test("Model manager loading options")
    func testModelManagerLoadingOptions() {
        let options = ModelManager.ManagerLoadingOptions(
            allowDownload: false,
            verifySignature: false
        )
        
        #expect(options.allowDownload == false)
        #expect(options.verifySignature == false)
        #expect(options.registry == nil)
    }
}

// Mock download delegate for testing
final class MockDownloadDelegate: ModelDownloader.DownloadDelegate, @unchecked Sendable {
    var didStartCalled = false
    var didProgressCalled = false
    var didCompleteCalled = false
    var didFailCalled = false
    
    var lastProgress: (bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpected: Int64)?
    var lastError: Error?
    var lastLocalURL: URL?
    
    func downloadDidStart(url: URL) {
        didStartCalled = true
    }
    
    func downloadDidProgress(bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpected: Int64) {
        didProgressCalled = true
        lastProgress = (bytesWritten, totalBytesWritten, totalBytesExpected)
    }
    
    func downloadDidComplete(url: URL, localURL: URL) {
        didCompleteCalled = true
        lastLocalURL = localURL
    }
    
    func downloadDidFail(url: URL, error: Error) {
        didFailCalled = true
        lastError = error
    }
}

@Suite("Model Download Network Tests", .disabled("Requires network"))
struct ModelDownloadNetworkTests {
    
    @Test("Download small test model")
    func testDownloadSmallModel() async throws {
        let delegate = MockDownloadDelegate()
        let downloader = ModelDownloader(delegate: delegate)
        
        // Use a small test file for testing
        let testURL = URL(string: "https://raw.githubusercontent.com/huggingface/swift-coreml-diffusers/main/LICENSE")!
        let source = ModelDownloader.ModelSource(
            url: testURL,
            modelIdentifier: ModelIdentifier(family: "test", variant: "license", version: "v1")
        )
        
        let localURL = try await downloader.downloadModel(from: source)
        
        #expect(FileManager.default.fileExists(atPath: localURL.path))
        #expect(delegate.didStartCalled)
        #expect(delegate.didCompleteCalled)
        
        // Clean up
        try? await downloader.deleteModel(source.modelIdentifier)
    }
    
    @Test("Model search functionality")
    func testModelSearch() async throws {
        let registry = HuggingFaceModelRegistry()
        
        let results = try await registry.searchModels(query: "minilm", limit: 3)
        
        #expect(results.count <= 3)
        for result in results {
            #expect(!result.modelId.isEmpty)
            #expect(result.downloads >= 0)
        }
    }
}