// Tests for ONNX Backend
import Testing
import Foundation
@testable import EmbedKit
@testable import EmbedKitONNX

// MARK: - ONNX Backend Initialization Tests

@Suite("ONNX Backend - Initialization")
struct ONNXBackendInitTests {

    @Test("Backend initializes with URL and default config")
    func initWithDefaults() async {
        let url = URL(fileURLWithPath: "/tmp/model.onnx")
        let backend = ONNXBackend(modelPath: url, config: .default)

        let isLoaded = await backend.isLoaded
        #expect(isLoaded == false)
    }

    @Test("Backend initializes with custom config")
    func initWithCustomConfig() async {
        let url = URL(fileURLWithPath: "/tmp/model.onnx")
        let config = ONNXBackendConfiguration(
            intraOpNumThreads: 4,
            useCoreMLProvider: false
        )
        let backend = ONNXBackend(modelPath: url, config: config)

        let isLoaded = await backend.isLoaded
        #expect(isLoaded == false)
    }

    @Test("Backend reports not loaded initially")
    func notLoadedInitially() async {
        let backend = ONNXBackend(
            modelPath: URL(fileURLWithPath: "/tmp/test.onnx"),
            config: .default
        )

        let isLoaded = await backend.isLoaded
        let memUsage = await backend.memoryUsage

        #expect(isLoaded == false)
        #expect(memUsage == 0)
    }
}

// MARK: - ONNX Backend Error Tests

@Suite("ONNX Backend - Errors")
struct ONNXBackendErrorTests {

    @Test("Process throws when not loaded")
    func processThrowsWhenNotLoaded() async throws {
        let backend = ONNXBackend(
            modelPath: URL(fileURLWithPath: "/tmp/nonexistent.onnx"),
            config: .default
        )

        let input = ONNXInput(tokenIDs: [1, 2, 3], attentionMask: [1, 1, 1])

        await #expect(throws: ONNXBackendError.self) {
            try await backend.process(input)
        }
    }

    @Test("Load fails for nonexistent file")
    func loadFailsForMissingFile() async throws {
        let backend = ONNXBackend(
            modelPath: URL(fileURLWithPath: "/nonexistent/path/model.onnx"),
            config: .default
        )

        await #expect(throws: ONNXBackendError.self) {
            try await backend.load()
        }
    }

    @Test("ProcessBatch fails when not loaded")
    func processBatchFailsWhenNotLoaded() async throws {
        let backend = ONNXBackend(
            modelPath: URL(fileURLWithPath: "/tmp/model.onnx"),
            config: .default
        )

        let inputs = [
            ONNXInput(tokenIDs: [1, 2], attentionMask: [1, 1]),
            ONNXInput(tokenIDs: [3, 4], attentionMask: [1, 1])
        ]

        await #expect(throws: ONNXBackendError.self) {
            try await backend.processBatch(inputs)
        }
    }
}

// MARK: - ONNX Backend Lifecycle Tests

@Suite("ONNX Backend - Lifecycle")
struct ONNXBackendLifecycleTests {

    @Test("Unload resets state")
    func unloadResetsState() async throws {
        let backend = ONNXBackend(
            modelPath: URL(fileURLWithPath: "/tmp/model.onnx"),
            config: .default
        )

        // Unload even without loading should work
        try await backend.unload()

        let isLoaded = await backend.isLoaded
        #expect(isLoaded == false)
    }

    @Test("Multiple unloads are safe")
    func multipleUnloads() async throws {
        let backend = ONNXBackend(
            modelPath: URL(fileURLWithPath: "/tmp/model.onnx"),
            config: .default
        )

        // Multiple unloads should be idempotent
        try await backend.unload()
        try await backend.unload()
        try await backend.unload()

        let isLoaded = await backend.isLoaded
        #expect(isLoaded == false)
    }
}
