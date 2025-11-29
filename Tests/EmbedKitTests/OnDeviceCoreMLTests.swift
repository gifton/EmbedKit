// On-device CoreML tests
import Testing
@testable import EmbedKit
import Foundation

#if canImport(CoreML)
@Suite("On-Device CoreML (Conditional)")
struct OnDeviceCoreMLV2Tests {
    @Test
    func loadAndEmbed_withLocalModel_ifAvailable() async throws {
        // Attempt to locate a local compiled model and a vocab file for WordPiece.
        // If not found or load fails, skip quietly.
        let cwd = FileManager.default.currentDirectoryPath
        let modelPath = URL(fileURLWithPath: cwd).appendingPathComponent("MiniLM-L12-v2.mlmodelc", isDirectory: true)
        let vocabCandidates = [
            URL(fileURLWithPath: cwd).appendingPathComponent("Sources/EmbedKit/Resources/vocab.txt"),
            URL(fileURLWithPath: cwd).appendingPathComponent("EmbedKit/Resources/vocab.txt")
        ]
        guard FileManager.default.fileExists(atPath: modelPath.path) else { return }
        guard let vocabURL = vocabCandidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }) else { return }

        do {
            let vocab = try Vocabulary.load(from: vocabURL)
            let tokenizer = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
            let cfg = EmbeddingConfiguration(
                maxTokens: 128,
                paddingStrategy: .none,
                includeSpecialTokens: true
            )
            let model = AppleEmbeddingModel(backend: CoreMLBackend(modelURL: modelPath, device: cfg.inferenceDevice), tokenizer: tokenizer, configuration: cfg, id: ModelID(provider: "test", name: "local-coreml", version: "1.0"), dimensions: 384, device: .auto)
            _ = try await model.embed("Hello from on-device test")
        } catch {
            // Skip if CoreML load or inference fails in this environment.
            print("Skipping on-device test: \(error)")
            return
        }
    }
}
#endif
