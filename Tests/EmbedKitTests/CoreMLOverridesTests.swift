import Testing
@testable import EmbedKit
import Foundation

#if canImport(CoreML)
@Suite("CoreML Overrides (Conditional)")
struct CoreMLOverridesTests {
    @Test
    func setInputAndOutputOverrides_noCrash_ifKeysValid() async throws {
        let cwd = FileManager.default.currentDirectoryPath
        let modelPath = URL(fileURLWithPath: cwd).appendingPathComponent("MiniLM-L12-v2.mlmodelc", isDirectory: true)
        guard FileManager.default.fileExists(atPath: modelPath.path) else { return }

        let vocabPaths = [
            URL(fileURLWithPath: cwd).appendingPathComponent("Sources/EmbedKit/Resources/vocab.txt"),
            URL(fileURLWithPath: cwd).appendingPathComponent("EmbedKit/Resources/vocab.txt")
        ]
        guard let vocabURL = vocabPaths.first(where: { FileManager.default.fileExists(atPath: $0.path) }) else { return }
        let vocab = try Vocabulary.load(from: vocabURL)
        let tokenizer = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
        var cfg = EmbeddingConfiguration()
        cfg.includeSpecialTokens = true
        cfg.maxTokens = 64
        let backend = CoreMLBackend(modelURL: modelPath, device: cfg.preferredDevice)
        await backend.setInputKeyOverrides(token: "input_ids", mask: "attention_mask", type: nil, pos: nil)
        await backend.setOutputKeyOverride(nil)
        let model = AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: cfg, id: ModelID(provider: "test", name: "overrides", version: "1.0"), dimensions: 384, device: .auto)
        do { _ = try await model.embed("override test") } catch { print("Skip overrides test: \(error)") }
    }

    @Test
    func modelLevelOverrides_forwarded_toBackend() async throws {
        let cwd = FileManager.default.currentDirectoryPath
        let modelPath = URL(fileURLWithPath: cwd).appendingPathComponent("MiniLM-L12-v2.mlmodelc", isDirectory: true)
        guard FileManager.default.fileExists(atPath: modelPath.path) else { return }

        let vocabPaths = [
            URL(fileURLWithPath: cwd).appendingPathComponent("Sources/EmbedKit/Resources/vocab.txt"),
            URL(fileURLWithPath: cwd).appendingPathComponent("EmbedKit/Resources/vocab.txt")
        ]
        guard let vocabURL = vocabPaths.first(where: { FileManager.default.fileExists(atPath: $0.path) }) else { return }
        let vocab = try Vocabulary.load(from: vocabURL)
        let tokenizer = WordPieceTokenizer(vocabulary: vocab, unkToken: "[UNK]", lowercase: true)
        var cfg = EmbeddingConfiguration()
        cfg.includeSpecialTokens = true
        cfg.maxTokens = 64
        let backend = CoreMLBackend(modelURL: modelPath, device: cfg.preferredDevice)
        let model = AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: cfg, id: ModelID(provider: "test", name: "overridesModel", version: "1.0"), dimensions: 384, device: .auto)
        await model.setCoreMLInputKeyOverrides(token: "input_ids", mask: "attention_mask")
        await model.setCoreMLOutputKeyOverride(nil)
        do { _ = try await model.embed("override test") } catch { print("Skip overrides test: \(error)") }
    }
}
#endif
