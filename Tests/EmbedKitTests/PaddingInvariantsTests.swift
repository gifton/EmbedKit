import Testing
@testable import EmbedKit

@Suite("Padding Invariants")
struct PaddingInvariantsTests {
    actor NoOpBackend: CoreMLProcessingBackend {
        private(set) var isLoaded: Bool = false
        var memoryUsage: Int64 { 0 }
        func load() async throws { isLoaded = true }
        func unload() async throws { isLoaded = false }
        func process(_ input: CoreMLInput) async throws -> CoreMLOutput {
            let dim = 4
            return CoreMLOutput(values: Array(repeating: 0, count: input.tokenIDs.count * dim), shape: [input.tokenIDs.count, dim])
        }
        func processBatch(_ inputs: [CoreMLInput]) async throws -> [CoreMLOutput] {
            let dim = 4
            return inputs.map { inp in
                CoreMLOutput(values: Array(repeating: 0, count: inp.tokenIDs.count * dim), shape: [inp.tokenIDs.count, dim])
            }
        }
    }

    struct MismatchTokenizer: Tokenizer {
        var vocabularySize: Int { 10 }
        var specialTokens: SpecialTokens { SpecialTokens() }
        func encode(_ text: String, config: TokenizerConfig) async throws -> TokenizedText {
            // Two ids, but only one mask entry (intentional mismatch)
            return TokenizedText(ids: [1,2], tokens: ["1","2"], attentionMask: [1])
        }
        func decode(_ ids: [Int]) async throws -> String { ids.map(String.init).joined(separator: " ") }
    }

    @Test
    func singleEmbed_idsMaskMismatch_throws() async {
        let backend = NoOpBackend()
        let tokenizer = MismatchTokenizer()
        let cfg = EmbeddingConfiguration()
        let model = AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: cfg, id: ModelID(provider: "test", name: "inv-single", version: "1.0"), dimensions: 4, device: .cpu)
        do {
            _ = try await model.embed("x")
            #expect(Bool(false), "Expected invalidConfiguration on ids/mask mismatch")
        } catch {
            guard let ek = error as? EmbedKitError else { return #expect(Bool(false), "Unexpected error: \(error)") }
            switch ek {
            case .invalidConfiguration:
                #expect(true)
            default:
                #expect(Bool(false), "Unexpected error: \(ek)")
            }
        }
    }

    @Test
    func embedBatch_idsMaskMismatch_throws() async {
        let backend = NoOpBackend()
        let tokenizer = MismatchTokenizer()
        let cfg = EmbeddingConfiguration(
            paddingStrategy: .batch
        )
        let model = AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: cfg, id: ModelID(provider: "test", name: "inv-batch", version: "1.0"), dimensions: 4, device: .cpu)
        do {
            _ = try await model.embedBatch(["x", "y"], options: BatchOptions())
            #expect(Bool(false), "Expected invalidConfiguration on ids/mask mismatch in batch")
        } catch {
            guard let ek = error as? EmbedKitError else { return #expect(Bool(false), "Unexpected error: \(error)") }
            switch ek {
            case .invalidConfiguration:
                #expect(true)
            default:
                #expect(Bool(false), "Unexpected error: \(ek)")
            }
        }
    }
}

