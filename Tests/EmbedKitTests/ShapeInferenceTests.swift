import Testing
@testable import EmbedKit

@Suite("Shape Inference")
struct ShapeInferenceTests {
    // Helper: create a model to call inferTokensAndDim (nonisolated instance method)
    private func makeModel(dim: Int) -> AppleEmbeddingModel {
        let backend: CoreMLProcessingBackend? = nil
        let tokenizer = SimpleTokenizer()
        let cfg = EmbeddingConfiguration()
        return AppleEmbeddingModel(backend: backend, tokenizer: tokenizer, configuration: cfg, id: ModelID(provider: "test", name: "shape", version: "1.0"), dimensions: dim, device: .cpu)
    }

    @Test
    func shape3D_batchSeqDim() async throws {
        let model = makeModel(dim: 4)
        let (t, d) = try await model.inferTokensAndDim(from: [1, 5, 4], valuesCount: 20)
        #expect(t == 5 && d == 4)
    }

    @Test
    func shape2D_seqDim() async throws {
        let model = makeModel(dim: 4)
        let (t, d) = try await model.inferTokensAndDim(from: [3, 4], valuesCount: 12)
        #expect(t == 3 && d == 4)
    }

    @Test
    func shape2D_batchDim() async throws {
        let model = makeModel(dim: 4)
        let (t, d) = try await model.inferTokensAndDim(from: [1, 4], valuesCount: 4)
        #expect(t == 1 && d == 4)
    }

    @Test
    func shape1D_dimOnly() async throws {
        let model = makeModel(dim: 4)
        let (t, d) = try await model.inferTokensAndDim(from: [4], valuesCount: 4)
        #expect(t == 1 && d == 4)
    }

    @Test
    func shape2D_dimSeq_swapped() async throws {
        let model = makeModel(dim: 4)
        let (t, d) = try await model.inferTokensAndDim(from: [4, 3], valuesCount: 12)
        #expect(t == 3 && d == 4)
    }

    @Test
    func fallback_factorization_usesKnownDim() async throws {
        let model = makeModel(dim: 4)
        // Shape metadata doesn't match valuesCount; rely on valuesCount and known dim
        let (t, d) = try await model.inferTokensAndDim(from: [2, 5], valuesCount: 12)
        #expect(t == 3 && d == 4)
    }

    @Test
    func invalid_unrecognizedShapeThrows() async throws {
        let model = makeModel(dim: 5)
        do {
            _ = try await model.inferTokensAndDim(from: [2, 3], valuesCount: 7)
            #expect(Bool(false), "Expected invalidConfiguration error")
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

