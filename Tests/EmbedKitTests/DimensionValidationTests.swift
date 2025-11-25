import Testing
@testable import EmbedKit

actor DimBackend: CoreMLProcessingBackend {
    private(set) var isLoaded: Bool = false
    var memoryUsage: Int64 { 0 }
    let dim: Int
    
    init(dim: Int) { self.dim = dim }
    
    func load() async throws { isLoaded = true }
    func unload() async throws { isLoaded = false }
    func process(_ input: CoreMLInput) async throws -> CoreMLOutput {
        let tokens = input.tokenIDs.count
        let values: [Float] = Array(repeating: 0, count: tokens * dim)
        return CoreMLOutput(values: values, shape: [tokens, dim])
    }
    func processBatch(_ inputs: [CoreMLInput]) async throws -> [CoreMLOutput] {
        var outs: [CoreMLOutput] = []
        outs.reserveCapacity(inputs.count)
        for inp in inputs { outs.append(try await process(inp)) }
        return outs
    }
}

@Suite("Dimension Validation")
struct DimensionValidationTests {
@Test
func appleModel_dimensionMismatchThrows() async {
    let backend = DimBackend(dim: 8)
    let tokenizer = SimpleTokenizer()
    let cfg = EmbeddingConfiguration()
    let model = AppleEmbeddingModel(
        backend: backend,
        tokenizer: tokenizer,
        configuration: cfg,
        id: ModelID(provider: "test", name: "dim", version: "1.0"),
        dimensions: 4,
        device: .cpu
    )
    
    do {
        _ = try await model.embed("a b c")
        #expect(Bool(false), "Expected dimension mismatch error")
    } catch {
        guard let ek = error as? EmbedKitError else { #expect(Bool(false), "Unexpected error type: \(error)"); return }
        switch ek {
        case .dimensionMismatch(let expected, let got):
            #expect(expected == 4)
            #expect(got == 8)
        default:
            #expect(Bool(false), "Unexpected error: \(ek)")
        }
    }
}
}
