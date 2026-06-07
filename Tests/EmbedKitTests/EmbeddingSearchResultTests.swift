// Tests for EmbeddingSearchResult distance→similarity conversion.
// Regression coverage for the dot-product similarity inversion (audit 4.4):
// producers store distance = -dotProduct, so similarity must increase as
// distance decreases.
import Testing
import Foundation
import VectorCore
@testable import EmbedKit

@Suite("EmbeddingSearchResult - distance→similarity")
struct EmbeddingSearchResultSimilarityTests {

    // Helper: the metric-aware internal initializer is what the search path uses.
    private func similarity(distance: Float, metric: SupportedDistanceMetric) -> Float {
        EmbeddingSearchResult(id: "x", distance: distance, metric: metric).similarity
    }

    @Test("Dot product: identical vectors (dot=1, distance=-1) → similarity 1.0")
    func dotProductIdentical() {
        // distance = -dotProduct, so an identical normalized pair (dot=1) → distance=-1
        let sim = similarity(distance: -1.0, metric: .dotProduct)
        #expect(abs(sim - 1.0) < 1e-6)
    }

    @Test("Dot product: opposite vectors (dot=-1, distance=1) → similarity 0.0")
    func dotProductOpposite() {
        let sim = similarity(distance: 1.0, metric: .dotProduct)
        #expect(abs(sim - 0.0) < 1e-6)
    }

    @Test("Dot product: orthogonal vectors (dot=0, distance=0) → similarity 0.5")
    func dotProductOrthogonal() {
        let sim = similarity(distance: 0.0, metric: .dotProduct)
        #expect(abs(sim - 0.5) < 1e-6)
    }

    @Test("Dot product: similarity is monotonically decreasing in distance")
    func dotProductMonotonic() {
        // More relevant (smaller distance = larger dot) must score higher.
        let near = similarity(distance: -0.8, metric: .dotProduct) // dot = 0.8
        let far = similarity(distance: 0.8, metric: .dotProduct)   // dot = -0.8
        #expect(near > far)
    }

    @Test("Cosine: identical (distance=0) → similarity 1.0")
    func cosineIdentical() {
        let sim = similarity(distance: 0.0, metric: .cosine)
        #expect(abs(sim - 1.0) < 1e-6)
    }
}
