import Foundation
@testable import EmbedKit

/// Test data fixtures for benchmarking
public enum TestFixtures {

    // MARK: - Tokenized Inputs

    /// Create valid tokenized input for testing
    /// - Parameter sequenceLength: Total sequence length (default 512)
    /// - Returns: Properly formatted TokenizedInput with [CLS], tokens, [SEP], and padding
    public static func createValidInput(sequenceLength: Int = 512) -> TokenizedInput {
        // Typical BERT input: [CLS] + tokens + [SEP] + padding
        var tokenIds = [101] // [CLS]
        tokenIds.append(contentsOf: Array(repeating: 100, count: min(sequenceLength - 2, 10))) // Some tokens
        tokenIds.append(102) // [SEP]
        tokenIds.append(contentsOf: Array(repeating: 0, count: max(0, sequenceLength - tokenIds.count))) // PAD

        // Attention mask: 1 for real tokens, 0 for padding
        let realTokenCount = 12 // [CLS] + 10 tokens + [SEP]
        var attentionMask = Array(repeating: 1, count: min(realTokenCount, sequenceLength))
        attentionMask.append(contentsOf: Array(repeating: 0, count: max(0, sequenceLength - attentionMask.count)))

        // Token type IDs (all 0 for single sequence)
        let tokenTypeIds = Array(repeating: 0, count: sequenceLength)

        return TokenizedInput(
            tokenIds: tokenIds,
            attentionMask: attentionMask,
            tokenTypeIds: tokenTypeIds,
            originalLength: realTokenCount
        )
    }

    /// Create batch of valid tokenized inputs
    /// - Parameters:
    ///   - count: Number of inputs to create
    ///   - sequenceLength: Sequence length for each input
    /// - Returns: Array of TokenizedInput
    public static func createBatchInputs(count: Int, sequenceLength: Int = 512) -> [TokenizedInput] {
        return (0..<count).map { _ in createValidInput(sequenceLength: sequenceLength) }
    }

    // MARK: - Embeddings

    /// Generate random embeddings for testing
    /// - Parameters:
    ///   - sequenceLength: Number of tokens
    ///   - dimensions: Embedding dimensions
    /// - Returns: 2D array of Float embeddings
    public static func generateRandomEmbeddings(
        sequenceLength: Int,
        dimensions: Int
    ) -> [[Float]] {
        return (0..<sequenceLength).map { _ in
            (0..<dimensions).map { _ in Float.random(in: -1...1) }
        }
    }

    /// Generate single random vector
    /// - Parameter dimensions: Vector dimensions
    /// - Returns: Array of random Float values
    public static func generateRandomVector(dimensions: Int) -> [Float] {
        return (0..<dimensions).map { _ in Float.random(in: -1...1) }
    }

    /// Generate normalized random vector (unit length)
    /// - Parameter dimensions: Vector dimensions
    /// - Returns: Normalized Float array
    public static func generateNormalizedVector(dimensions: Int) -> [Float] {
        let vector = generateRandomVector(dimensions: dimensions)
        let magnitude = sqrt(vector.reduce(0) { $0 + $1 * $1 })
        return vector.map { $0 / magnitude }
    }

    // MARK: - Text Samples

    /// Generate random texts for batch testing
    /// - Parameters:
    ///   - count: Number of texts to generate
    ///   - averageWords: Average number of words per text
    /// - Returns: Array of random text strings
    public static func generateRandomTexts(count: Int, averageWords: Int) -> [String] {
        let words = [
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "machine", "learning", "model", "process", "language", "efficiently",
            "swift", "programming", "embeddings", "vector", "semantic", "natural",
            "neural", "network", "transformer", "attention", "encode", "decode"
        ]

        return (0..<count).map { _ in
            let wordCount = Int.random(in: (averageWords-5)...(averageWords+5))
            return (0..<wordCount)
                .map { _ in words.randomElement()! }
                .joined(separator: " ")
        }
    }

    /// Short text sample (5-10 words)
    public static let shortText = "Quick performance test for embeddings"

    /// Typical text sample (30 words)
    public static let typicalText30Words = """
        The weather today is beautiful with clear skies and comfortable temperatures. \
        Perfect conditions for outdoor activities and spending time with family and friends. \
        Everyone is enjoying the day.
        """

    /// Medium text sample (50 words)
    public static let mediumText50Words = """
        Machine learning has revolutionized the way we process and understand natural language. \
        Modern embedding models can capture semantic relationships between words and sentences, \
        enabling applications ranging from search engines to recommendation systems. EmbedKit \
        provides a high-performance Swift implementation for on-device processing.
        """

    /// Long text sample (200 words)
    public static let longText200Words = """
        Machine learning has revolutionized the way we process and understand natural language. \
        Modern embedding models can capture semantic relationships between words and sentences, \
        enabling applications ranging from search engines to recommendation systems. EmbedKit \
        provides a high-performance Swift implementation that runs entirely on-device, ensuring \
        privacy and low latency. By leveraging CoreML and Metal acceleration, the framework can \
        process thousands of texts per second while maintaining numerical accuracy. The type-safe \
        embedding system prevents common bugs by catching dimension mismatches at compile time. \
        Integration with VectorIndex enables efficient similarity search and clustering operations. \
        The actor-based concurrency model ensures thread safety without sacrificing performance. \
        GPU acceleration through Metal compute shaders provides significant speedups for batch \
        operations, particularly for pooling and normalization tasks. The framework supports \
        multiple tokenization strategies including BERT WordPiece and Byte-Pair Encoding. \
        Production-ready features include LRU caching, memory profiling, and comprehensive error \
        handling. The modular architecture allows easy integration into existing Swift projects. \
        Performance benchmarks demonstrate sub-100ms latency for typical text processing on Apple \
        Silicon hardware. The framework is designed to scale from iPhone to Mac with consistent API.
        """

    // MARK: - Test Data Sets

    /// Real-world text samples for different domains
    public static let technicalTexts = [
        "Swift is a powerful programming language for iOS development",
        "Metal compute shaders enable GPU-accelerated machine learning",
        "CoreML provides on-device inference with Neural Engine support",
        "Actor-based concurrency ensures thread-safe async operations"
    ]

    public static let conversationalTexts = [
        "Hey, how are you doing today?",
        "I'm really excited about this new project!",
        "Thanks for your help, I appreciate it.",
        "Let's meet up for coffee sometime soon."
    ]

    public static let sentimentTexts = [
        "This product is absolutely amazing, I love it!",
        "Terrible experience, would not recommend at all.",
        "It's okay, nothing special but gets the job done.",
        "Outstanding quality and excellent customer service!"
    ]

    // MARK: - Edge Cases

    /// Edge case inputs for robustness testing
    public static let edgeCaseTexts = [
        "",                                    // Empty string
        "a",                                   // Single character
        String(repeating: "word ", count: 200), // Very long
        "😀🎉🚀💯",                              // Emoji only
        "Hello 世界",                           // Mixed scripts
        "test@email.com https://example.com",  // Special formats
        "!!@@##$$%%",                          // Punctuation only
        "     \t\n   "                         // Whitespace only
    ]

    // MARK: - Seeded Random Generation

    /// Deterministic random number generator for reproducible benchmarks
    public struct SeededRNG: RandomNumberGenerator {
        private var state: UInt64

        public init(seed: UInt64) {
            self.state = seed
        }

        public mutating func next() -> UInt64 {
            // Simple LCG for deterministic random numbers
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return state
        }
    }

    /// Generate reproducible random vector with seed
    /// - Parameters:
    ///   - dimensions: Vector dimensions
    ///   - seed: Random seed for reproducibility
    /// - Returns: Deterministic random vector
    public static func randomSeeded(dimensions: Int, seed: UInt64) -> [Float] {
        var rng = SeededRNG(seed: seed)
        return (0..<dimensions).map { _ in
            Float.random(in: -1...1, using: &rng)
        }
    }
}
