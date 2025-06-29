import Foundation
import OSLog

/// Factory for creating appropriate tokenizers based on model type
public enum TokenizerFactory {
    fileprivate static let logger = Logger(subsystem: "EmbedKit", category: "TokenizerFactory")
    
    /// Tokenizer types supported by the factory
    public enum TokenizerType: String, CaseIterable, Sendable {
        case simple = "simple"
        case bert = "bert"
        case wordpiece = "wordpiece"
        case sentencepiece = "sentencepiece"
        case bpe = "bpe"
        
        /// Get the appropriate tokenizer type for a model identifier
        public static func from(modelIdentifier: ModelIdentifier) -> TokenizerType {
            let modelName = modelIdentifier.rawValue.lowercased()
            
            if modelName.contains("bert") || modelName.contains("minilm") {
                return .bert
            } else if modelName.contains("sentence") || modelName.contains("sbert") {
                return .bert // Many sentence transformers use BERT tokenization
            } else if modelName.contains("gpt") || modelName.contains("roberta") {
                return .bpe
            } else if modelName.contains("t5") || modelName.contains("xlm") {
                return .sentencepiece
            } else {
                return .simple // Fallback
            }
        }
    }
    
    /// Create a tokenizer for the specified type
    public static func createTokenizer(
        type: TokenizerType,
        configuration: TokenizerConfiguration,
        vocabularyPath: String? = nil
    ) async throws -> any Tokenizer {
        logger.info("Creating tokenizer of type: \(type.rawValue)")
        
        switch type {
        case .simple:
            return SimpleTokenizer(
                maxSequenceLength: configuration.maxSequenceLength,
                vocabularySize: configuration.vocabularySize
            )
            
        case .bert, .wordpiece:
            return try await BERTTokenizer(
                vocabularyPath: vocabularyPath,
                maxSequenceLength: configuration.maxSequenceLength,
                doLowerCase: configuration.doLowerCase
            )
            
        case .bpe:
            // For now, use AdvancedTokenizer with BPE mode
            return try await AdvancedTokenizer(
                type: .bpe,
                vocabularyPath: vocabularyPath,
                maxSequenceLength: configuration.maxSequenceLength
            )
            
        case .sentencepiece:
            // For now, use AdvancedTokenizer with SentencePiece mode
            // In production, this should use a proper SentencePiece implementation
            logger.warning("SentencePiece tokenizer not fully implemented, using BPE fallback")
            return try await AdvancedTokenizer(
                type: .sentencepiece,
                vocabularyPath: vocabularyPath,
                maxSequenceLength: configuration.maxSequenceLength
            )
        }
    }
    
    /// Create a tokenizer automatically based on model identifier
    public static func createTokenizer(
        for modelIdentifier: ModelIdentifier,
        configuration: TokenizerConfiguration? = nil,
        vocabularyPath: String? = nil
    ) async throws -> any Tokenizer {
        // Use provided configuration or create model-specific one
        let config = configuration ?? ModelSpecificTokenizerConfig.forModel(modelIdentifier)
        let type = TokenizerType.from(modelIdentifier: modelIdentifier)
        return try await createTokenizer(
            type: type,
            configuration: config,
            vocabularyPath: vocabularyPath
        )
    }
    
    /// Load tokenizer from a model directory
    /// Note: If configuration is not provided, it must be loaded from the model directory
    public static func loadTokenizer(
        from modelDirectory: URL,
        type: TokenizerType? = nil,
        configuration: TokenizerConfiguration? = nil
    ) async throws -> any Tokenizer {
        // Look for vocabulary files in the directory
        let vocabTxtPath = modelDirectory.appendingPathComponent("vocab.txt").path
        let vocabJsonPath = modelDirectory.appendingPathComponent("vocab.json").path
        let tokenizerConfigPath = modelDirectory.appendingPathComponent("tokenizer_config.json").path
        
        // Determine vocabulary path
        let vocabularyPath: String?
        if FileManager.default.fileExists(atPath: vocabTxtPath) {
            vocabularyPath = vocabTxtPath
        } else if FileManager.default.fileExists(atPath: vocabJsonPath) {
            vocabularyPath = vocabJsonPath
        } else {
            vocabularyPath = nil
        }
        
        // Load tokenizer config if available
        let config: TokenizerConfiguration
        if let providedConfig = configuration {
            config = providedConfig
        } else if FileManager.default.fileExists(atPath: tokenizerConfigPath) {
            config = try await loadTokenizerConfig(from: URL(fileURLWithPath: tokenizerConfigPath))
        } else {
            // Try to infer from vocabulary size if available
            let vocabSize = try await inferVocabularySize(from: vocabularyPath ?? vocabTxtPath)
            config = TokenizerConfiguration(
                maxSequenceLength: 512, // Common default, should be overridden
                vocabularySize: vocabSize ?? 30522 // BERT default as last resort
            )
            logger.warning("No tokenizer configuration found. Using defaults which may not be optimal.")
        }
        
        // Determine tokenizer type
        let tokenizerType: TokenizerType
        if let type = type {
            tokenizerType = type
        } else if let detectedType = try await detectTokenizerType(from: modelDirectory) {
            tokenizerType = detectedType
        } else {
            tokenizerType = .simple
        }
        
        return try await createTokenizer(
            type: tokenizerType,
            configuration: config,
            vocabularyPath: vocabularyPath
        )
    }
    
    // MARK: - Private Helpers
    
    /// Load tokenizer configuration from file
    private static func loadTokenizerConfig(from url: URL) async throws -> TokenizerConfiguration {
        let data = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        
        let doLowerCase = json["do_lower_case"] as? Bool ?? true
        let stripAccents = json["strip_accents"] as? Bool ?? true
        let maxLength = json["max_length"] as? Int ?? 512
        
        let vocabSize = json["vocab_size"] as? Int ?? 30522
        
        return TokenizerConfiguration(
            doLowerCase: doLowerCase,
            stripAccents: stripAccents,
            maxSequenceLength: maxLength,
            vocabularySize: vocabSize
        )
    }
    
    /// Detect tokenizer type from model files
    private static func detectTokenizerType(from directory: URL) async throws -> TokenizerType? {
        let fileManager = FileManager.default
        
        // Check for specific tokenizer files
        if fileManager.fileExists(atPath: directory.appendingPathComponent("tokenizer.model").path) {
            return .sentencepiece
        }
        
        if fileManager.fileExists(atPath: directory.appendingPathComponent("merges.txt").path) {
            return .bpe
        }
        
        if fileManager.fileExists(atPath: directory.appendingPathComponent("vocab.txt").path) {
            // Could be BERT or WordPiece
            // Try to read first few lines to detect
            let vocabPath = directory.appendingPathComponent("vocab.txt")
            let content = try String(contentsOf: vocabPath, encoding: .utf8)
            let lines = content.components(separatedBy: .newlines).prefix(10)
            
            if lines.contains(where: { $0 == "[PAD]" || $0 == "[CLS]" || $0 == "[SEP]" }) {
                return .bert
            }
        }
        
        return nil
    }
    
    /// Infer vocabulary size from vocabulary file
    private static func inferVocabularySize(from path: String) async throws -> Int? {
        guard FileManager.default.fileExists(atPath: path) else { return nil }
        
        let url = URL(fileURLWithPath: path)
        let content = try String(contentsOf: url, encoding: .utf8)
        
        if path.hasSuffix(".json") {
            // JSON vocabulary
            if let data = content.data(using: .utf8),
               let json = try JSONSerialization.jsonObject(with: data) as? [String: Int] {
                return json.count
            }
        } else {
            // Text vocabulary (one token per line)
            let lines = content.components(separatedBy: .newlines)
                .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            return lines.count
        }
        
        return nil
    }
}

/// Model-specific tokenizer configurations
public enum ModelSpecificTokenizerConfig {
    /// Create BERT configuration with explicit dimensions
    public static func bert(maxSequenceLength: Int = 512, vocabularySize: Int = 30522) -> TokenizerConfiguration {
        TokenizerConfiguration(
            doLowerCase: true,
            stripAccents: true,
            addSpecialTokens: true,
            paddingStrategy: .maxLength,
            truncationStrategy: .longestFirst,
            maxSequenceLength: maxSequenceLength,
            vocabularySize: vocabularySize
        )
    }
    
    /// Create multilingual configuration with explicit dimensions
    public static func multilingual(maxSequenceLength: Int = 512, vocabularySize: Int = 250002) -> TokenizerConfiguration {
        TokenizerConfiguration(
            doLowerCase: false,
            stripAccents: false,
            addSpecialTokens: true,
            paddingStrategy: .maxLength,
            truncationStrategy: .longestFirst,
            maxSequenceLength: maxSequenceLength,
            vocabularySize: vocabularySize
        )
    }
    
    /// Create sentence transformer configuration with explicit dimensions
    public static func sentenceTransformer(maxSequenceLength: Int = 256, vocabularySize: Int = 30522) -> TokenizerConfiguration {
        TokenizerConfiguration(
            doLowerCase: true,
            stripAccents: true,
            addSpecialTokens: true,
            paddingStrategy: .maxLength,
            truncationStrategy: .longestFirst,
            maxSequenceLength: maxSequenceLength,
            vocabularySize: vocabularySize
        )
    }
    
    /// Get configuration for a specific model identifier
    public static func forModel(_ identifier: ModelIdentifier) -> TokenizerConfiguration {
        switch identifier {
        case .miniLM_L6_v2:
            return bert(maxSequenceLength: 512, vocabularySize: 30522)
        default:
            // For unknown models, use conservative defaults
            TokenizerFactory.logger.warning("Unknown model '\(identifier.rawValue)'. Using default configuration (maxLength: 512, vocab: 30522). Consider providing explicit TokenizerConfiguration.")
            return TokenizerConfiguration(
                doLowerCase: true,
                maxSequenceLength: 512,
                vocabularySize: 30522
            )
        }
    }
}

