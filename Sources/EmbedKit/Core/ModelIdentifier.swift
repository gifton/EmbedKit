import Foundation

// MARK: - Model Identifier

/// A type-safe identifier for embedding models that provides validation and normalization.
///
/// This type replaces string-based model identifiers with a structured approach that
/// prevents typos and enables compile-time checks for model compatibility.
///
/// ## Examples
///
/// ```swift
/// // Create from a known model
/// let identifier = ModelIdentifier.miniLM_L6_v2
///
/// // Create from a string (validated)
/// let custom = try ModelIdentifier("custom-model-v1")
///
/// // Use in APIs
/// let embedder = try modelManager.loadModel(identifier)
/// ```
public struct ModelIdentifier: Hashable, Sendable {
    /// The normalized identifier string
    public let rawValue: String
    
    /// The model family (e.g., "MiniLM", "MPNet", "GTE")
    public let family: String
    
    /// The model variant (e.g., "L6", "base", "small")
    public let variant: String?
    
    /// The model version (e.g., "v2", "v1")
    public let version: String?
    
    /// The full display name
    public var displayName: String {
        rawValue
    }
    
    // MARK: - Initialization
    
    /// Creates a model identifier from a raw string with validation.
    ///
    /// The string is normalized and validated to ensure it follows the expected format.
    /// Valid formats include:
    /// - `family-variant-version` (e.g., "MiniLM-L6-v2")
    /// - `family-version` (e.g., "GTE-v1")
    /// - `family` (e.g., "BERT")
    ///
    /// - Parameter rawValue: The raw model identifier string
    /// - Throws: `ModelIdentifierError` if the identifier is invalid
    public init(_ rawValue: String) throws {
        let normalized = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        
        guard !normalized.isEmpty else {
            throw ModelIdentifierError.emptyIdentifier
        }
        
        guard normalized.count <= 128 else {
            throw ModelIdentifierError.identifierTooLong
        }
        
        // Validate characters (alphanumeric, hyphens, underscores, dots)
        let allowedCharacters = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_."))
        guard normalized.rangeOfCharacter(from: allowedCharacters.inverted) == nil else {
            throw ModelIdentifierError.invalidCharacters(normalized)
        }
        
        self.rawValue = normalized
        
        // Parse components
        let components = normalized.split(separator: "-").map(String.init)
        
        guard !components.isEmpty else {
            throw ModelIdentifierError.invalidFormat(normalized)
        }
        
        self.family = components[0]
        
        if components.count > 1 {
            // Check if last component is a version
            let lastComponent = components.last!
            if lastComponent.hasPrefix("v") && lastComponent.dropFirst().allSatisfy({ $0.isNumber || $0 == "." }) {
                self.version = lastComponent
                if components.count > 2 {
                    self.variant = components[1..<components.count-1].joined(separator: "-")
                } else {
                    self.variant = nil
                }
            } else {
                self.version = nil
                self.variant = components[1...].joined(separator: "-")
            }
        } else {
            self.variant = nil
            self.version = nil
        }
    }
    
    /// Creates a model identifier with explicit components.
    public init(family: String, variant: String? = nil, version: String? = nil) {
        self.family = family
        self.variant = variant
        self.version = version
        
        var components = [family]
        if let variant = variant {
            components.append(variant)
        }
        if let version = version {
            components.append(version)
        }
        
        self.rawValue = components.joined(separator: "-")
    }
    
    // MARK: - Well-Known Models
    
    /// All-MiniLM-L6-v2 model identifier
    public static let miniLM_L6_v2 = ModelIdentifier(
        family: "all-MiniLM",
        variant: "L6",
        version: "v2"
    )
    
    /// All-MPNet-base-v2 model identifier
    public static let mpnet_base_v2 = ModelIdentifier(
        family: "all-mpnet",
        variant: "base",
        version: "v2"
    )
    
    /// GTE-small model identifier
    public static let gte_small = ModelIdentifier(
        family: "gte",
        variant: "small"
    )
    
    /// BGE-small-en model identifier
    public static let bge_small_en = ModelIdentifier(
        family: "bge",
        variant: "small-en",
        version: "v1.5"
    )
    
    /// Default model identifier
    public static let `default` = miniLM_L6_v2
}

// MARK: - String Interoperability

extension ModelIdentifier: ExpressibleByStringLiteral {
    /// Creates a model identifier from a string literal.
    ///
    /// This initializer will trap if the string is invalid. Use only for known-good identifiers.
    public init(stringLiteral value: String) {
        do {
            try self.init(value)
        } catch {
            fatalError("Invalid model identifier literal: \(value). Error: \(error)")
        }
    }
}

extension ModelIdentifier: RawRepresentable {
    public init?(rawValue: String) {
        try? self.init(rawValue)
    }
}

extension ModelIdentifier: CustomStringConvertible {
    public var description: String {
        rawValue
    }
}

extension ModelIdentifier: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let rawValue = try container.decode(String.self)
        try self.init(rawValue)
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}

// MARK: - Errors

/// Errors that can occur when creating or validating model identifiers.
public enum ModelIdentifierError: LocalizedError, Sendable {
    case emptyIdentifier
    case identifierTooLong
    case invalidCharacters(String)
    case invalidFormat(String)
    
    public var errorDescription: String? {
        switch self {
        case .emptyIdentifier:
            return "Model identifier cannot be empty"
        case .identifierTooLong:
            return "Model identifier is too long (max 128 characters)"
        case .invalidCharacters(let identifier):
            return "Model identifier contains invalid characters: \(identifier)"
        case .invalidFormat(let identifier):
            return "Model identifier has invalid format: \(identifier)"
        }
    }
}

// MARK: - Model Registry

/// A registry of known models with their metadata.
public struct ModelRegistry: Sendable {
    /// Known models with their identifiers and metadata
    public static let knownModels: Set<ModelIdentifier> = [
        .miniLM_L6_v2,
        .mpnet_base_v2,
        .gte_small,
        .bge_small_en
    ]
    
    /// Checks if a model identifier represents a known model.
    public static func isKnownModel(_ identifier: ModelIdentifier) -> Bool {
        knownModels.contains(identifier)
    }
    
    /// Gets the embedding dimensions for a known model.
    public static func embeddingDimensions(for identifier: ModelIdentifier) -> Int? {
        switch identifier {
        case .miniLM_L6_v2:
            return 384
        case .mpnet_base_v2:
            return 768
        case .gte_small:
            return 384
        case .bge_small_en:
            return 384
        default:
            return nil
        }
    }
    
    /// Gets the maximum sequence length for a known model.
    public static func maxSequenceLength(for identifier: ModelIdentifier) -> Int? {
        switch identifier {
        case .miniLM_L6_v2:
            return 256
        case .mpnet_base_v2:
            return 384
        case .gte_small:
            return 512
        case .bge_small_en:
            return 512
        default:
            return nil
        }
    }
}

