import Foundation

// MARK: - Configuration Validation

extension Configuration {
    /// Validates the configuration and returns any issues
    public func validate() -> [ConfigurationIssue] {
        var issues: [ConfigurationIssue] = []
        
        // Validate model configuration
        if model.maxSequenceLength < 1 {
            issues.append(.invalid("maxSequenceLength must be positive"))
        }
        
        // Validate resource configuration
        if resources.batchSize < 1 {
            issues.append(.invalid("batchSize must be positive"))
        }
        
        if resources.maxConcurrentOperations < 1 {
            issues.append(.invalid("maxConcurrentOperations must be positive"))
        }
        
        // Validate cache configuration
        if cache.maxCacheSize < 0 {
            issues.append(.invalid("maxCacheSize cannot be negative"))
        }
        
        if cache.ttl < 0 {
            issues.append(.invalid("cache TTL cannot be negative"))
        }
        
        // Check for incompatible settings
        if performance.useMetalAcceleration && model.computeUnits == .cpuOnly {
            issues.append(.incompatible("Metal acceleration requested but compute units set to CPU only"))
        }
        
        return issues
    }
}

/// Configuration validation issues
public enum ConfigurationIssue: Equatable {
    case invalid(String)
    case missing(String)
    case incompatible(String)
    case outOfRange(String)
    
    public var description: String {
        switch self {
        case .invalid(let message):
            return "Invalid configuration: \(message)"
        case .missing(let message):
            return "Missing configuration: \(message)"
        case .incompatible(let message):
            return "Incompatible configuration: \(message)"
        case .outOfRange(let message):
            return "Configuration out of range: \(message)"
        }
    }
}