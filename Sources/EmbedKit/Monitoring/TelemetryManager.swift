import Foundation
import OSLog

/// Global telemetry manager for EmbedKit
/// 
/// Swift 6 compliant implementation using proper Sendable design
/// instead of @unchecked Sendable
public final class TelemetryManager: Sendable {
    /// Shared instance for global access
    public static let shared = TelemetryManager()
    
    /// The underlying telemetry system
    public let system: TelemetrySystem
    
    /// Private initializer to ensure singleton pattern
    private init() {
        self.system = TelemetrySystem()
    }
    
    /// Convenience access to the telemetry system
    public static var telemetry: TelemetrySystem {
        shared.system
    }
}

/// Global telemetry instance for backward compatibility and convenience
public let telemetry = TelemetryManager.shared.system