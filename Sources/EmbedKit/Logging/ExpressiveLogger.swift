import Foundation
import OSLog

/// Expressive logging system with emojis and rich formatting
public struct ExpressiveLogger: Sendable {
    private let logger: Logger
    private let category: String
    
    /// Log level emojis and prefixes
    private enum LogEmoji {
        static let debug = "🔍"
        static let info = "ℹ️"
        static let success = "✅"
        static let warning = "⚠️"
        static let error = "❌"
        static let critical = "🚨"
        static let performance = "⚡"
        static let memory = "💾"
        static let network = "🌐"
        static let model = "🤖"
        static let cache = "📦"
        static let security = "🔒"
        static let start = "🚀"
        static let complete = "🎉"
        static let processing = "⚙️"
        static let thinking = "🤔"
    }
    
    public init(subsystem: String = "EmbedKit", category: String) {
        self.logger = Logger(subsystem: subsystem, category: category)
        self.category = category
    }
    
    // MARK: - Core Logging Methods
    
    /// Log debug information with context
    public func debug(_ message: String, context: String? = nil) {
        let formattedMessage = formatMessage(LogEmoji.debug, message, context: context)
        logger.debug("\(formattedMessage)")
    }
    
    /// Log general information
    public func info(_ message: String, context: String? = nil) {
        let formattedMessage = formatMessage(LogEmoji.info, message, context: context)
        logger.info("\(formattedMessage)")
    }
    
    /// Log successful operations
    public func success(_ message: String, context: String? = nil) {
        let formattedMessage = formatMessage(LogEmoji.success, message, context: context)
        logger.info("\(formattedMessage)")
    }
    
    /// Log warnings
    public func warning(_ message: String, context: String? = nil) {
        let formattedMessage = formatMessage(LogEmoji.warning, message, context: context)
        logger.warning("\(formattedMessage)")
    }
    
    /// Log errors
    public func error(_ message: String, error: Error? = nil, context: String? = nil) {
        var fullMessage = message
        if let error = error {
            fullMessage += " → \(error.localizedDescription)"
        }
        let formattedMessage = formatMessage(LogEmoji.error, fullMessage, context: context)
        logger.error("\(formattedMessage)")
    }
    
    /// Log critical issues
    public func critical(_ message: String, error: Error? = nil, context: String? = nil) {
        var fullMessage = message
        if let error = error {
            fullMessage += " → \(error.localizedDescription)"
        }
        let formattedMessage = formatMessage(LogEmoji.critical, fullMessage, context: context)
        logger.critical("\(formattedMessage)")
    }
    
    // MARK: - Specialized Logging Methods
    
    /// Log performance metrics
    public func performance(_ operation: String, duration: TimeInterval, throughput: Double? = nil) {
        var message = "\(operation) completed in \(formatDuration(duration))"
        if let throughput = throughput {
            message += " • \(formatThroughput(throughput))"
        }
        let formattedMessage = formatMessage(LogEmoji.performance, message)
        logger.info("\(formattedMessage)")
    }
    
    /// Log memory usage
    public func memory(_ operation: String, bytes: Int64, peak: Int64? = nil) {
        var message = "\(operation) • Memory: \(formatBytes(bytes))"
        if let peak = peak {
            message += " (Peak: \(formatBytes(peak)))"
        }
        let formattedMessage = formatMessage(LogEmoji.memory, message)
        logger.info("\(formattedMessage)")
    }
    
    /// Log model operations
    public func model(_ operation: String, modelId: String? = nil, version: String? = nil) {
        var message = operation
        if let modelId = modelId {
            message += " • Model: \(modelId)"
        }
        if let version = version {
            message += " v\(version)"
        }
        let formattedMessage = formatMessage(LogEmoji.model, message)
        logger.info("\(formattedMessage)")
    }
    
    /// Log cache operations
    public func cache(_ operation: String, hitRate: Double? = nil, size: Int? = nil) {
        var message = operation
        if let hitRate = hitRate {
            message += " • Hit Rate: \(formatPercentage(hitRate))"
        }
        if let size = size {
            message += " • Size: \(size) items"
        }
        let formattedMessage = formatMessage(LogEmoji.cache, message)
        logger.info("\(formattedMessage)")
    }
    
    /// Log security operations
    public func security(_ operation: String, status: String? = nil) {
        var message = operation
        if let status = status {
            message += " • Status: \(status)"
        }
        let formattedMessage = formatMessage(LogEmoji.security, message)
        logger.info("\(formattedMessage)")
    }
    
    /// Log operation start
    public func start(_ operation: String, details: String? = nil) {
        var message = "Starting \(operation)"
        if let details = details {
            message += " • \(details)"
        }
        let formattedMessage = formatMessage(LogEmoji.start, message)
        logger.info("\(formattedMessage)")
    }
    
    /// Log operation completion
    public func complete(_ operation: String, result: String? = nil) {
        var message = "Completed \(operation)"
        if let result = result {
            message += " • \(result)"
        }
        let formattedMessage = formatMessage(LogEmoji.complete, message)
        logger.info("\(formattedMessage)")
    }
    
    /// Log processing status
    public func processing(_ item: String, progress: Double? = nil) {
        var message = "Processing \(item)"
        if let progress = progress {
            message += " • \(formatProgress(progress))"
        }
        let formattedMessage = formatMessage(LogEmoji.processing, message)
        logger.info("\(formattedMessage)")
    }
    
    /// Log thinking/analysis operations
    public func thinking(_ operation: String) {
        let formattedMessage = formatMessage(LogEmoji.thinking, "Analyzing \(operation)")
        logger.info("\(formattedMessage)")
    }
    
    // MARK: - Formatting Helpers
    
    private func formatMessage(_ emoji: String, _ message: String, context: String? = nil) -> String {
        var formatted = "\(emoji) [\(category)]"
        if let context = context {
            formatted += " {\(context)}"
        }
        formatted += " \(message)"
        return formatted
    }
    
    private func formatDuration(_ duration: TimeInterval) -> String {
        if duration < 0.001 {
            return String(format: "%.0fμs", duration * 1_000_000)
        } else if duration < 1.0 {
            return String(format: "%.1fms", duration * 1000)
        } else if duration < 60.0 {
            return String(format: "%.2fs", duration)
        } else {
            let minutes = Int(duration / 60)
            let seconds = duration.truncatingRemainder(dividingBy: 60)
            return String(format: "%dm %.1fs", minutes, seconds)
        }
    }
    
    private func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        formatter.includesUnit = true
        formatter.includesCount = true
        return formatter.string(fromByteCount: bytes)
    }
    
    private func formatThroughput(_ throughput: Double) -> String {
        if throughput < 1.0 {
            return String(format: "%.2f ops/sec", throughput)
        } else if throughput < 1000.0 {
            return String(format: "%.1f ops/sec", throughput)
        } else if throughput < 1_000_000.0 {
            return String(format: "%.1fK ops/sec", throughput / 1000.0)
        } else {
            return String(format: "%.1fM ops/sec", throughput / 1_000_000.0)
        }
    }
    
    private func formatPercentage(_ value: Double) -> String {
        return String(format: "%.1f%%", value * 100.0)
    }
    
    private func formatProgress(_ progress: Double) -> String {
        let percentage = min(max(progress * 100, 0), 100)
        let filledBars = Int(progress * 10)
        let emptyBars = 10 - filledBars
        let progressBar = String(repeating: "█", count: filledBars) + String(repeating: "░", count: emptyBars)
        return String(format: "%@ %.0f%%", progressBar, percentage)
    }
}

// MARK: - Global Logger Factory

public struct EmbedKitLogger {
    /// Create a logger for embeddings
    public static func embeddings() -> ExpressiveLogger {
        ExpressiveLogger(category: "Embeddings")
    }
    
    /// Create a logger for Metal acceleration
    public static func metal() -> ExpressiveLogger {
        ExpressiveLogger(category: "Metal")
    }
    
    /// Create a logger for cache operations
    public static func cache() -> ExpressiveLogger {
        ExpressiveLogger(category: "Cache")
    }
    
    /// Create a logger for streaming
    public static func streaming() -> ExpressiveLogger {
        ExpressiveLogger(category: "Streaming")
    }
    
    /// Create a logger for model management
    public static func modelManagement() -> ExpressiveLogger {
        ExpressiveLogger(category: "ModelMgmt")
    }
    
    /// Create a logger for security
    public static func security() -> ExpressiveLogger {
        ExpressiveLogger(category: "Security")
    }
    
    /// Create a logger for telemetry
    public static func telemetry() -> ExpressiveLogger {
        ExpressiveLogger(category: "Telemetry")
    }
    
    /// Create a logger for benchmarks
    public static func benchmarks() -> ExpressiveLogger {
        ExpressiveLogger(category: "Benchmarks")
    }
    
    /// Create a custom logger
    public static func custom(_ category: String) -> ExpressiveLogger {
        ExpressiveLogger(category: category)
    }
}

// MARK: - Structured Logging Support

public struct LogContext: Sendable {
    let requestId: String?
    let userId: String?
    let modelId: String?
    let operation: String?
    let metadata: [String: String]
    
    public init(
        requestId: String? = nil,
        userId: String? = nil,
        modelId: String? = nil,
        operation: String? = nil,
        metadata: [String: String] = [:]
    ) {
        self.requestId = requestId
        self.userId = userId
        self.modelId = modelId
        self.operation = operation
        self.metadata = metadata
    }
}

// MARK: - Extensions for Rich Logging

extension ExpressiveLogger {
    /// Log with structured context
    public func log(
        level: OSLogType = .info,
        _ message: String,
        context: LogContext,
        error: Error? = nil
    ) {
        var contextString = ""
        
        if let requestId = context.requestId {
            contextString += "req:\(requestId)"
        }
        if let userId = context.userId {
            if !contextString.isEmpty { contextString += "|" }
            contextString += "user:\(userId)"
        }
        if let modelId = context.modelId {
            if !contextString.isEmpty { contextString += "|" }
            contextString += "model:\(modelId)"
        }
        
        switch level {
        case .debug:
            debug(message, context: contextString)
        case .info:
            info(message, context: contextString)
        case .error:
            self.error(message, error: error, context: contextString)
        case .fault:
            critical(message, error: error, context: contextString)
        default:
            info(message, context: contextString)
        }
    }
    
    /// Log operation timing
    public func timed<T>(_ operation: String, block: () async throws -> T) async rethrows -> T {
        let startTime = CFAbsoluteTimeGetCurrent()
        start(operation)
        
        do {
            let result = try await block()
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            performance(operation, duration: duration)
            return result
        } catch {
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            self.error("\(operation) failed after \(formatDuration(duration))", error: error)
            throw error
        }
    }
}