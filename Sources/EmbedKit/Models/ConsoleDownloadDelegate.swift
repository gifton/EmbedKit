import Foundation
import OSLog

/// Console-based download delegate for model downloads
/// 
/// Swift 6 compliant implementation using actor isolation
/// instead of @unchecked Sendable
public actor ConsoleDownloadDelegate: ModelDownloadDelegate {
    private let logger = Logger(subsystem: "EmbedKit", category: "Download")
    private var progressFormatter: ByteCountFormatter
    
    public init() {
        self.progressFormatter = ByteCountFormatter()
        self.progressFormatter.allowedUnits = [.useAll]
        self.progressFormatter.countStyle = .binary
    }
    
    public func downloadDidStart(url: URL) {
        logger.info("Starting download: \(url.lastPathComponent)")
    }
    
    public func downloadDidProgress(
        bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpected: Int64
    ) {
        let progress = Double(totalBytesWritten) / Double(totalBytesExpected) * 100
        let written = progressFormatter.string(fromByteCount: totalBytesWritten)
        let expected = progressFormatter.string(fromByteCount: totalBytesExpected)
        
        logger.info("""
            Download progress: \(String(format: "%.1f", progress))% \
            (\(written) / \(expected))
            """)
    }
    
    public func downloadDidComplete(url: URL, localURL: URL) {
        logger.info("Download complete: \(localURL.lastPathComponent)")
    }
    
    public func downloadDidFail(url: URL, error: Error) {
        logger.error("Download failed: \(error.localizedDescription)")
    }
}

/// Protocol for model download delegates
/// 
/// All methods are async to ensure Swift 6 compliance
public protocol ModelDownloadDelegate: Actor {
    /// Called when download starts
    func downloadDidStart(url: URL) async
    
    /// Called periodically with download progress
    func downloadDidProgress(
        bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpected: Int64
    ) async
    
    /// Called when download completes successfully
    func downloadDidComplete(url: URL, localURL: URL) async
    
    /// Called when download fails
    func downloadDidFail(url: URL, error: Error) async
}

/// Silent download delegate that doesn't log anything
public actor SilentDownloadDelegate: ModelDownloadDelegate {
    public init() {}
    
    public func downloadDidStart(url: URL) {
        // Silent - no logging
    }
    
    public func downloadDidProgress(
        bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpected: Int64
    ) {
        // Silent - no logging
    }
    
    public func downloadDidComplete(url: URL, localURL: URL) {
        // Silent - no logging
    }
    
    public func downloadDidFail(url: URL, error: Error) {
        // Silent - no logging
    }
}

/// Detailed download delegate with verbose logging
public actor DetailedDownloadDelegate: ModelDownloadDelegate {
    private let logger = Logger(subsystem: "EmbedKit", category: "DetailedDownload")
    private let progressFormatter: ByteCountFormatter
    private let dateFormatter: DateFormatter
    private var startTime: Date?
    private var lastProgressTime: Date?
    
    public init() {
        self.progressFormatter = ByteCountFormatter()
        self.progressFormatter.allowedUnits = [.useAll]
        self.progressFormatter.countStyle = .binary
        
        self.dateFormatter = DateFormatter()
        self.dateFormatter.dateStyle = .none
        self.dateFormatter.timeStyle = .medium
    }
    
    public func downloadDidStart(url: URL) {
        startTime = Date()
        logger.info("""
            📥 Download started
            URL: \(url.absoluteString)
            File: \(url.lastPathComponent)
            Time: \(self.dateFormatter.string(from: Date()))
            """)
    }
    
    public func downloadDidProgress(
        bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpected: Int64
    ) {
        let now = Date()
        
        // Only log progress every second to avoid spam
        if let lastTime = lastProgressTime,
           now.timeIntervalSince(lastTime) < 1.0 {
            return
        }
        
        lastProgressTime = now
        
        let progress = Double(totalBytesWritten) / Double(totalBytesExpected) * 100
        let written = progressFormatter.string(fromByteCount: totalBytesWritten)
        let expected = progressFormatter.string(fromByteCount: totalBytesExpected)
        
        // Calculate download speed
        var speed = "calculating..."
        if let start = startTime {
            let elapsed = now.timeIntervalSince(start)
            if elapsed > 0 {
                let bytesPerSecond = Double(totalBytesWritten) / elapsed
                speed = progressFormatter.string(fromByteCount: Int64(bytesPerSecond)) + "/s"
            }
        }
        
        // Estimate time remaining
        var eta = "calculating..."
        if totalBytesWritten > 0 && startTime != nil {
            let elapsed = now.timeIntervalSince(startTime!)
            let totalTime = elapsed * Double(totalBytesExpected) / Double(totalBytesWritten)
            let remaining = totalTime - elapsed
            if remaining > 0 {
                eta = formatTimeInterval(remaining)
            }
        }
        
        logger.info("""
            📊 Progress: \(String(format: "%.1f", progress))%
            Downloaded: \(written) / \(expected)
            Speed: \(speed)
            ETA: \(eta)
            """)
    }
    
    public func downloadDidComplete(url: URL, localURL: URL) {
        let duration: String
        if let start = startTime {
            duration = formatTimeInterval(Date().timeIntervalSince(start))
        } else {
            duration = "unknown"
        }
        
        logger.info("""
            ✅ Download completed
            File: \(localURL.lastPathComponent)
            Location: \(localURL.path)
            Duration: \(duration)
            """)
    }
    
    public func downloadDidFail(url: URL, error: Error) {
        logger.error("""
            ❌ Download failed
            URL: \(url.absoluteString)
            Error: \(error.localizedDescription)
            """)
    }
    
    private func formatTimeInterval(_ interval: TimeInterval) -> String {
        let hours = Int(interval) / 3600
        let minutes = (Int(interval) % 3600) / 60
        let seconds = Int(interval) % 60
        
        if hours > 0 {
            return String(format: "%dh %dm %ds", hours, minutes, seconds)
        } else if minutes > 0 {
            return String(format: "%dm %ds", minutes, seconds)
        } else {
            return String(format: "%ds", seconds)
        }
    }
}

/// Download delegate that reports to a callback
public actor CallbackDownloadDelegate: ModelDownloadDelegate {
    public typealias ProgressCallback = @Sendable (DownloadProgress) async -> Void
    
    private let callback: ProgressCallback
    
    public init(callback: @escaping ProgressCallback) {
        self.callback = callback
    }
    
    public func downloadDidStart(url: URL) {
        await callback(.started(url: url))
    }
    
    public func downloadDidProgress(
        bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpected: Int64
    ) {
        await callback(.progress(
            bytesWritten: bytesWritten,
            totalBytesWritten: totalBytesWritten,
            totalBytesExpected: totalBytesExpected
        ))
    }
    
    public func downloadDidComplete(url: URL, localURL: URL) {
        await callback(.completed(url: url, localURL: localURL))
    }
    
    public func downloadDidFail(url: URL, error: Error) {
        await callback(.failed(url: url, error: error))
    }
}

/// Download progress events
public enum DownloadProgress: Sendable {
    case started(url: URL)
    case progress(bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpected: Int64)
    case completed(url: URL, localURL: URL)
    case failed(url: URL, error: Error)
}