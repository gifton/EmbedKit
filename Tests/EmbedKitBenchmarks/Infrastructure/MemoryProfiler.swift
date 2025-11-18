import Foundation
import Darwin

/// Utilities for measuring memory usage during benchmarks
public enum MemoryProfiler {

    /// Get current memory usage in bytes (resident memory)
    public static func currentUsage() -> Int64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    $0,
                    &count
                )
            }
        }

        guard kerr == KERN_SUCCESS else {
            return 0
        }

        return Int64(info.resident_size)
    }

    /// Measure memory usage of a synchronous operation
    /// - Parameter operation: The operation to measure
    /// - Returns: Tuple of (result, bytes allocated)
    public static func measure<T>(operation: () throws -> T) rethrows -> (result: T, bytes: Int64) {
        let before = currentUsage()
        let result = try operation()
        let after = currentUsage()
        return (result, after - before)
    }

    /// Measure memory usage of an async operation
    /// - Parameter operation: The async operation to measure
    /// - Returns: Tuple of (result, bytes allocated)
    public static func measureAsync<T>(operation: () async throws -> T) async rethrows -> (result: T, bytes: Int64) {
        let before = currentUsage()
        let result = try await operation()
        let after = currentUsage()
        return (result, after - before)
    }

    /// Format bytes as human-readable string
    public static func formatBytes(_ bytes: Int64) -> String {
        let kb = Double(bytes) / 1024.0
        let mb = kb / 1024.0
        let gb = mb / 1024.0

        if gb >= 1.0 {
            return String(format: "%.2f GB", gb)
        } else if mb >= 1.0 {
            return String(format: "%.2f MB", mb)
        } else if kb >= 1.0 {
            return String(format: "%.2f KB", kb)
        } else {
            return "\(bytes) bytes"
        }
    }
}
