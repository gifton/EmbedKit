// EmbedKit - Minimal Profiler Hook

import Foundation

/// Lightweight, opt‑in profiler interface for external benchmarking tools.
/// Implementations can aggregate/export metrics; EmbedKit only invokes callbacks.
public protocol Profiler: Sendable {
    /// Record per-stage timings for a set of items.
    func recordStage(
        model: ModelID,
        items: Int,
        tokenization: TimeInterval,
        inference: TimeInterval,
        pooling: TimeInterval,
        context: [String: String]
    )

    /// Record a micro‑batch event with padded length and batch size.
    func recordMicroBatch(
        model: ModelID,
        batchSize: Int,
        paddedLength: Int,
        device: ComputeDevice,
        context: [String: String]
    )
}

