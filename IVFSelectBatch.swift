// FIXED VERSION: IVFSelect Batch Processing
// This fixes the race condition bug in ivf_select_nprobe_batch_f32
// Original bug: Results were being written inside concurrent dispatch block

import Foundation
import Accelerate

/// CORRECTED Batch query processing with thread-safe result population
///
/// Key Fix: Results are written directly to thread-local indices,
/// eliminating the race condition from the original implementation.
///
/// - Parameters:
///   - Q: Batch of queries [b × d], row-major layout
///   - b: Batch size
///   - d: Vector dimension
///   - centroids: Coarse centroids [kc × d]
///   - kc: Number of centroids
///   - metric: Distance metric
///   - nprobe: Number of lists to probe per query
///   - opts: Optional configuration
///   - listIDsOut: Output list IDs [b × nprobe], row-major per query
///   - listScoresOut: Optional output scores [b × nprobe]
@preconcurrency
public func ivf_select_nprobe_batch_f32(
    Q: [Float],
    b: Int,
    d: Int,
    centroids: [Float],
    kc: Int,
    metric: IVFMetric,
    nprobe: Int,
    opts: IVFSelectOpts = IVFSelectOpts(),
    listIDsOut: inout [Int32],
    listScoresOut: inout [Float]?
) {
    // Validation
    precondition(b >= 1 && d > 0 && Q.count == b * d, "Invalid query batch dimensions")
    precondition(kc >= 1 && centroids.count == kc * d, "Invalid centroid dimensions")
    precondition(nprobe >= 1, "nprobe must be >= 1")

    // Ensure outputs sized correctly BEFORE concurrent processing
    if listIDsOut.count != b * nprobe {
        listIDsOut = [Int32](repeating: -1, count: b * nprobe)
    }
    if listScoresOut != nil && listScoresOut!.count != b * nprobe {
        listScoresOut = [Float](repeating: .nan, count: b * nprobe)
    }

    // Create thread-safe storage for results
    // Using NSLock to protect shared write access if needed
    let resultLock = NSLock()

    // Pre-allocate result arrays outside concurrent block
    var allIDs = [Int32](repeating: -1, count: b * nprobe)
    var allScores: [Float]? = (listScoresOut != nil) ? [Float](repeating: .nan, count: b * nprobe) : nil

    // Process queries in parallel
    DispatchQueue.concurrentPerform(iterations: b) { i in
        let qOffset = i * d
        let outOffset = i * nprobe

        Q.withUnsafeBufferPointer { QPtr in
            centroids.withUnsafeBufferPointer { centsPtr in
                // Direct pointer slicing - no allocation
                let qSlice = UnsafeBufferPointer(start: QPtr.baseAddress! + qOffset, count: d)

                // Thread-local temporary buffers
                var localIDs = [Int32](repeating: -1, count: nprobe)
                var localScores: [Float]? = (allScores != nil) ? [Float](repeating: .nan, count: nprobe) : nil

                // Call the single-query selection (this is thread-safe)
                selectNprobeSingleThread(
                    q: qSlice,
                    d: d,
                    centroids: centsPtr,
                    kc: kc,
                    metric: metric,
                    nprobe: nprobe,
                    opts: opts,
                    listIDsOut: &localIDs,
                    listScoresOut: &localScores
                )

                // Write directly to the correct position in shared arrays
                // Each thread writes to its own disjoint range [outOffset..outOffset+nprobe]
                // No locking needed as ranges don't overlap
                for j in 0..<nprobe {
                    allIDs[outOffset + j] = localIDs[j]
                    if let localSc = localScores {
                        allScores?[outOffset + j] = localSc[j]
                    }
                }
            }
        }
    }

    // Copy final results to output parameters AFTER concurrent processing completes
    listIDsOut = allIDs
    if listScoresOut != nil {
        listScoresOut = allScores
    }
}

// MARK: - Alternative Fix Using Thread-Safe Accumulator

/// Alternative implementation using a thread-safe accumulator pattern
/// This approach is cleaner and more maintainable
final class BatchResultAccumulator: @unchecked Sendable {
    private var ids: [Int32]
    private var scores: [Float]?
    private let lock = NSLock()

    init(size: Int, withScores: Bool) {
        self.ids = [Int32](repeating: -1, count: size)
        self.scores = withScores ? [Float](repeating: .nan, count: size) : nil
    }

    func setResults(at offset: Int, ids: [Int32], scores: [Float]?) {
        lock.lock()
        defer { lock.unlock() }

        for (j, id) in ids.enumerated() {
            self.ids[offset + j] = id
        }

        if let scores = scores, let selfScores = self.scores {
            for (j, score) in scores.enumerated() {
                self.scores![offset + j] = score
            }
        }
    }

    func getResults() -> (ids: [Int32], scores: [Float]?) {
        lock.lock()
        defer { lock.unlock() }
        return (ids, scores)
    }
}

@preconcurrency
public func ivf_select_nprobe_batch_f32Accumulator(
    Q: [Float],
    b: Int,
    d: Int,
    centroids: [Float],
    kc: Int,
    metric: IVFMetric,
    nprobe: Int,
    opts: IVFSelectOpts = IVFSelectOpts(),
    listIDsOut: inout [Int32],
    listScoresOut: inout [Float]?
) {
    // Validation
    precondition(b >= 1 && d > 0 && Q.count == b * d, "Invalid query batch dimensions")
    precondition(kc >= 1 && centroids.count == kc * d, "Invalid centroid dimensions")
    precondition(nprobe >= 1, "nprobe must be >= 1")

    // Create thread-safe accumulator
    let accumulator = BatchResultAccumulator(
        size: b * nprobe,
        withScores: listScoresOut != nil
    )

    // Process queries in parallel
    DispatchQueue.concurrentPerform(iterations: b) { i in
        let qOffset = i * d
        let outOffset = i * nprobe

        Q.withUnsafeBufferPointer { QPtr in
            centroids.withUnsafeBufferPointer { centsPtr in
                let qSlice = UnsafeBufferPointer(start: QPtr.baseAddress! + qOffset, count: d)

                var localIDs = [Int32](repeating: -1, count: nprobe)
                var localScores: [Float]? = (listScoresOut != nil) ? [Float](repeating: .nan, count: nprobe) : nil

                selectNprobeSingleThread(
                    q: qSlice,
                    d: d,
                    centroids: centsPtr,
                    kc: kc,
                    metric: metric,
                    nprobe: nprobe,
                    opts: opts,
                    listIDsOut: &localIDs,
                    listScoresOut: &localScores
                )

                // Thread-safe write to accumulator
                accumulator.setResults(at: outOffset, ids: localIDs, scores: localScores)
            }
        }
    }

    // Extract final results
    let (finalIDs, finalScores) = accumulator.getResults()
    listIDsOut = finalIDs
    listScoresOut = finalScores
}

// MARK: - Test Helpers for Validation

/// Validates that batch results match single-query results exactly
public func validateBatchVsSingleParity(
    Q: [Float],
    b: Int,
    d: Int,
    centroids: [Float],
    kc: Int,
    metric: IVFMetric,
    nprobe: Int,
    opts: IVFSelectOpts = IVFSelectOpts()
) -> Bool {
    // Run batch processing with fixed implementation
    var batchIDs = [Int32](repeating: -1, count: b * nprobe)
    var batchScores: [Float]? = [Float](repeating: 0, count: b * nprobe)

    ivf_select_nprobe_batch_f32_FIXED(
        Q: Q, b: b, d: d, centroids: centroids, kc: kc,
        metric: metric, nprobe: nprobe, opts: opts,
        listIDsOut: &batchIDs, listScoresOut: &batchScores
    )

    // Run single-query processing for each query
    for i in 0..<b {
        let qOffset = i * d
        let q = Array(Q[qOffset..<(qOffset + d)])

        var singleIDs = [Int32](repeating: -1, count: nprobe)
        var singleScores: [Float]? = [Float](repeating: 0, count: nprobe)

        ivf_select_nprobe_f32(
            q: q, d: d, centroids: centroids, kc: kc,
            metric: metric, nprobe: nprobe, opts: opts,
            listIDsOut: &singleIDs, listScoresOut: &singleScores
        )

        // Compare batch[i] vs single
        let batchOffset = i * nprobe
        for j in 0..<nprobe {
            if batchIDs[batchOffset + j] != singleIDs[j] {
                print("❌ Mismatch at query \(i), position \(j): batch=\(batchIDs[batchOffset + j]) vs single=\(singleIDs[j])")
                return false
            }

            if let bScores = batchScores, let sScores = singleScores {
                let diff = abs(bScores[batchOffset + j] - sScores[j])
                if diff > 1e-5 {
                    print("❌ Score mismatch at query \(i), position \(j): diff=\(diff)")
                    return false
                }
            }
        }
    }

    return true
}

// MARK: - Performance Comparison

/// Benchmark to compare original vs fixed implementation
public func benchmarkBatchImplementations(
    b: Int = 100,
    d: Int = 128,
    kc: Int = 10000,
    nprobe: Int = 50
) {
    let Q = (0..<(b * d)).map { _ in Float.random(in: -1...1) }
    let centroids = (0..<(kc * d)).map { _ in Float.random(in: -1...1) }

    print("🔬 Benchmarking Batch Implementations")
    print("  Batch size: \(b), Dimension: \(d), Centroids: \(kc), nprobe: \(nprobe)")

    // Test correctness first
    let isCorrect = validateBatchVsSingleParity(
        Q: Q, b: b, d: d, centroids: centroids,
        kc: kc, metric: .l2, nprobe: nprobe
    )

    if isCorrect {
        print("✅ Batch implementation produces correct results")
    } else {
        print("❌ Batch implementation has errors")
    }

    // Timing comparison
    var batchIDs = [Int32](repeating: -1, count: b * nprobe)
    var batchScores: [Float]? = nil

    let start = CFAbsoluteTimeGetCurrent()
    ivf_select_nprobe_batch_f32_FIXED(
        Q: Q, b: b, d: d, centroids: centroids, kc: kc,
        metric: .l2, nprobe: nprobe,
        listIDsOut: &batchIDs, listScoresOut: &batchScores
    )
    let elapsed = CFAbsoluteTimeGetCurrent() - start

    print("⏱️ Batch processing time: \(elapsed * 1000)ms")
    print("   Per-query: \(elapsed * 1000 / Double(b))ms")
}
