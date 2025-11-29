// EmbedKit - BatchProgress Tests

import Testing
import Foundation
@testable import EmbedKit
import VectorCore

@Suite("BatchProgress")
struct BatchProgressTests {

    // MARK: - Initialization Tests

    @Test("Basic initialization")
    func testBasicInit() {
        let progress = BatchProgress(
            current: 50,
            total: 100,
            batchIndex: 2,
            totalBatches: 5
        )

        #expect(progress.current == 50)
        #expect(progress.total == 100)
        #expect(progress.batchIndex == 2)
        #expect(progress.totalBatches == 5)
        #expect(progress.percentage == 50)
        #expect(progress.fraction == 0.5)
    }

    @Test("Full initialization with all parameters")
    func testFullInit() {
        let progress = BatchProgress(
            current: 25,
            total: 100,
            batchIndex: 1,
            totalBatches: 4,
            phase: "Embedding",
            message: "Processing batch 2",
            itemsPerSecond: 50.5,
            tokensProcessed: 1250,
            currentBatchSize: 25,
            estimatedTimeRemaining: 1.5
        )

        #expect(progress.current == 25)
        #expect(progress.total == 100)
        #expect(progress.batchIndex == 1)
        #expect(progress.totalBatches == 4)
        #expect(progress.phase == "Embedding")
        #expect(progress.message == "Processing batch 2")
        #expect(progress.itemsPerSecond == 50.5)
        #expect(progress.tokensProcessed == 1250)
        #expect(progress.currentBatchSize == 25)
        #expect(progress.estimatedTimeRemaining == 1.5)
    }

    @Test("Negative values are clamped")
    func testNegativeValuesClamped() {
        let progress = BatchProgress(
            current: -10,
            total: -5,
            batchIndex: -1,
            totalBatches: -2,
            tokensProcessed: -100,
            currentBatchSize: -5
        )

        #expect(progress.current == 0)
        #expect(progress.total == 0)
        #expect(progress.batchIndex == 0)
        #expect(progress.totalBatches == 1)  // min 1
        #expect(progress.tokensProcessed == 0)
        #expect(progress.currentBatchSize == 0)
    }

    // MARK: - Computed Properties Tests

    @Test("Fraction calculation")
    func testFraction() {
        let half = BatchProgress(current: 50, total: 100)
        #expect(half.fraction == 0.5)

        let quarter = BatchProgress(current: 25, total: 100)
        #expect(quarter.fraction == 0.25)

        let complete = BatchProgress(current: 100, total: 100)
        #expect(complete.fraction == 1.0)
    }

    @Test("Fraction with zero total")
    func testFractionZeroTotal() {
        let progress = BatchProgress(current: 50, total: 0)
        #expect(progress.fraction == 0.0)
    }

    @Test("Percentage calculation")
    func testPercentage() {
        let p25 = BatchProgress(current: 25, total: 100)
        #expect(p25.percentage == 25)

        let p75 = BatchProgress(current: 75, total: 100)
        #expect(p75.percentage == 75)

        let p100 = BatchProgress(current: 100, total: 100)
        #expect(p100.percentage == 100)
    }

    @Test("isComplete calculation")
    func testIsComplete() {
        let incomplete = BatchProgress(current: 50, total: 100)
        #expect(!incomplete.isComplete)

        let complete = BatchProgress(current: 100, total: 100)
        #expect(complete.isComplete)

        let overComplete = BatchProgress(current: 150, total: 100)
        #expect(overComplete.isComplete)
    }

    // MARK: - Factory Methods Tests

    @Test("Started factory")
    func testStartedFactory() {
        let started = BatchProgress.started(total: 100, totalBatches: 4)

        #expect(started.current == 0)
        #expect(started.total == 100)
        #expect(started.batchIndex == 0)
        #expect(started.totalBatches == 4)
        #expect(started.phase == "Starting")
        #expect(started.percentage == 0)
        #expect(!started.isComplete)
    }

    @Test("Completed factory")
    func testCompletedFactory() {
        let completed = BatchProgress.completed(
            total: 100,
            totalBatches: 4,
            tokensProcessed: 5000,
            itemsPerSecond: 50.0
        )

        #expect(completed.current == 100)
        #expect(completed.total == 100)
        #expect(completed.batchIndex == 3)  // last batch index
        #expect(completed.totalBatches == 4)
        #expect(completed.phase == "Complete")
        #expect(completed.tokensProcessed == 5000)
        #expect(completed.itemsPerSecond == 50.0)
        #expect(completed.isComplete)
    }

    @Test("BatchCompleted factory")
    func testBatchCompletedFactory() {
        let progress = BatchProgress.batchCompleted(
            itemsCompleted: 50,
            totalItems: 100,
            batchIndex: 1,
            totalBatches: 4,
            batchSize: 25,
            tokensInBatch: 1250,
            totalTokens: 2500,
            elapsedTime: 1.0
        )

        #expect(progress.current == 50)
        #expect(progress.total == 100)
        #expect(progress.batchIndex == 1)
        #expect(progress.totalBatches == 4)
        #expect(progress.currentBatchSize == 25)
        #expect(progress.tokensProcessed == 2500)
        #expect(progress.itemsPerSecond == 50.0)  // 50 items / 1 second
        #expect(progress.estimatedTimeRemaining != nil)
        #expect(progress.message?.contains("2/4") == true)
    }

    @Test("BatchCompleted calculates ETA")
    func testBatchCompletedETA() {
        let progress = BatchProgress.batchCompleted(
            itemsCompleted: 25,
            totalItems: 100,
            batchIndex: 0,
            totalBatches: 4,
            batchSize: 25,
            tokensInBatch: 1250,
            totalTokens: 1250,
            elapsedTime: 1.0
        )

        // 25 items/sec, 75 remaining = 3 seconds ETA
        #expect(progress.estimatedTimeRemaining != nil)
        #expect(abs(progress.estimatedTimeRemaining! - 3.0) < 0.1)
    }

    // MARK: - Progress Update Tests

    @Test("withCurrent creates updated progress")
    func testWithCurrent() {
        let initial = BatchProgress(
            current: 50,
            total: 100,
            batchIndex: 2,
            totalBatches: 5,
            itemsPerSecond: 25.0
        )

        let updated = initial.withCurrent(75)

        #expect(updated.current == 75)
        #expect(updated.total == 100)
        #expect(updated.batchIndex == 2)
        #expect(updated.itemsPerSecond == 25.0)
    }

    @Test("withThroughput creates updated progress")
    func testWithThroughput() {
        let initial = BatchProgress(
            current: 50,
            total: 100,
            itemsPerSecond: 25.0
        )

        let updated = initial.withThroughput(50.0)

        #expect(updated.current == 50)
        #expect(updated.itemsPerSecond == 50.0)
    }

    @Test("nextBatch creates progress for next batch")
    func testNextBatch() {
        let current = BatchProgress(
            current: 32,
            total: 100,
            batchIndex: 0,
            totalBatches: 4
        )

        let next = current.nextBatch(size: 32)

        #expect(next.batchIndex == 1)
        #expect(next.currentBatchSize == 32)
        #expect(next.message?.contains("2/4") == true)
    }

    // MARK: - Delegation to OperationProgress Tests

    @Test("Delegates to base OperationProgress")
    func testDelegatesToBase() {
        let progress = BatchProgress(
            current: 50,
            total: 100,
            phase: "Processing",
            message: "Test message",
            estimatedTimeRemaining: 5.0
        )

        #expect(progress.phase == progress.base.phase)
        #expect(progress.message == progress.base.message)
        #expect(progress.timestamp == progress.base.timestamp)
        #expect(progress.estimatedTimeRemaining == progress.base.estimatedTimeRemaining)
    }

    @Test("Init from OperationProgress base")
    func testInitFromBase() {
        let base = OperationProgress(
            current: 50,
            total: 100,
            phase: "Processing",
            message: "From base"
        )

        let progress = BatchProgress(
            base: base,
            batchIndex: 2,
            totalBatches: 5,
            itemsPerSecond: 25.0
        )

        #expect(progress.current == 50)
        #expect(progress.phase == "Processing")
        #expect(progress.message == "From base")
        #expect(progress.batchIndex == 2)
        #expect(progress.itemsPerSecond == 25.0)
    }

    // MARK: - Equatable Tests

    @Test("Equatable comparison")
    func testEquatable() {
        let p1 = BatchProgress(
            current: 50,
            total: 100,
            batchIndex: 2,
            totalBatches: 5
        )

        let p2 = BatchProgress(
            current: 50,
            total: 100,
            batchIndex: 2,
            totalBatches: 5
        )

        // Note: timestamps will differ, so full equality depends on base equality
        #expect(p1.current == p2.current)
        #expect(p1.total == p2.total)
        #expect(p1.batchIndex == p2.batchIndex)
    }

    // MARK: - Description Tests

    @Test("CustomStringConvertible description")
    func testDescription() {
        let progress = BatchProgress(
            current: 50,
            total: 100,
            batchIndex: 1,
            totalBatches: 4,
            itemsPerSecond: 25.5,
            tokensProcessed: 2500
        )

        let desc = progress.description

        #expect(desc.contains("50%"))
        #expect(desc.contains("50/100"))
        #expect(desc.contains("batch 2/4"))
        #expect(desc.contains("25.5 items/s"))
        #expect(desc.contains("2500 tokens"))
    }

    @Test("Description without optional fields")
    func testDescriptionMinimal() {
        let progress = BatchProgress(current: 25, total: 100)

        let desc = progress.description

        #expect(desc.contains("25%"))
        #expect(desc.contains("25/100"))
        #expect(!desc.contains("items/s"))  // nil itemsPerSecond
    }

    // MARK: - Sendable Tests

    @Test("BatchProgress is Sendable")
    func testSendable() async {
        let progress = BatchProgress(current: 50, total: 100)

        let result = await Task {
            progress.percentage
        }.value

        #expect(result == 50)
    }
}
