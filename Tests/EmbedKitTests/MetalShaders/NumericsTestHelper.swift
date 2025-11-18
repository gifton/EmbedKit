import XCTest
@testable import EmbedKit

/// Helper utilities for configuring numerics in GPU tests
///
/// Use to toggle stable normalization (two‑pass) and epsilon threshold
/// before running GPU normalization tests.
enum NumericsTestHelper {
    /// Enable stable (two‑pass) GPU normalization with a given epsilon.
    /// Falls back silently if Metal is unavailable.
    static func enableStableNormalization(
        _ accelerator: MetalAccelerator?,
        epsilon: Float = 1e-8
    ) async {
        guard let accelerator = accelerator else { return }
        await accelerator.setNumerics(stableNormalization: true, epsilon: epsilon)
    }

    /// Disable stable normalization (use fast single‑pass) with a given epsilon.
    /// Falls back silently if Metal is unavailable.
    static func disableStableNormalization(
        _ accelerator: MetalAccelerator?,
        epsilon: Float = 1e-8
    ) async {
        guard let accelerator = accelerator else { return }
        await accelerator.setNumerics(stableNormalization: false, epsilon: epsilon)
    }
}

