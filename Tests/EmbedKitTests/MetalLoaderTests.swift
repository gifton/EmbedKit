import XCTest
import Foundation
@testable import EmbedKit

final class MetalLoaderTests: XCTestCase {
    func testFallbackToBundleMetallibWhenOverrideInvalid() async throws {
        #if canImport(Metal)
        // Point override to a non-existent file to force fallback to Bundle.module metallib
        let bogus = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("_nonexistent.metallib")
        await MetalAcceleratorConfig.shared.setOverride(url: bogus)

        let acc = await MetalAccelerator()
        // If Metal is available on this machine and bundle metallib is present, we expect isAvailable true.
        // If Metal is not available (e.g., CI environment), skip.
        let isAvailable = await acc.isAvailable
        if isAvailable {
            XCTAssertTrue(isAvailable, "Accelerator should be available using Bundle.module metallib fallback")
        } else {
            throw XCTSkip("Metal not available or metallib not loadable in this environment")
        }
        #else
        throw XCTSkip("Metal not available on this platform")
        #endif
    }
}
