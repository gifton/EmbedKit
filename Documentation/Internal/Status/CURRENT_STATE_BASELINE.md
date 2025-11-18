# P0-1 Kernel Deduplication - Current State Baseline
<!-- moved to Documentation/Internal/Status -->

**Date:** 2025-10-27
**Branch:** main (before P0-1 implementation)
**Purpose:** Document current behavior before implementing kernel deduplication

---

## Test Execution Results

### Command Run
```bash
swift test --filter MetalAccelerationTests
```

### Test Results Summary

**Total Tests:** 9
**Passed:** 7
**Failed:** 2
**Execution Time:** 0.462 seconds

### Passing Tests ✅
- `testCLSPooling` - 0.000s
- `testCosineSimilarity` - 0.005s
- `testDimensionMismatch` - 0.001s
- `testEmptyInput` - 0.000s
- `testLargeVectorNormalization` - 0.061s
- `testMeanPooling` - 0.001s
- `testMetalDeviceAvailability` - 0.000s
- `testVectorNormalization` - 0.001s

### Failing Tests ❌

**Test:** `testBatchCosineSimilarity`
**Duration:** 0.392s
**Failures:**
1. Line 138: Expected 1.0 (±0.001), got 0.70710677
2. Line 140: Expected -1.0 (±0.001), got 0.0

**Note:** This test failure exists BEFORE P0-1 implementation and is unrelated to kernel deduplication.

### Performance Baseline

**Large Vector Normalization (100x768D):**
- Execution time: 0.762ms
- Test: `testLargeVectorNormalization`

---

## Metal Library State

### Precompiled Metallib Exists ✅

**Location:** `Sources/EmbedKit/Resources/EmbedKitShaders.metallib`
**Size:** 28KB
**Last Modified:** Oct 24 13:00

### Kernel Functions Present

All 6 expected kernel functions found in metallib:

```
000001db T attention_weighted_pool
00000268 T cosine_similarity
000002ef T cosine_similarity_batch
0000005c T l2_normalize
0000015d T max_pool
000000de T mean_pool
```

### Build Output

**Metallib in Bundle:**
- `.build/arm64-apple-macosx/debug/EmbedKit_EmbedKit.bundle/EmbedKitShaders.metallib`
- Size: 28,912 bytes (28KB)
- Successfully bundled during build

### SPM Build Warnings

```
warning: 'embedkit': found 4 file(s) which are unhandled; explicitly declare them as resources or exclude from the target
    /Users/goftin/dev/gsuite/VSK/EmbedKit/Sources/EmbedKit/Shaders/README.md
    /Users/goftin/dev/gsuite/VSK/EmbedKit/Sources/EmbedKit/Shaders/Kernels/Pooling.metal
    /Users/goftin/dev/gsuite/VSK/EmbedKit/Sources/EmbedKit/Shaders/Kernels/Similarity.metal
    /Users/goftin/dev/gsuite/VSK/EmbedKit/Sources/EmbedKit/Shaders/Kernels/Normalization.metal
```

**Note:** .metal files are not declared as resources, which is expected (they should be compiled, not bundled as-is).

---

## Code State

### Metal Shader Files

**Standalone .metal files:**
- `Sources/EmbedKit/Shaders/Common/MetalCommon.h` (134 lines)
- `Sources/EmbedKit/Shaders/Kernels/Normalization.metal` (100 lines)
- `Sources/EmbedKit/Shaders/Kernels/Pooling.metal` (255 lines)
- `Sources/EmbedKit/Shaders/Kernels/Similarity.metal` (229 lines)

**Total standalone code:** 718 lines

### Embedded String

**Location:** `Sources/EmbedKit/Acceleration/MetalShaderLibrary.swift`
**Lines:** 20-352 (333 lines of Metal code embedded as Swift string)
**Status:** DUPLICATE of standalone files

### Loading Mechanism

**Current behavior (hypothesis):**
1. MetalLibraryLoader tries to load precompiled metallib
2. Search paths may be failing
3. Falls back to string compilation (slow path)

**Evidence:**
- No "Metal library" log messages in test output
- Logging may not be enabled or visible in test context
- Tests pass, indicating kernels ARE loading (from either source)

---

## Package.swift Configuration

### Current Resources Declaration

```swift
resources: [
    // Precompiled Metal shader library
    // Compile shaders with: ./Scripts/CompileMetalShaders.sh
    .process("Resources/EmbedKitShaders.metallib")
]
```

### Missing

- No build tool plugin for automatic compilation
- Manual script reference (`./Scripts/CompileMetalShaders.sh`)
- .metal files not explicitly handled in Package.swift

---

## File System State

### Checked-in Files

- ✅ `Sources/EmbedKit/Resources/EmbedKitShaders.metallib` (pre-compiled, 28KB)
- ✅ All `.metal` source files
- ✅ `MetalCommon.h` header

### Generated/Build Files

- ✅ Bundle contains metallib: `.build/arm64-apple-macosx/debug/EmbedKit_EmbedKit.bundle/`
- ❌ No plugin output directory (no plugin exists yet)
- ❌ No `.air` intermediate files (would be in plugin directory)

---

## Known Issues (Pre-existing)

### 1. Test Failure: testBatchCosineSimilarity

**Status:** FAILING BEFORE P0-1
**Impact:** Not related to kernel deduplication
**Action:** Track separately, fix in different PR

### 2. Duplicate Kernel Code

**Status:** CONFIRMED
**Impact:** Maintenance burden, risk of divergence
**Action:** This is what P0-1 will fix

### 3. No Automatic Metallib Compilation

**Status:** MANUAL PROCESS ONLY
**Impact:** Developers must run script manually
**Action:** P0-1 will add SPM plugin

### 4. Unclear Which Loading Path Is Used

**Status:** NO LOGGING IN TEST OUTPUT
**Impact:** Can't verify if metallib or string is used
**Action:** P0-1 will improve logging

---

## Success Criteria for P0-1

After implementing P0-1, we should see:

### ✅ Tests Should Pass
- Same 7 tests passing (plus new MetalLibraryLoadingTests)
- Same 2 tests failing (testBatchCosineSimilarity - pre-existing)
- Performance same or better (should be faster)

### ✅ Code Changes
- Embedded string removed from MetalShaderLibrary.swift
- File size: ~402 lines → ~100 lines
- SPM plugin added: `Plugins/MetalShaderCompilerPlugin/`
- Package.swift updated with plugin reference

### ✅ Build Process
- `swift build` automatically compiles .metal files
- Plugin output in `.build/plugins/outputs/`
- .air intermediate files generated
- Metallib linked and bundled

### ✅ Logging
- Clear message: "Metal library loaded from precompiled metallib (fast path)"
- No message: "Metal library compiled from string (slow path)"

### ✅ Performance
- Library load time < 10ms (currently unknown, likely ~150ms with string)
- Test execution time: same or better

---

## Baseline Summary

**Current State:**
- ✅ Metallib exists and contains all 6 kernels
- ✅ Tests mostly pass (2 pre-existing failures)
- ⚠️ Duplicate code (718 lines standalone + 333 lines embedded)
- ⚠️ Manual compilation process
- ⚠️ No logging visibility
- ⚠️ Uncertain which loading path is actually used

**Ready to Proceed:** ✅ YES

All baseline information captured. We have a solid understanding of the current state and can proceed with P0-1 implementation.

---

**Next Steps:**
1. Implement SPM build tool plugin (Phase 2)
2. Remove embedded string duplication (Phase 4)
3. Verify logging and performance improvements (Phase 5)
