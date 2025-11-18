# L2 Normalization Fix Plan
<!-- moved to Documentation/Internal/EngineeringNotes -->

## Overview
Fix L2 normalization for non-multiple-of-32 dimensions by ensuring proper SIMD group dispatch and adding robustness improvements.

## Phase 1: Fix Thread Dispatch (EXECUTE NOW)
**Goal:** Ensure single SIMD group processes each vector's norm computation

### Changes Required:
1. **MetalVectorProcessor.swift**:
   - Query `threadExecutionWidth` instead of assuming 32
   - Dispatch threads to ensure single SIMD group per vector
   - Use dimension-aware dispatch for correct reduction

### Implementation Strategy:
```swift
// Current (BROKEN):
let threadsPerGrid = MTLSize(width: dimensions, height: batchSize, depth: 1)
let threadsPerThreadgroup = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)

// Fixed (PHASE 1):
let simdWidth = pipeline.threadExecutionWidth  // Query actual SIMD width
let threadsPerGrid = MTLSize(
    width: min(dimensions, simdWidth),  // Single SIMD group handles all dims
    height: batchSize,
    depth: 1
)
let threadsPerThreadgroup = MTLSize(
    width: min(dimensions, simdWidth),  // Match grid width
    height: 1,
    depth: 1
)
```

### Kernel Adjustments:
- Ensure kernel handles case where `gid.x < dimensions`
- Each thread still writes one output element
- SIMD reduction works correctly with all threads in single group

## Phase 2: Add Zero Vector Guards & NaN Protection
**Goal:** Robust handling of edge cases

### Changes Required:
1. **Normalization.metal**:
   ```metal
   // Add zero vector guard
   const float norm = metal::sqrt(norm_squared);
   const float inv_norm = (norm > EPSILON_NORMAL) ? (1.0f / norm) : 0.0f;

   // For zero vectors, output zero vector (not NaN)
   if (norm <= EPSILON_NORMAL) {
       output[baseIndex + dimIndex] = 0.0f;
   } else {
       output[baseIndex + dimIndex] = input[baseIndex + dimIndex] * inv_norm;
   }
   ```

2. **Add float16 path support** (if needed)

## Phase 3: Comprehensive Unit Tests
**Goal:** Test all edge cases and dimensions

### Test Matrix:
1. **Dimension Coverage**:
   - All dimensions 1-129 individually
   - Focus on: 31, 32, 33, 63, 64, 65, 127, 128, 129

2. **Input Patterns**:
   - Random vectors
   - All zeros (test NaN protection)
   - Single non-zero at last index
   - Denormal values
   - Very large/small values
   - Negative values

3. **Batch Testing**:
   - Single vector
   - Multiple vectors with different dimensions

### Test Implementation:
```swift
func testComprehensiveL2Normalization() async throws {
    // Test all dimensions 1-129
    for dim in 1...129 {
        // Random vector
        let random = (0..<dim).map { _ in Float.random(in: -10...10) }

        // Zero vector
        let zeros = Array(repeating: Float(0), count: dim)

        // Single non-zero at end
        var singleNonZero = zeros
        singleNonZero[dim-1] = 1.0

        // Denormals
        let denormals = (0..<dim).map { _ in Float.leastNormalMagnitude }

        // Test each pattern
        for (name, vector) in [
            ("random", random),
            ("zeros", zeros),
            ("single", singleNonZero),
            ("denormals", denormals)
        ] {
            // Test normalization
            let batch = try VectorBatch(vectors: [vector])
            let normalized = try await accelerator.normalizeVectors(batch)

            // Verify results
            verifyNormalization(name, dim, vector, normalized)
        }
    }
}
```

## Phase 4: Optimize Occupancy (Optional)
**Goal:** Better GPU utilization for large batches

### Strategy:
- Use multiple SIMD groups per threadgroup (e.g., 128 threads = 4 SIMD groups)
- Each SIMD group processes one vector independently
- Index by `simdgroup_index_in_threadgroup`

### Implementation:
```metal
kernel void l2_normalize_optimized(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int32_t& dimensions [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]
) {
    // Each SIMD group handles a different vector
    const uint vectorIndex = gid.y * simdgroups_per_threadgroup + simdgroup_id;
    // ... rest of kernel
}
```

## Success Criteria

### Phase 1 Success:
- ✅ Dimensions 33, 34, 65, etc. produce magnitude = 1.0 ± 0.01
- ✅ No performance regression for power-of-2 dimensions
- ✅ Existing tests continue to pass

### Phase 2 Success:
- ✅ Zero vectors return zero vectors (not NaN)
- ✅ Denormal inputs handled correctly
- ✅ No numerical overflow/underflow

### Phase 3 Success:
- ✅ All dimensions 1-129 pass tests
- ✅ All edge cases handled correctly
- ✅ Test coverage > 95%

### Phase 4 Success:
- ✅ 20%+ throughput improvement for batch size > 100
- ✅ Better GPU occupancy metrics

## Timeline

- **Phase 1**: Immediate (15 mins) - CRITICAL FIX
- **Phase 2**: Next (10 mins) - Robustness
- **Phase 3**: Follow-up (30 mins) - Validation
- **Phase 4**: Optional (20 mins) - Optimization

## Risk Assessment

**Phase 1 Risks**:
- Potential performance impact if dispatch pattern changes
- Mitigation: Benchmark before/after

**Phase 2 Risks**:
- Slight performance overhead from additional checks
- Mitigation: Use fast-math where safe

**Phase 3 Risks**:
- Test execution time
- Mitigation: Parallelize tests

**Phase 4 Risks**:
- Increased kernel complexity
- Mitigation: Keep original kernel as fallback
