# Metal Shader Implementation Guide
<!-- renamed for neutral naming -->

## Critical Issues Summary

### 🔴 High Priority (Must Fix)

1. **Monolithic String Literal** → Separate .metal files
   - **Current**: 331-line string hampering development
   - **Fix**: Individual .metal files with syntax highlighting
   - **Impact**: 10x faster iteration, compile-time validation

2. **Numerical Stability** → Two-pass norm computation
   - **Current**: Single-pass can overflow/underflow
   - **Fix**: Scale by max value before summation
   - **Impact**: Handles values from 1e-20 to 1e+20 safely

3. **Suboptimal Memory Access** → Coalesced reads/writes
   - **Current**: Scattered memory access in L2 normalize
   - **Fix**: All threads cooperate on norm, then each writes one element
   - **Impact**: 15-30% performance improvement

### 🟡 Medium Priority (Should Fix)

4. **No Function Constants** → Runtime specialization
   - **Current**: Hardcoded values (epsilon, tile sizes)
   - **Fix**: Use MTLFunctionConstantValues for specialization
   - **Impact**: Better numerical control, platform optimization

5. **Manual Vectorization** → Better float4 usage
   - **Current**: Manual unpacking of float4
   - **Fix**: Use proper vector types and FMA instructions
   - **Impact**: 10-15% performance gain

6. **No Precompiled Metallib** → Runtime compilation overhead
   - **Current**: 150-200ms first launch compile
   - **Fix**: Build-time compilation to .metallib
   - **Impact**: 20x faster cold start

### 🟢 Low Priority (Nice to Have)

7. **Limited Testing** → Comprehensive validation suite
8. **No Profiling Hooks** → Performance measurement infrastructure
9. **Missing Tiled Kernels** → Large matrix optimizations

---

## Implementation Checklist

### Phase 1: Infrastructure Setup (3-5 days)

#### Day 1: Directory Structure
- [ ] Create `Sources/EmbedKit/Shaders/` directory
- [ ] Create subdirectories: `Common/`, `Kernels/`, `Tests/`
- [ ] Add `.metal` extension to Xcode project settings

#### Day 2: Common Headers
- [ ] Create `MetalCommon.h` with shared types
- [ ] Define aligned structs (16-byte alignment)
- [ ] Add function constant declarations
- [ ] Add bounds checking utilities

#### Day 3: Build System
- [ ] Add "Compile Metal Shaders" build phase
- [ ] Configure Metal compiler flags (`-std=metal3.0`, `-O3`)
- [ ] Set up metallib output path
- [ ] Test build process

#### Day 4: Swift Integration
- [ ] Update `MetalShaderLibrary.swift`
- [ ] Add `loadLibrary()` for metallib loading
- [ ] Implement function constants configuration
- [ ] Add fallback to source compilation

#### Day 5: Basic Testing
- [ ] Create `MetalShaderValidationTests.swift`
- [ ] Add `testAllKernelsCompile()` test
- [ ] Set up benchmark framework
- [ ] Run baseline performance measurements

### Phase 2: L2 Normalization Refactor (2-3 days)

#### Step 1: Extract Current Kernel
```bash
# Create new file
touch Sources/EmbedKit/Shaders/Kernels/Normalization.metal
```

- [ ] Copy L2 normalize kernel to `Normalization.metal`
- [ ] Add `#include "MetalCommon.h"`
- [ ] Update kernel signature with proper attributes

#### Step 2: Implement Two-Pass Algorithm
- [ ] Add `stable_l2_norm()` function to `MetalMath.metal`
- [ ] Implement max-finding pass
- [ ] Implement scaled summation pass
- [ ] Add epsilon protection

#### Step 3: Optimize Memory Access
- [ ] Phase 1: All threads cooperate on norm (SIMD reduction)
- [ ] Phase 2: Each thread normalizes one element
- [ ] Add threadgroup barrier if needed
- [ ] Validate coalesced access pattern

#### Step 4: Testing
- [ ] Test zero vectors → should output zero
- [ ] Test tiny values (1e-20) → no underflow
- [ ] Test huge values (1e+20) → no overflow
- [ ] Test mixed magnitudes → correct norm
- [ ] Benchmark vs current implementation

**Acceptance Criteria**:
- ✅ All numerical stability tests pass
- ✅ Performance ≥ current implementation
- ✅ Zero regression in existing tests

### Phase 3: Pooling Kernels (2-3 days)

#### Step 1: Mean Pooling
- [ ] Create `Pooling.metal`
- [ ] Port mean pooling kernel
- [ ] Add tiling for cache efficiency
- [ ] Optimize mask handling (reduce branch divergence)

#### Step 2: Max Pooling
- [ ] Port max pooling kernel
- [ ] Add float4 vectorization
- [ ] Handle boundary conditions
- [ ] Test with various sequence lengths

#### Step 3: Attention-Weighted Pooling
- [ ] Port attention pooling kernel
- [ ] Use FMA for weighted sum
- [ ] Add softmax normalization (if needed)
- [ ] Test with various weight distributions

**Acceptance Criteria**:
- ✅ Masked pooling works correctly
- ✅ Boundary cases handled (empty mask, all zeros)
- ✅ Performance within 5% of target

### Phase 4: Similarity Kernels (2-3 days)

#### Step 1: Pairwise Similarity
- [ ] Create `Similarity.metal`
- [ ] Port cosine similarity matrix kernel
- [ ] Optimize dot product with FMA
- [ ] Add float4 vectorization

#### Step 2: Batch Similarity
- [ ] Port batch similarity kernel
- [ ] Optimize SIMD group reductions
- [ ] Test various batch sizes
- [ ] Validate output range [-1, 1]

#### Step 3: Tiled Similarity (Optional)
- [ ] Implement tiled kernel for large matrices
- [ ] Add threadgroup memory caching
- [ ] Benchmark vs non-tiled version
- [ ] Document when to use each variant

**Acceptance Criteria**:
- ✅ Orthogonal vectors → 0.0 similarity
- ✅ Identical vectors → 1.0 similarity
- ✅ Opposite vectors → -1.0 similarity
- ✅ Output always in [-1, 1] range

### Phase 5: Production Hardening (2-3 days)

#### Error Handling
- [ ] Validate all buffer sizes
- [ ] Check for null pointers
- [ ] Handle device loss gracefully
- [ ] Add meaningful error messages

#### Edge Cases
- [ ] Empty inputs
- [ ] Single-element inputs
- [ ] Maximum-size inputs
- [ ] Mismatched dimensions

#### Memory Management
- [ ] Test memory pressure scenarios
- [ ] Validate buffer reuse
- [ ] Check for memory leaks
- [ ] Profile peak memory usage

#### Documentation
- [ ] Update API documentation
- [ ] Add shader development guide
- [ ] Create performance tuning guide
- [ ] Write migration guide for users

---

## Quick Fixes You Can Implement Today

### Fix #1: Add Epsilon to L2 Normalize (5 minutes)

**Current**:
```metal
const float inv_norm = (norm > 0.0f) ? (1.0f / norm) : 0.0f;
```

**Fixed**:
```metal
constant float EPSILON = 1e-8f;
const float inv_norm = (norm > EPSILON) ? (1.0f / norm) : 0.0f;
```

### Fix #2: Use FMA in Cosine Similarity (10 minutes)

**Current**:
```metal
dotProduct += queryVal * keyVal;
queryNorm += queryVal * queryVal;
```

**Fixed**:
```metal
dotProduct = fma(queryVal, keyVal, dotProduct);
queryNorm = fma(queryVal, queryVal, queryNorm);
```

**Impact**: 5-10% faster, better numerical accuracy

### Fix #3: Align Parameter Structs (5 minutes)

**Current**:
```swift
public struct PoolingParams {
    let sequenceLength: Int32
    let dimensions: Int32
}
```

**Fixed**:
```swift
@frozen
public struct PoolingParams {
    let sequenceLength: Int32
    let dimensions: Int32
    private let _padding0: Int32 = 0
    private let _padding1: Int32 = 0
}
```

**Impact**: Ensures proper alignment, prevents subtle bugs

---

## Performance Validation Script

Create this as a test to ensure no regression:

```swift
func testPerformanceBaseline() async throws {
    let accelerator = try XCTUnwrap(MetalAccelerator.shared)

    // Baseline configuration
    let batchSize = 1000
    let dimensions = 384
    let iterations = 10

    let vectors = (0..<batchSize).map { _ in
        (0..<dimensions).map { _ in Float.random(in: -1...1) }
    }

    var times: [Double] = []

    for _ in 0..<iterations {
        let start = CFAbsoluteTimeGetCurrent()
        _ = try await accelerator.normalizeVectors(vectors)
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        times.append(elapsed)
    }

    let median = times.sorted()[iterations / 2]
    let mean = times.reduce(0, +) / Double(iterations)

    print("Median: \(median * 1000) ms")
    print("Mean: \(mean * 1000) ms")

    // Fail if performance degrades by more than 10%
    let baselineMedian = 5.0  // Update with your baseline
    XCTAssertLessThan(median, baselineMedian * 1.1, "Performance regression detected")
}
```

---

## Code Review Checklist

Before merging any shader changes, verify:

### Correctness
- [ ] Unit tests pass (zero vectors, edge cases)
- [ ] Numerical stability tests pass (extreme values)
- [ ] Output validated against reference implementation
- [ ] No warnings from Metal validation layer

### Performance
- [ ] Benchmarks show no regression (±5% tolerance)
- [ ] Profiled with Xcode Metal Debugger
- [ ] Occupancy ≥ 75% for compute-bound kernels
- [ ] Memory bandwidth utilization ≥ 70%

### Code Quality
- [ ] Shader code follows style guide
- [ ] All parameters properly aligned
- [ ] Function constants used for configuration
- [ ] Comprehensive inline documentation

### Testing
- [ ] Tests added for new functionality
- [ ] Edge cases covered
- [ ] Platform-specific behavior tested (M1/M2/M3)
- [ ] Debug and release builds both work

---

## Common Mistakes to Avoid

### Mistake #1: Unaligned Structures
❌ **Wrong**:
```metal
struct Params {
    int32_t a;
    int32_t b;
};  // Only 8 bytes, may cause alignment issues
```

✅ **Correct**:
```metal
struct Params {
    int32_t a;
    int32_t b;
    int32_t padding0;
    int32_t padding1;
} __attribute__((aligned(16)));  // Explicitly 16-byte aligned
```

### Mistake #2: Forgetting Threadgroup Barriers
❌ **Wrong**:
```metal
shared_memory[tid] = input[tid];
// Missing barrier!
float sum = shared_memory[0] + shared_memory[1];  // Race condition!
```

✅ **Correct**:
```metal
shared_memory[tid] = input[tid];
threadgroup_barrier(mem_flags::mem_threadgroup);
float sum = shared_memory[0] + shared_memory[1];  // Safe
```

### Mistake #3: Not Clamping Similarity Scores
❌ **Wrong**:
```metal
output[idx] = dotProduct / normProduct;  // Can be outside [-1, 1] due to FP error
```

✅ **Correct**:
```metal
float similarity = dotProduct / normProduct;
output[idx] = clamp(similarity, -1.0f, 1.0f);
```

---

## Performance Optimization Quick Tips

1. **Memory Access**:
   - Always prefer coalesced access (consecutive threads → consecutive addresses)
   - Use threadgroup memory for data reuse (16KB L1 cache)
   - Align all data structures to 16 bytes

2. **SIMD Utilization**:
   - Use `simd_sum()` instead of manual reduction loops
   - Process data in float4 chunks when possible
   - Avoid divergent branches within SIMD groups

3. **Occupancy**:
   - Target 256-512 threads per threadgroup on Apple Silicon
   - Balance threadgroup size vs register usage
   - Check occupancy with Xcode GPU profiler

4. **Numerical Stability**:
   - Always use epsilon for comparisons and divisions
   - Consider two-pass algorithms for better stability
   - Use FMA for better accuracy and performance

---

## Resources

### Documentation
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Best Practices Guide](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)

### Tools
- Xcode Metal Debugger: Capture GPU frames and analyze
- Instruments Metal System Trace: Track GPU/CPU interaction
- Metal Validation Layer: Enable in scheme for runtime checks

### Books
- "Metal by Example" by Warren Moore
- "Metal Programming Guide" by Janie Clayton

---

## Success Metrics

Track these metrics throughout the refactor:

| Metric | Baseline | Target | Current |
|--------|----------|--------|---------|
| Compile time (cold) | 150ms | 10ms | TBD |
| L2 norm (1000x384) | XXms | -15% | TBD |
| Mean pool (512x384) | XXms | -10% | TBD |
| Cosine sim (256x256) | XXms | -20% | TBD |
| Test coverage | 0% | 90% | TBD |
| Peak GPU memory | XXmb | -10% | TBD |

Update "Current" column as you implement each phase.

---

## Getting Help

If you encounter issues:

1. **Metal Validation Errors**: Enable Metal API validation in scheme
2. **Crashes**: Check alignment of parameter structs
3. **Wrong Results**: Add printf debugging in shaders (metal-xcode 15+)
4. **Performance Issues**: Profile with Xcode Metal Debugger
5. **Build Issues**: Clean build folder, verify .metal files are in target

---

**Remember**: Make incremental changes, validate frequently, and maintain backward compatibility until the full refactor is complete. Good luck!
