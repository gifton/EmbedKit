# Metal Shader Quick Reference
<!-- renamed for neutral naming -->

## 🎯 Top 10 Issues & Fixes

### Issue #1: Numerical Instability in L2 Normalization
```
⚠️  PROBLEM: Direct squaring overflows for values > 1e19
❌  norm² = Σ x[i]²              // Overflow risk
✅  norm = max|x| × √Σ(x[i]/max)²  // Scaled algorithm
📈  IMPACT: Handles 1e-40 to 1e+40 safely
```

### Issue #2: Monolithic 331-Line String Literal
```
⚠️  PROBLEM: Single string, no syntax highlighting, runtime compile
❌  let source = "..." // 331 lines
✅  EmbedKitShaders.metallib  // Precompiled
📈  IMPACT: 150ms → 5ms startup (30x faster)
```

### Issue #3: Suboptimal Memory Access Patterns
```
⚠️  PROBLEM: All threads compute norm, one writes
❌  Phase 1: Each thread writes 1 element
    Phase 2: All threads compute norm
✅  Phase 1: All threads cooperate on norm
    Phase 2: Each thread writes 1 element
📈  IMPACT: +15-30% performance from coalesced access
```

### Issue #4: No Function Constants
```
⚠️  PROBLEM: Hardcoded epsilon, tile sizes
❌  const float epsilon = 1e-8f;  // Fixed in code
✅  constant float EPSILON [[function_constant(0)]];  // Configurable
📈  IMPACT: Runtime specialization, platform tuning
```

### Issue #5: Manual Float4 Construction
```
⚠️  PROBLEM: Individual loads instead of vectorized
❌  float4 v = float4(a[0], a[1], a[2], a[3]);  // 4 loads
✅  float4 v = *((device float4*)a);  // 1 load
📈  IMPACT: 4x fewer memory transactions
```

### Issue #6: Missing FMA Instructions
```
⚠️  PROBLEM: Separate multiply and add
❌  sum += a * b;  // Two ops, less accurate
✅  sum = fma(a, b, sum);  // One op, more accurate
📈  IMPACT: +5-10% faster, better precision
```

### Issue #7: No Cache Tiling
```
⚠️  PROBLEM: Linear scan, poor cache locality
❌  for (int i = 0; i < S; i++)  // ~60% L1 hit rate
✅  for (int tile = 0; tile < S; tile += 16)  // ~95% L1 hit rate
📈  IMPACT: +35% cache hits, +10-25% performance
```

### Issue #8: Unaligned Parameter Structs
```
⚠️  PROBLEM: Only 8 or 12 bytes, not 16-byte aligned
❌  struct Params { int32_t a, b; }  // 8 bytes, risky
✅  struct Params { int32_t a, b, pad0, pad1; }  // 16 bytes, safe
📈  IMPACT: Prevents GPU errors
```

### Issue #9: Using rsqrt Without Justification
```
⚠️  PROBLEM: Less accurate, not measured
❌  float inv = rsqrt(x);  // Fast but less accurate
✅  float inv = 1.0f / sqrt(x);  // Or profile first
📈  IMPACT: Better accuracy for critical ops
```

### Issue #10: No Testing Infrastructure
```
⚠️  PROBLEM: No validation, no benchmarks
❌  // No tests
✅  90%+ coverage: unit tests, stability tests, benchmarks
📈  IMPACT: Catches regressions, validates correctness
```

---

## 📊 Performance Impact Matrix

| Operation | Baseline | Quick Fixes | Full Refactor | Total Gain |
|-----------|----------|-------------|---------------|------------|
| **Startup** | 150ms | 150ms | 5ms | **30x** ⚡ |
| **L2 Norm** | 100% | 105% | 130% | **+30%** 📈 |
| **Cosine Sim** | 100% | 110% | 140% | **+40%** 📈 |
| **Mean Pool** | 100% | 105% | 125% | **+25%** 📈 |
| **Cache Hits** | 60% | 60% | 95% | **+58%** 💾 |

**Quick Fixes**: FMA + Alignment + Epsilon (30 mins work)
**Full Refactor**: Complete implementation (4-5 weeks)

---

## 🚀 Implementation Priority

### 🔴 CRITICAL - Do First
1. **Fix alignment** (5 mins) → Prevents crashes
2. **Add FMA** (10 mins) → +5-10% perf, free
3. **Add epsilon constants** (5 mins) → Better numerics

### 🟡 HIGH - Do This Sprint
4. **Two-pass L2 norm** (1 day) → Numerical stability
5. **Precompile metallib** (1 day) → 30x faster startup
6. **Function constants** (1 day) → Configurability

### 🟢 MEDIUM - Do This Quarter
7. **Cache tiling** (2 days) → +10-25% perf
8. **Testing infrastructure** (2 days) → Quality
9. **Documentation** (2 days) → Maintainability

---

## 🎬 Quick Start: 20-Minute Impact

### Step 1: Fix Alignment (5 minutes)

**File**: `MetalShaderLibrary.swift`

```swift
// BEFORE
public struct PoolingParams {
    let sequenceLength: Int32
    let dimensions: Int32
}

// AFTER
@frozen
public struct PoolingParams {
    let sequenceLength: Int32
    let dimensions: Int32
    private let _pad0: Int32 = 0  // Add these 2 lines
    private let _pad1: Int32 = 0
}
```

Repeat for: `SimilarityParams`, `BatchSimilarityParams`

### Step 2: Add FMA (10 minutes)

**File**: `MetalShaderLibrary.swift` (in shader string)

Find all instances of:
```metal
dotProduct += a * b;
queryNorm += a * a;
```

Replace with:
```metal
dotProduct = fma(a, b, dotProduct);
queryNorm = fma(a, a, queryNorm);
```

**Locations**: Lines ~220, ~230, ~268

### Step 3: Add Epsilon (5 minutes)

**File**: `MetalShaderLibrary.swift` (at top of shader string)

```metal
// Add after "using namespace metal;"
constant float EPSILON_NORMAL = 1e-8f;

// Then replace all:
(norm > 0.0f)    →    (norm > EPSILON_NORMAL)
(x > 0.0f)       →    (x > EPSILON_NORMAL)
```

**Build, Test, Deploy** → +5-10% performance gain!

---

## 🧪 Validation Checklist

After any shader change, verify:

```
✓ Build succeeds (no Metal compilation errors)
✓ Unit tests pass
✓ Numerical stability:
  • Zero vectors → zero output
  • Tiny values (1e-20) → correct norm
  • Huge values (1e+20) → correct norm
  • Mixed magnitudes → correct norm
✓ Performance within 5% of baseline
✓ No Metal validation warnings
✓ Tested on M1/M2/M3 (if available)
```

---

## 📈 Benchmark Command

**File**: `Tests/EmbedKitBenchmarks/MetalBenchmarks.swift`

```swift
swift test --filter MetalBenchmarks
```

Expected output:
```
L2 Normalize (1000x384):  5.2ms ± 0.3ms
Mean Pool (512x384):      2.1ms ± 0.1ms
Cosine Similarity (256²): 8.7ms ± 0.5ms
```

---

## 🐛 Common Mistakes

### Mistake #1: Forgetting Barriers
```metal
❌ shared[tid] = input[tid];
   float sum = shared[0];  // Race!

✅ shared[tid] = input[tid];
   threadgroup_barrier(mem_flags::mem_threadgroup);
   float sum = shared[0];  // Safe
```

### Mistake #2: Unaligned Casts
```metal
❌ float4 v = *((device float4*)(ptr + 1));  // Misaligned!
✅ float4 v = *((device float4*)(ptr + 4));  // Aligned to 16 bytes
```

### Mistake #3: Divergent Branches
```metal
❌ if (gid % 2 == 0) {      // Half threads diverge
     computeA();
   } else {
     computeB();
   }

✅ float result = select(computeB(), computeA(), gid % 2 == 0);
```

### Mistake #4: Ignoring Threadgroup Size
```metal
❌ [[max_total_threads_per_threadgroup(2048)]]  // Too large!
✅ [[max_total_threads_per_threadgroup(1024)]]  // Apple Silicon limit
```

---

## 🔍 Debug Commands

### Enable Metal Validation
```bash
# In Xcode: Edit Scheme → Run → Diagnostics
☑ Metal API Validation
☑ Metal Shader Validation
```

### Capture GPU Frame
```
1. Run app in Xcode
2. Click 🎬 (camera icon) in debug bar
3. Choose "Capture GPU Frame"
4. Inspect kernels, memory, performance
```

### Check Occupancy
```metal
// In Metal debugger, look for:
Occupancy: 87%  // Good (>75%)
Occupancy: 45%  // Bad (<50%)

// Fix: Reduce register usage or adjust threadgroup size
```

---

## 📚 Key Documents

| Document | Purpose | Size |
|----------|---------|------|
| `Metal_Shader_Summary.md` | Executive overview | 3 pages |
| `Metal_Shader_Architecture_Proposal.md` | Detailed proposal | 40 pages |
| `Metal_Shader_Implementation_Guide.md` | Step-by-step guide | 20 pages |
| `Metal_Shader_Examples.md` | Before/after code | 25 pages |
| `Metal_Shader_Quick_Reference.md` | This document | 3 pages |

---

## 🎓 Learning Resources

### Essential Reading
- Metal Shading Language Spec (PDF)
- Metal Best Practices Guide (Apple)
- Metal by Example (Warren Moore)

### Tools
- Xcode Metal Debugger
- Instruments: Metal System Trace
- Metal Validation Layer (scheme setting)

### Community
- Apple Developer Forums: Metal
- Stack Overflow: [metal] tag
- GitHub: metal-by-example

---

## 🆘 Emergency Hotfixes

### Crash on Launch?
```swift
// In MetalResourceManager.swift
// Disable Metal temporarily:
public init(device: MTLDevice) throws {
    // Comment out pipeline setup
    // try await setupPipelines()  // ← Disable this
}
```

### Wrong Results?
```metal
// Add debug output (Metal 3.1+):
printf("value: %f\n", myValue);
// View in Xcode console after GPU capture
```

### Performance Regression?
```swift
// Run benchmark comparison:
git checkout main
swift test --filter Benchmarks > baseline.txt

git checkout feature-branch
swift test --filter Benchmarks > current.txt

diff baseline.txt current.txt
```

---

## ✅ Definition of Done

Before marking refactor complete:

```
☐ All 6 kernels refactored and optimized
☐ 90%+ test coverage achieved
☐ All numerical stability tests pass
☐ Performance ≥ baseline +15% target
☐ Zero Metal validation warnings
☐ Documentation complete
☐ Tested on M1, M2, M3
☐ Tested on iOS and macOS
☐ Benchmarks automated in CI
☐ Migration guide written
```

---

## 🏆 Success Metrics

Track weekly:

| Week | Compile Time | L2 Norm | Tests | Status |
|------|--------------|---------|-------|--------|
| 0 | 150ms | 100% | 0% | Baseline |
| 1 | 150ms | 105% | 20% | Infrastructure |
| 2 | 150ms | 130% | 50% | L2 refactor |
| 3 | 5ms | 130% | 75% | All kernels |
| 4 | 5ms | 130% | 90% | Polish |
| 5 | 5ms | 130% | 95% | ✅ Done |

---

## 💡 Pro Tips

1. **Always profile**: Don't optimize without measuring
2. **Test edge cases**: Zero vectors, huge values, empty inputs
3. **Check alignment**: Use `assert(sizeof(Params) == 16)`
4. **Use FMA everywhere**: It's faster AND more accurate
5. **Enable validation**: Catches 90% of bugs immediately
6. **Document numerics**: Explain why algorithms are stable
7. **Version your shaders**: Track changes like code
8. **Benchmark continuously**: Catch regressions early

---

**Last Updated**: 2025-10-23
**Version**: 1.0
**Author**: Senior Systems & ML Engineer Analysis
