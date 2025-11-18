# EmbedKit Metal Shaders

This directory contains GPU-accelerated Metal compute kernels for embedding operations.

## Directory Structure

```
Shaders/
├── Common/
│   └── MetalCommon.h           # Shared types, constants, utilities
├── Kernels/
│   ├── Normalization.metal     # L2 normalization kernel
│   ├── NormalizationBatchOptimized.metal  # Batch-optimized normalization
│   ├── Pooling.metal           # Pooling kernels (mean, max, attention)
│   └── Similarity.metal        # Cosine similarity kernels
└── README.md                   # This file
```

## Kernels Overview

### Normalization.metal
- **`l2_normalize`**: L2 (Euclidean) normalization to unit length
- **`l2_normalize_batch_optimized`**: Batch-optimized normalization for higher throughput

### Pooling.metal
- **`mean_pool`**: Average pooling across sequence dimension
- **`max_pool`**: Max pooling across sequence dimension
- **`attention_weighted_pool`**: Weighted average using attention scores

### Similarity.metal
- **`cosine_similarity`**: Pairwise cosine similarity matrix (queries × keys)
- **`cosine_similarity_batch`**: Batch cosine similarity for vector pairs

## Manual Compilation

For development and testing, you can compile shaders manually:

### Compile Individual Kernel
```bash
cd Sources/EmbedKit/Shaders

# Compile to .air (Metal intermediate representation)
xcrun metal -std=metal3.0 -O3 -c Kernels/Normalization.metal -o /tmp/norm.air
xcrun metal -std=metal3.0 -O3 -c Kernels/Pooling.metal -o /tmp/pool.air
xcrun metal -std=metal3.0 -O3 -c Kernels/Similarity.metal -o /tmp/sim.air
```

### Link into Metallib
```bash
# Create metallib from .air files
xcrun metallib \
    /tmp/norm.air \
    /tmp/pool.air \
    /tmp/sim.air \
    -o /tmp/EmbedKitShaders.metallib
```

### Inspect Metallib
```bash
# List all functions in metallib
xcrun metal-nm /tmp/EmbedKitShaders.metallib

# View detailed information
xcrun metal-objdump --macho --private-headers /tmp/EmbedKitShaders.metallib
```

## Development Workflow

### 1. Edit Shader
Edit `.metal` files in `Kernels/` directory using Xcode for syntax highlighting.

### 2. Compile and Test
```bash
# Compile to verify syntax
xcrun metal -std=metal3.0 -Wall -Wextra -c Kernels/YourKernel.metal

# Link and inspect
xcrun metallib -o test.metallib YourKernel.air
xcrun metal-nm test.metallib
```

### 3. Update Swift Code
If adding new kernels:
1. Add kernel name to `MetalShaderLibrary.KernelName` enum
2. Update Swift parameter structs if needed (must match Metal structs exactly)
3. Create wrapper function in appropriate processor (Vector/Pooling/Similarity)

### 4. Test
Run existing tests to ensure no regressions:
```bash
swift test
```

## Coding Conventions

### Metal Code Style
- Use descriptive function and variable names
- Add comprehensive doc comments for kernels
- Include complexity analysis and memory access patterns
- Use `const` and `constant` qualifiers where appropriate

### Memory Access
- Prefer coalesced memory access (consecutive threads → consecutive addresses)
- Use threadgroup memory for data reuse (16KB L1 cache on Apple Silicon)
- Align all data structures to 16 bytes

### SIMD Optimization
- Use `simd_sum()` for reductions (faster than manual loops)
- Process data in float4 chunks when possible
- Avoid divergent branches within SIMD groups

### Numerical Stability
- Always use epsilon for comparisons and divisions
- Use FMA (`fma(a, b, c)`) instead of `a * b + c`
- Consider overflow/underflow for extreme values

## Parameter Struct Alignment

**Critical**: Metal structs must exactly match Swift structs.

```metal
// Metal (MetalCommon.h)
struct PoolingParams {
    int32_t sequenceLength;
    int32_t dimensions;
    int32_t _padding0;  // Required!
    int32_t _padding1;
};
```

```swift
// Swift (MetalShaderLibrary.swift)
@frozen
public struct PoolingParams {
    public let sequenceLength: Int32
    public let dimensions: Int32
    private let _padding0: Int32 = 0
    private let _padding1: Int32 = 0
}
```

**Verification**:
```metal
static_assert(sizeof(PoolingParams) == 16, "Must be 16 bytes");
```

## Performance Guidelines

### Occupancy
- Target 75-100% occupancy for compute-bound kernels
- Balance threadgroup size vs register usage
- Use Xcode GPU profiler to check actual occupancy

### Memory Bandwidth
- Aim for >70% memory bandwidth utilization
- Use vectorized loads/stores (float4) when possible
- Minimize DRAM traffic through cache-friendly access

### Common Optimizations
1. **Loop Unrolling**: Process 4 elements at a time
2. **FMA Usage**: Combine multiply + add operations
3. **SIMD Reductions**: Use `simd_sum()` instead of manual reduction
4. **Coalesced Access**: Ensure consecutive threads access consecutive memory

## Troubleshooting

### Compilation Errors

**Error**: `no such file or directory: ../Common/MetalCommon.h`
**Solution**: Compile from `Shaders/` directory with proper include path

**Error**: `struct size mismatch`
**Solution**: Ensure Swift and Metal structs have identical padding

**Error**: `Metal 3.0 not supported`
**Solution**: Requires Xcode 14+, macOS 13+, or iOS 16+

### Sign Comparison Warnings

Metal compiler may warn about signed/unsigned comparisons:
```metal
if (gid >= params.dimensions)  // Warning: uint vs int32_t
```

These warnings are benign but can be silenced:
```metal
if (gid >= uint(params.dimensions))  // Explicit cast
```

### Runtime Issues

**Symptom**: Kernel produces incorrect results
**Debug**:
1. Enable Metal API validation in Xcode scheme
2. Use Metal Frame Capture to inspect GPU state
3. Add temporary `printf()` in shader (Metal 3.1+)
4. Verify buffer sizes match expectations

**Symptom**: Performance regression
**Debug**:
1. Profile with Xcode Metal Debugger
2. Check occupancy (should be >75%)
3. Verify memory access patterns (coalesced?)
4. Look for unexpected synchronization

## Testing Checklist

Before committing shader changes:
- [ ] Manual compilation succeeds without errors
- [ ] All kernels present in metallib (verify with `metal-nm`)
- [ ] Swift tests pass (no regressions)
- [ ] Metal validation enabled, no warnings
- [ ] Benchmarks show no performance regression (±5% tolerance)
- [ ] Tested on M1/M2/M3 (if available)
- [ ] Documentation updated

## Useful Commands Reference

```bash
# Check Metal version
xcrun metal --version

# Compile with warnings
xcrun metal -std=metal3.0 -Wall -Wextra -c input.metal

# Optimize compilation
xcrun metal -std=metal3.0 -O3 -ffast-math -c input.metal

# List metallib contents
xcrun metal-nm /path/to/lib.metallib

# Disassemble shader
xcrun metal-objdump --disassemble /path/to/lib.metallib

# Validate Metal code
xcrun metal -std=metal3.0 -Werror -c input.metal
```

## Resources

### Apple Documentation
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Best Practices Guide](https://developer.apple.com/documentation/metal/best_practices)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)

### Tools
- **Xcode Metal Debugger**: Capture GPU frames and analyze
- **Instruments Metal System Trace**: Track GPU/CPU interaction
- **Metal Validation Layer**: Enable in scheme for runtime checks

### Books
- "Metal by Example" by Warren Moore
- "Metal Programming Guide" by Janie Clayton

## Next Steps (Phase 1 Continuation)

**Current Status**: ✅ Step 1 Complete (Manual Compilation Validated)

**Next Steps**:
- **Step 2**: Set up SPM build script for automatic compilation
- **Step 3**: Create Swift API for metallib loading (with string fallback)
- **Step 4**: Add Xcode build phase support
- **Step 5**: Comprehensive testing and validation
- **Step 6**: Remove string-based fallback (production-ready)

See `Documentation/XcodeIntegration.md` for Xcode build integration steps.

---

**Last Updated**: 2025-10-23
**Metal Version**: 3.0+
**Compatibility**: iOS 16+ / macOS 13+
