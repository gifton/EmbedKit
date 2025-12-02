# MSL 4.0 API Clarifications

**Source**: Actual Metal Performance Primitives headers from Xcode 16 SDK
**Path**: `MetalPerformancePrimitives.framework/Headers/MPPTensorOpsMatMul2d.h`

---

## 1. Tensor Binding Model

**Answer: Option A - Bind MTLTensor using `tensor_handle`**

```metal
// Correct binding syntax from Apple headers:
kernel void myKernel(
    tensor<device half, dextents<int32_t, 2>, tensor_handle> A [[buffer(0)]],
    tensor<device half, dextents<int32_t, 2>, tensor_handle> B [[buffer(1)]],
    tensor<device float, dextents<int32_t, 2>, tensor_handle> C [[buffer(2)]],
    ...
)
```

**Tensor descriptor types available**:
- `tensor_handle` - For binding MTLTensor resources from host
- `tensor_offset` - For creating offset views (shifted origin)
- `tensor_inline` - For thread-local tensor views (tightly packed, no strides needed)

**Creating inline tensors**:
```metal
auto inputTensor = tensor(inputs, extents<int, INPUT_WIDTH, 1>());
// Inline tensors are assumed tightly packed, no strides needed
```

---

## 2. matmul2d Signature and Descriptor

**Exact descriptor definition** (from `MPPTensorOpsMatMul2d.h` lines 372-400):

```metal
namespace mpp {
namespace tensor_ops {

struct matmul2d_descriptor {
    enum class mode {
        multiply,
        multiply_accumulate,
    };

    int m, n, k;
    bool transpose_left, transpose_right;
    bool relaxed_precision;
    mode matmul_mode;

    constexpr matmul2d_descriptor(
        int __m,                                    // Output rows (tile size)
        int __n,                                    // Output cols (tile size)
        int __k = dynamic_length_v<int>,            // Reduction dim (0 = read from tensor)
        bool __transpose_left = false,              // Transpose A
        bool __transpose_right = false,             // Transpose B
        bool __relaxed_precision = false,           // Allow precision trade-off
        mode __matmul_mode = mode::multiply         // multiply vs multiply_accumulate
    );
};

}}
```

**Usage for transposed multiply (Q × K^T)**:

```metal
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp;
using namespace mpp::tensor_ops;

// For similarity: C[Q,K] = Queries[Q,D] × Keys[K,D]^T
// This is Queries × transpose(Keys)
constexpr auto desc = matmul2d_descriptor(
    64,     // m: tile rows (output rows per threadgroup)
    32,     // n: tile cols (output cols per threadgroup)
    0,      // k: dynamic (read from tensor extent)
    false,  // transpose_left: Queries not transposed
    true,   // transpose_right: Keys ARE transposed ← KEY FOR SIMILARITY
    true    // relaxed_precision: OK for embeddings
);

// Create operation with execution scope
matmul2d<desc, opscope_simdgroups<4>> matmulOp;

// Execute
matmulOp.run(queriesTensor, keysTensor, outputTensor);
```

**Execution scopes**:
- `metal::execution_thread` - Single thread (fragment shaders only)
- `metal::execution_simdgroup` - One SIMD group cooperatively
- `opscope_simdgroups<N>` - N SIMD groups cooperatively (for compute kernels)

---

## 3. Reduction Operations Signatures

**From header lines 503-525**:

```metal
namespace mpp {
namespace tensor_ops {

enum class reduction_operation {
    sum,
    max,
    min,
};

template <class ElementType, class SrcExtents, class DstExtents, class SrcLayout, class DstLayout>
inline void reduce_rows(
    thread metal::cooperative_tensor<ElementType, SrcExtents, SrcLayout>& source,
    thread metal::cooperative_tensor<ElementType, DstExtents, DstLayout>& destination,
    reduction_operation op = reduction_operation::sum,
    ElementType identity = reduction_operation_identity<ElementType>::sum_identity
);

template <class ElementType, class SrcExtents, class DstExtents, class SrcLayout, class DstLayout>
inline void reduce_columns(
    thread metal::cooperative_tensor<ElementType, SrcExtents, SrcLayout>& source,
    thread metal::cooperative_tensor<ElementType, DstExtents, DstLayout>& destination,
    reduction_operation op = reduction_operation::sum,
    ElementType identity = reduction_operation_identity<ElementType>::sum_identity
);

}}
```

**Important**: These work on `cooperative_tensor`, not on `tensor_handle` directly!

**Reduction identities** (lines 403-409):
```metal
template <typename ElementType>
struct reduction_operation_identity {
    static const constant ElementType sum_identity = (ElementType)0;
    static const constant ElementType max_identity = metal::numeric_limits<ElementType>::lowest;
    static const constant ElementType min_identity = metal::numeric_limits<ElementType>::max;
};
```

**Usage pattern**:
```metal
// Get cooperative tensor from matmul operation
auto coopTensor = matmulOp.get_destination_cooperative_tensor<...>();

// For row reduction destination
auto rowReduced = matmulOp.get_row_reduction_destination_cooperative_tensor<...>();

// Perform reduction
reduce_rows(coopTensor, rowReduced, reduction_operation::sum);
// or
reduce_rows(coopTensor, rowReduced, reduction_operation::max,
            reduction_operation_identity<float>::max_identity);
```

---

## 4. Masking Strategy for Pooling

**Answer**: MSL 4.0 does NOT provide predicated reduction or tensor-level `select`/`where`.

**Recommended approach**:
1. Use element-wise operations on cooperative tensors
2. Apply mask before reduction

```metal
// Get cooperative tensor
auto cT = matmulOp.get_destination_cooperative_tensor<...>();

// Apply mask element-wise (using iterator pattern from header lines 527-549)
#pragma unroll full
for (uint16_t i = 0; i < cT.capacity(); ++i) {
    if (cT.mask(i)) {  // Check if this element is valid for this thread
        auto ids = cT.multidimensional_indices(i);  // Get [row, col] indices
        int tokenIdx = ids[0];  // Assuming row = token

        // Apply attention mask
        if (maskBuffer[batchOffset + tokenIdx] == 0) {
            cT[i] = -INFINITY;  // For max pooling
            // or: cT[i] = 0.0f;  // For sum pooling
        }
    }
}

// Then reduce
reduce_rows(cT, reducedCT, reduction_operation::max);
```

---

## 5. Tensor Slicing and Broadcasting

**Slicing operations** (from header examples lines 100-154):

```metal
// offset() - Creates tensor_offset view with shifted origin
// mA[x,y] == A[x, origin_y + y]
auto mA = A.offset(0, tgid.y * 64);
auto mB = B.offset(tgid.x * 32, 0);

// static_slice<extent0, extent1>() - Creates fixed-size view
// Enables bounds-check elimination for "inside" tiles
auto tA = A.static_slice<dynamic_extent, 64>(k, tgid.y * 64);
auto tB = B.static_slice<32, dynamic_extent>(tgid.x * 32, k);
auto tC = C.static_slice<32, 64>(tgid.x * 32, tgid.y * 64);

// dynamic_extent means "use tensor's actual extent for this dimension"
```

**Broadcasting**: Not directly supported via operator syntax. Use cooperative tensors with explicit loops:

```metal
// For attention pooling: input[B,S,D] * weights[B,S] -> weighted[B,S,D]
// Must handle manually:

#pragma unroll full
for (uint16_t i = 0; i < cT.capacity(); ++i) {
    if (cT.mask(i)) {
        auto ids = cT.multidimensional_indices(i);
        int batchIdx = ids[0];
        int tokenIdx = ids[1];
        int dimIdx = ids[2];

        float weight = weights[batchIdx * seqLen + tokenIdx];
        cT[i] *= weight;  // Apply weight
    }
}
```

---

## 6. Cooperative Tensor Patterns

**Key patterns from header** (lines 234-311):

```metal
// Create cooperative tensor for matmul output
auto cT = matmulOp.get_destination_cooperative_tensor<
    decltype(mA),
    decltype(mB),
    float  // Output element type
>();

// Initialize (MUST use #pragma unroll full for performance)
#pragma unroll full
for (uint16_t i = 0; i < cT.capacity(); ++i) {
    if (cT.mask(i))
        cT[i] = 0.0f;
}

// Execute matmul into cooperative tensor
matmulOp.run(mA, mB, cT);

// Post-process (activation, bias, etc.)
#pragma unroll full
for (uint16_t i = 0; i < cT.capacity(); ++i) {
    if (cT.mask(i)) {
        cT[i] = relu(cT[i]);  // Apply activation
    }
}

// Store to device memory
cT.store(mC);

// Or load from tensor into cooperative tensor
auto biasCT = matmulOp.get_destination_cooperative_tensor<...>();
biasCT.load(biasTensorHandle);
```

---

## 7. Complete Example: Similarity Matrix

```metal
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp;
using namespace mpp::tensor_ops;

kernel void tensor_similarity_matrix_v2(
    tensor<device float, dextents<int32_t, 2>, tensor_handle> queries [[buffer(0)]],  // [Q, D]
    tensor<device float, dextents<int32_t, 2>, tensor_handle> keys [[buffer(1)]],     // [K, D]
    tensor<device float, dextents<int32_t, 2>, tensor_handle> output [[buffer(2)]],   // [Q, K]
    constant TensorSimilarityParams& params [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    // Tile sizes for this threadgroup
    constexpr int TILE_Q = 64;  // Queries per threadgroup
    constexpr int TILE_K = 32;  // Keys per threadgroup

    // Descriptor: C[Q,K] = Q[Q,D] × K[K,D]^T
    // transpose_right = true because we're doing Q × K^T
    constexpr auto desc = matmul2d_descriptor(
        TILE_Q,  // m
        TILE_K,  // n
        0,       // k (dynamic)
        false,   // transpose_left
        true,    // transpose_right ← For Q × K^T
        true     // relaxed_precision
    );

    matmul2d<desc, opscope_simdgroups<4>> matmulOp;

    // Create slices for this threadgroup
    int qStart = tgid.y * TILE_Q;
    int kStart = tgid.x * TILE_K;

    auto tQ = queries.offset(0, qStart);  // Slice of queries
    auto tK = keys.offset(0, kStart);     // Slice of keys (will be transposed by matmul)
    auto tO = output.offset(kStart, qStart);

    // For normalized inputs, similarity = dot product
    // matmul handles the transpose internally
    matmulOp.run(tQ, tK, tO);
}
```

---

## 8. Header Includes Required

```metal
// For tensor operations
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// Standard Metal
#include <metal_stdlib>

using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;
```

---

## 9. Compilation Requirements

The tensor operations require:
```metal
#if defined(__METAL_VERSION__) && defined(__HAVE_TENSOR__)
// Tensor code here
#endif
```

This is automatically defined on Metal 4.0+ devices.

---

## Summary: Answers to Agent Questions

| Question | Answer |
|----------|--------|
| 1. Tensor binding model | **Option A**: Bind `MTLTensor` using `tensor_handle` |
| 2. matmul2d transpose | Use `transpose_right = true` in descriptor |
| 3. Reduction signatures | `reduce_rows(cooperative_src, cooperative_dst, op, identity)` |
| 4. Masking strategy | Manual element-wise in cooperative tensor loop |
| 5. Slicing syntax | `tensor.offset(x, y)` and `tensor.static_slice<E0, E1>(x, y)` |

**Recommended starting point**: Similarity Matrix kernel using `matmul2d` with `transpose_right = true`.

---

## Sources

- [WWDC25: Combine Metal 4 ML and Graphics](https://developer.apple.com/videos/play/wwdc2025/262/)
- [Metal Shading Language Specification 4.0](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- Xcode 16 SDK: `MetalPerformancePrimitives.framework/Headers/MPPTensorOpsMatMul2d.h`
