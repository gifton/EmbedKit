# Unified Memory Path: CoreML → Metal Zero-Copy Pipeline

## Agent Specification

**Objective:** Eliminate the double-copy in the CoreML output → Metal GPU pooling pipeline by leveraging Apple Silicon's unified memory architecture.

---

## Current Pipeline (2 copies)

```
CoreML MLMultiArray (unified memory, page-aligned on Apple Silicon)
  │
  ├─ COPY 1: flattenFloatArray() → Array(UnsafeBufferPointer(...))
  │   Creates a [Float] on the Swift heap via memcpy
  │   File: Sources/EmbedKit/Backends/CoreMLBackend.swift:624-640
  │
  ├─ Returns CoreMLOutput { values: [Float], shape: [Int] }
  │   Crosses actor boundary: CoreMLBackend → AppleEmbeddingModel
  │
  ├─ COPY 2: MTLDevice.makeBuffer(bytes: out.values, ...)
  │   Copies [Float] into a new MTLBuffer for GPU kernel dispatch
  │   File: Sources/EmbedKit/Acceleration/MetalAccelerator.swift:790+
  │
  └─ Metal kernel operates on MTLBuffer
```

## Target Pipeline (0 copies)

```
CoreML MLMultiArray (unified memory)
  │
  ├─ MTLDevice.makeBuffer(bytesNoCopy: array.dataPointer, ...)
  │   Wraps the SAME physical memory as an MTLBuffer (no copy)
  │
  └─ Metal kernel operates directly on CoreML's output buffer
```

---

## Implementation Tasks

### 1. Create `CoreMLOutputRef` type

**File:** `Sources/EmbedKit/Backends/CoreMLBackend.swift`

Create a new type that holds the `MLMultiArray` alive and exposes its raw pointer for direct Metal consumption:

```swift
/// Holds a reference to a CoreML output MLMultiArray, keeping it alive
/// so that its data pointer can be used directly by Metal without copying.
public final class CoreMLOutputRef: @unchecked Sendable {
    /// The underlying MLMultiArray — kept alive for pointer validity.
    private let array: MLMultiArray

    /// Shape of the output tensor (e.g., [1, seq, dim] or [seq, dim]).
    public let shape: [Int]

    /// Number of float elements in the tensor.
    public let count: Int

    /// Raw pointer to the float data. Valid as long as this object is alive.
    /// - Important: Only valid when `dataType == .float32`.
    public var dataPointer: UnsafeRawPointer {
        array.dataPointer
    }

    /// Whether the data pointer is page-aligned (required for makeBuffer(bytesNoCopy:)).
    public var isPageAligned: Bool {
        Int(bitPattern: array.dataPointer) % Int(vm_page_size) == 0
    }

    /// Byte count of the data buffer.
    public var byteCount: Int {
        count * MemoryLayout<Float>.stride
    }

    /// The data type of the underlying array.
    public var dataType: MLMultiArrayDataType {
        array.dataType
    }

    public init(array: MLMultiArray) {
        self.array = array
        self.shape = array.shape.map { $0.intValue }
        self.count = array.count
    }
}
```

### 2. Add zero-copy buffer wrapping to `MetalAccelerator`

**File:** `Sources/EmbedKit/Acceleration/MetalAccelerator.swift`

Add a method that wraps an existing pointer as an `MTLBuffer` without copying:

```swift
/// Wraps an external pointer as an MTLBuffer without copying.
/// Falls back to a copy if the pointer is not page-aligned.
///
/// - Parameters:
///   - pointer: Raw pointer to wrap (must be valid for the lifetime of returned buffer usage)
///   - byteCount: Size in bytes
/// - Returns: An MTLBuffer backed by the same memory, or a copy if alignment fails
private func wrapOrCopyBuffer(
    pointer: UnsafeRawPointer,
    byteCount: Int
) -> MTLBuffer? {
    guard let dev = device else { return nil }

    // Page-aligned → zero-copy wrap
    if Int(bitPattern: pointer) % Int(vm_page_size) == 0 {
        // Round byte count up to page size for Metal requirements
        let alignedBytes = (byteCount + Int(vm_page_size) - 1) & ~(Int(vm_page_size) - 1)
        return dev.makeBuffer(
            bytesNoCopy: UnsafeMutableRawPointer(mutating: pointer),
            length: alignedBytes,
            options: .storageModeShared,
            deallocator: nil  // We do NOT own this memory — CoreMLOutputRef does
        )
    }

    // Not page-aligned → fall back to copy
    return dev.makeBuffer(bytes: pointer, length: byteCount, options: .storageModeShared)
}
```

Then add overloads to the pooling methods that accept `CoreMLOutputRef`:

```swift
/// Mean-pool directly from a CoreML output reference (zero-copy when possible).
public func meanPool(
    outputRef: CoreMLOutputRef,
    sequenceLength: Int,
    dimensions: Int,
    mask: [Int]? = nil
) async -> [Float] {
    guard outputRef.dataType == .float32 else {
        // Fall back to the [Float] path for non-float32
        // (caller should have already converted)
        return meanPool(
            embeddings: Array(UnsafeBufferPointer(
                start: UnsafePointer<Float>(OpaquePointer(outputRef.dataPointer)),
                count: outputRef.count
            )),
            sequenceLength: sequenceLength,
            dimensions: dimensions,
            mask: mask
        )
    }

    guard let buffer = wrapOrCopyBuffer(
        pointer: outputRef.dataPointer,
        byteCount: outputRef.byteCount
    ) else {
        // GPU unavailable — CPU fallback
        let floats = Array(UnsafeBufferPointer(
            start: UnsafePointer<Float>(OpaquePointer(outputRef.dataPointer)),
            count: outputRef.count
        ))
        return AccelerateBLAS.meanPool(sequence: floats, tokens: sequenceLength, dim: dimensions, mask: mask)
    }

    // ... dispatch Metal kernel using `buffer` directly ...
}
```

### 3. Add batch `tensorPoolNormalize` with `CoreMLOutputRef` array

**File:** `Sources/EmbedKit/Acceleration/MetalAccelerator.swift`

Add an overload of `tensorPoolNormalize` that accepts `[CoreMLOutputRef]` instead of `[Float]`:

```swift
public func tensorPoolNormalize(
    outputRefs: [CoreMLOutputRef],
    batchSize: Int,
    sequenceLength: Int,
    dimensions: Int,
    mask: [[Int]]? = nil,
    strategy: PoolingStrategy = .mean,
    normalize: Bool = true
) async -> [[Float]] {
    // For each ref, wrap as MTLBuffer (zero-copy if aligned)
    // Pack into a single batched MTLBuffer or dispatch per-item
    // ...
}
```

### 4. Wire up in `AppleEmbeddingModel`

**File:** `Sources/EmbedKit/Models/AppleEmbeddingModel.swift`

Modify the `process()` return path in `CoreMLBackend` to optionally return `CoreMLOutputRef` alongside the existing `CoreMLOutput`:

```swift
public struct CoreMLResult: Sendable {
    /// The flattened float values (always available, backward compat)
    public let output: CoreMLOutput
    /// Optional reference to the raw MLMultiArray for zero-copy GPU access
    public let ref: CoreMLOutputRef?
}
```

In `AppleEmbeddingModel.embed()`:
```swift
let result = try await backend.process(input)
let out = result.output

// When GPU pooling is enabled and we have a ref, use zero-copy path
if useGPUPooling, let ref = result.ref, let acc = await ensureMetal() {
    pooled = await acc.meanPool(outputRef: ref, sequenceLength: tokens, dimensions: dim, mask: ...)
} else {
    pooled = PoolingHelpers.mean(sequence: out.values, tokens: tokens, dim: dim, mask: ...)
}
```

---

## Constraints & Safety

### Page Alignment
- `makeBuffer(bytesNoCopy:)` requires the pointer to be page-aligned (typically 16KB on Apple Silicon)
- CoreML's internal allocator generally provides page-aligned buffers, but there is **no API guarantee**
- **Must check at runtime:** `Int(bitPattern: ptr) % vm_page_size == 0`
- Fall back to `makeBuffer(bytes:)` (copy) if not aligned

### Lifetime Management
- `CoreMLOutputRef` must be kept alive for the entire duration of:
  1. `MTLBuffer` creation (synchronous, OK)
  2. Metal command buffer encoding (synchronous, OK)
  3. Metal command buffer execution (asynchronous — **critical**)
- Use `withExtendedLifetime(ref) { ... }` or store `ref` in a local until `commandBuffer.waitUntilCompleted()`
- Alternatively, capture `ref` in the completion handler

### Sendability
- `CoreMLOutputRef` is `@unchecked Sendable` because `MLMultiArray` is an ObjC object (thread-safe for reads)
- The `dataPointer` itself is just a raw pointer — valid for reads from any thread as long as the array is alive
- Do NOT pass raw `UnsafePointer`/`UnsafeBufferPointer` across actor boundaries — always pass `CoreMLOutputRef`

### Data Type
- This zero-copy path only works for `float32` output arrays
- For `double` output, the conversion to `Float` inherently requires a copy (use the existing `vDSP_vdpsp` path)
- Check `outputRef.dataType == .float32` before using the zero-copy path

---

## Verification

### Correctness
- Output must be **bit-identical** to the copy-based path
- Test with the existing test suite (1,431+ tests) — no regressions
- Add targeted tests comparing zero-copy vs copy output for float32 and double data types

### Performance
- Benchmark `embed()` latency: single text, MiniLM-L12-v2 (384 dim, ~128 tokens)
- Benchmark `embedBatch()` latency: 32 texts, same model
- Measure with Instruments:
  - Memory allocations: should see fewer heap allocations in the zero-copy path
  - Metal GPU trace: verify the `MTLBuffer` is backed by shared memory (no GPU copy)
- Report: % reduction in latency and peak memory for both single and batch paths

### Edge Cases
- Non-page-aligned pointer (verify fallback path works)
- Very small outputs (1 token, 1 dim) — should still work
- Very large outputs (4096 tokens, 1024 dim) — verify no issues
- Concurrent embedding requests — verify no data races on shared buffers
- Model output that is double instead of float32 — verify graceful fallback

---

## Files to Modify

| File | Change |
|------|--------|
| `Sources/EmbedKit/Backends/CoreMLBackend.swift` | Add `CoreMLOutputRef`, `CoreMLResult`, modify `process()`/`processBatch()` to return refs |
| `Sources/EmbedKit/Acceleration/MetalAccelerator.swift` | Add `wrapOrCopyBuffer()`, add `outputRef:` overloads for pooling methods |
| `Sources/EmbedKit/Models/AppleEmbeddingModel.swift` | Wire up zero-copy path in `embed()` and `embedBatch()` |

## Dependencies
- Task 2.1 (zero-copy flattenFloatArray) — already completed
- Task 2.2 (pointer-based pooling overloads) — already completed
- These provide the CPU fallback paths that this task builds upon
