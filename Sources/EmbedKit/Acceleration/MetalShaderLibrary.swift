import Foundation

/// Centralized Metal shader library containing all GPU kernels for embedding operations
///
/// This struct provides a single source of truth for all Metal shaders used in EmbedKit,
/// making shader management easier and more maintainable.
public struct MetalShaderLibrary {
    
    /// Names of available compute kernels
    public enum KernelName: String, CaseIterable {
        case l2Normalize = "l2_normalize"
        case meanPool = "mean_pool"
        case maxPool = "max_pool"
        case cosineSimilarity = "cosine_similarity"
        case attentionWeightedPool = "attention_weighted_pool"
    }
    
    /// Complete Metal shader source code with Metal 3 optimizations
    public static let source = """
    #include <metal_stdlib>
    using namespace metal;
    
    // Metal 3 optimizations
    #pragma METAL internals : enable
    #pragma METAL fast_math enable
    
    // Use relaxed precision for better performance where appropriate
    using namespace metal::precise;
    
    // L2 normalization kernel with Metal 3 optimizations
    kernel void l2_normalize(device const float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant int32_t& dimensions [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]],
                            uint simd_lane_id [[thread_index_in_simdgroup]],
                            uint simd_size [[threads_per_simdgroup]]) {
        const uint vectorIndex = gid.y;
        const uint dimIndex = gid.x;
        
        // Early exit for out-of-bounds threads
        if (dimIndex >= dimensions) return;
        
        const uint baseIndex = vectorIndex * dimensions;
        
        // Improved L2 norm calculation using SIMD group operations
        float norm_squared = 0.0f;
        
        // Each thread in the SIMD group processes different elements
        for (uint i = simd_lane_id; i < dimensions; i += simd_size) {
            const float val = input[baseIndex + i];
            norm_squared += val * val;
        }
        
        // SIMD group reduction - more efficient than manual reduction
        norm_squared = simd_sum(norm_squared);
        
        // All threads now have the same norm_squared value
        // Use fast inverse square root approximation for better performance
        const float norm = precise::sqrt(norm_squared);
        const float inv_norm = (norm > 0.0f) ? (1.0f / norm) : 0.0f;
        
        // Write normalized value (coalesced memory access)
        output[baseIndex + dimIndex] = input[baseIndex + dimIndex] * inv_norm;
    }
    
    // Mean pooling kernel with optimizations
    kernel void mean_pool(device const float* input [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         device const int32_t* mask [[buffer(2)]],
                         constant PoolingParams& params [[buffer(3)]],
                         uint gid [[thread_position_in_grid]],
                         uint simd_lane_id [[thread_index_in_simdgroup]],
                         uint simd_size [[threads_per_simdgroup]]) {
        if (gid >= params.dimensions) return;
        
        float sum = 0.0f;
        int count = 0;
        
        // Unroll loop for better performance
        const int32_t seqLen = params.sequenceLength;
        const int32_t dim = params.dimensions;
        
        // Process 4 elements at a time when possible
        int i = 0;
        for (; i <= seqLen - 4; i += 4) {
            // Prefetch mask values
            const bool m0 = !mask || mask[i] == 1;
            const bool m1 = !mask || mask[i + 1] == 1;
            const bool m2 = !mask || mask[i + 2] == 1;
            const bool m3 = !mask || mask[i + 3] == 1;
            
            // Vectorized accumulation
            sum += m0 ? input[i * dim + gid] : 0.0f;
            sum += m1 ? input[(i + 1) * dim + gid] : 0.0f;
            sum += m2 ? input[(i + 2) * dim + gid] : 0.0f;
            sum += m3 ? input[(i + 3) * dim + gid] : 0.0f;
            
            count += m0 + m1 + m2 + m3;
        }
        
        // Handle remaining elements
        for (; i < seqLen; i++) {
            if (!mask || mask[i] == 1) {
                sum += input[i * dim + gid];
                count++;
            }
        }
        
        // Use reciprocal multiplication instead of division
        output[gid] = count > 0 ? sum * (1.0f / float(count)) : 0.0f;
    }
    
    // Max pooling kernel with optimizations
    kernel void max_pool(device const float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        device const int32_t* mask [[buffer(2)]],
                        constant PoolingParams& params [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
        if (gid >= params.dimensions) return;
        
        float maxVal = -INFINITY;
        bool foundValid = false;
        
        const int32_t seqLen = params.sequenceLength;
        const int32_t dim = params.dimensions;
        
        // Unroll loop by 4 for better performance
        int i = 0;
        for (; i <= seqLen - 4; i += 4) {
            // Check mask values
            const bool m0 = !mask || mask[i] == 1;
            const bool m1 = !mask || mask[i + 1] == 1;
            const bool m2 = !mask || mask[i + 2] == 1;
            const bool m3 = !mask || mask[i + 3] == 1;
            
            // Load values conditionally
            if (m0) {
                maxVal = max(maxVal, input[i * dim + gid]);
                foundValid = true;
            }
            if (m1) {
                maxVal = max(maxVal, input[(i + 1) * dim + gid]);
                foundValid = true;
            }
            if (m2) {
                maxVal = max(maxVal, input[(i + 2) * dim + gid]);
                foundValid = true;
            }
            if (m3) {
                maxVal = max(maxVal, input[(i + 3) * dim + gid]);
                foundValid = true;
            }
        }
        
        // Handle remaining elements
        for (; i < seqLen; i++) {
            if (!mask || mask[i] == 1) {
                maxVal = max(maxVal, input[i * dim + gid]);
                foundValid = true;
            }
        }
        
        output[gid] = foundValid ? maxVal : 0.0f;
    }
    
    // Cosine similarity kernel with optimizations
    kernel void cosine_similarity(device const float* queries [[buffer(0)]],
                                 device const float* keys [[buffer(1)]],
                                 device float* output [[buffer(2)]],
                                 constant SimilarityParams& params [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]]) {
        const uint queryIdx = gid.y;
        const uint keyIdx = gid.x;
        
        if (queryIdx >= params.queryCount || keyIdx >= params.keyCount) return;
        
        const uint queryOffset = queryIdx * params.dimensions;
        const uint keyOffset = keyIdx * params.dimensions;
        
        // Use float4 for vectorized operations when possible
        float dotProduct = 0.0f;
        float queryNorm = 0.0f;
        float keyNorm = 0.0f;
        
        const int32_t dims = params.dimensions;
        int i = 0;
        
        // Process 4 elements at a time using vector operations
        for (; i <= dims - 4; i += 4) {
            float4 q = float4(queries[queryOffset + i],
                             queries[queryOffset + i + 1],
                             queries[queryOffset + i + 2],
                             queries[queryOffset + i + 3]);
            
            float4 k = float4(keys[keyOffset + i],
                             keys[keyOffset + i + 1],
                             keys[keyOffset + i + 2],
                             keys[keyOffset + i + 3]);
            
            // Vectorized dot product and norms
            float4 qk = q * k;
            float4 qq = q * q;
            float4 kk = k * k;
            
            dotProduct += qk.x + qk.y + qk.z + qk.w;
            queryNorm += qq.x + qq.y + qq.z + qq.w;
            keyNorm += kk.x + kk.y + kk.z + kk.w;
        }
        
        // Handle remaining elements
        for (; i < dims; i++) {
            const float queryVal = queries[queryOffset + i];
            const float keyVal = keys[keyOffset + i];
            
            dotProduct += queryVal * keyVal;
            queryNorm += queryVal * queryVal;
            keyNorm += keyVal * keyVal;
        }
        
        // Use fast inverse square root for normalization
        const float invNormProduct = rsqrt(queryNorm * keyNorm);
        const float similarity = dotProduct * invNormProduct;
        
        // Clamp to valid cosine similarity range
        output[queryIdx * params.keyCount + keyIdx] = clamp(similarity, -1.0f, 1.0f);
    }
    
    // Attention-weighted pooling kernel with optimizations
    kernel void attention_weighted_pool(device const float* input [[buffer(0)]],
                                       device const float* weights [[buffer(1)]],
                                       device float* output [[buffer(2)]],
                                       constant PoolingParams& params [[buffer(3)]],
                                       uint gid [[thread_position_in_grid]]) {
        if (gid >= params.dimensions) return;
        
        float weightedSum = 0.0f;
        float weightSum = 0.0f;
        
        const int32_t seqLen = params.sequenceLength;
        const int32_t dim = params.dimensions;
        
        // Unroll by 4 for better performance
        int i = 0;
        for (; i <= seqLen - 4; i += 4) {
            // Load weights
            const float w0 = weights[i];
            const float w1 = weights[i + 1];
            const float w2 = weights[i + 2];
            const float w3 = weights[i + 3];
            
            // Load input values
            const float v0 = input[i * dim + gid];
            const float v1 = input[(i + 1) * dim + gid];
            const float v2 = input[(i + 2) * dim + gid];
            const float v3 = input[(i + 3) * dim + gid];
            
            // Accumulate weighted sum
            weightedSum += v0 * w0 + v1 * w1 + v2 * w2 + v3 * w3;
            weightSum += w0 + w1 + w2 + w3;
        }
        
        // Handle remaining elements
        for (; i < seqLen; i++) {
            const float weight = weights[i];
            weightedSum += input[i * dim + gid] * weight;
            weightSum += weight;
        }
        
        // Use reciprocal for division
        const float invWeightSum = (weightSum > 0.0f) ? (1.0f / weightSum) : 0.0f;
        output[gid] = weightedSum * invWeightSum;
    }
    
    struct PoolingParams {
        int32_t sequenceLength;
        int32_t dimensions;
    };
    
    struct SimilarityParams {
        int32_t queryCount;
        int32_t keyCount;
        int32_t dimensions;
    };
    """
}

// MARK: - Supporting Parameter Structures

/// Parameters for pooling operations
public struct PoolingParams {
    let sequenceLength: Int32
    let dimensions: Int32
}

/// Parameters for similarity calculations
public struct SimilarityParams {
    let queryCount: Int32
    let keyCount: Int32
    let dimensions: Int32
}