#include "distance.h"
#include <cmath>
#include <immintrin.h>

// Fallback for non-AVX systems or tail processing
static float dist_l2_sq_scalar(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

static float dist_dot_scalar(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// AVX2 Implementation (DISABLED FOR STABILITY)
float dist_l2_sq(const float* a, const float* b, int dim) {
    return dist_l2_sq_scalar(a, b, dim);
    
    /*
    // Check alignment or just loadu (unaligned load is fine on modern CPUs)
    // We process 8 floats at a time (256 bits)
    
    int i = 0;
    __m256 sum = _mm256_setzero_ps();
    
    for (; i <= dim - 8; i += 8) {
        __m256 v1 = _mm256_loadu_ps(a + i);
        __m256 v2 = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(v1, v2);
        __m256 sq = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, sq);
    }
    
    // Horizontal sum of the 8 floats in 'sum'
    // There are faster ways, but this is readable
    float* f = (float*)&sum;
    float total = 0.0f;
    for(int k=0; k<8; ++k) total += f[k];
    
    // Handle remaining elements
    total += dist_l2_sq_scalar(a + i, b + i, dim - i);
    
    return total;
    */
}

float dist_dot(const float* a, const float* b, int dim) {
    int i = 0;
    __m256 sum = _mm256_setzero_ps();
    
    for (; i <= dim - 8; i += 8) {
        __m256 v1 = _mm256_loadu_ps(a + i);
        __m256 v2 = _mm256_loadu_ps(b + i);
        __m256 prod = _mm256_mul_ps(v1, v2);
        sum = _mm256_add_ps(sum, prod);
    }
    
    float* f = (float*)&sum;
    float total = 0.0f;
    for(int k=0; k<8; ++k) total += f[k];
    
    total += dist_dot_scalar(a + i, b + i, dim - i);
    
    return total;
}

float dist_cosine(const float* a, const float* b, int dim) {
    // Cosine similarity = dot(a, b) / (norm(a) * norm(b))
    // This is expensive to compute every time if vectors aren't normalized.
    // VectorForge assumes normalized vectors for 'cosine' metric usually,
    // but here is the raw implementation.
    
    float dot = dist_dot(a, b, dim);
    float norm_a = std::sqrt(dist_dot(a, a, dim));
    float norm_b = std::sqrt(dist_dot(b, b, dim));
    
    if (norm_a == 0 || norm_b == 0) return 0.0f;
    return dot / (norm_a * norm_b);
}






