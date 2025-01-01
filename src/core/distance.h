#pragma once

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

extern "C" {
    // Computes squared L2 distance between two vectors of size dim
    EXPORT float dist_l2_sq(const float* a, const float* b, int dim);
    
    // Computes dot product between two vectors of size dim
    EXPORT float dist_dot(const float* a, const float* b, int dim);
    
    // Computes cosine similarity (assumes pre-normalized or handles internally if needed)
    // For raw vectors, cosine = dot(a, b) / (norm(a) * norm(b))
    // We will stick to dot product for normalized vectors usually, but provide this.
    EXPORT float dist_cosine(const float* a, const float* b, int dim);
}




