#pragma once
#include <vector>
#include <cstdint>

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

extern "C" {
    struct IVFPQIndex;
    
    // Create IVF-PQ index
    // dim: dimension of vectors
    // n_centroids: number of coarse centroids (IVF lists)
    // m_subquantizers: number of sub-vectors for PQ
    // n_bits: bits per sub-vector (usually 8 for 256 centroids)
    EXPORT IVFPQIndex* ivf_pq_create(int dim, int n_centroids, int m_subquantizers, int n_bits);
    
    EXPORT void ivf_pq_destroy(IVFPQIndex* index);
    
    // Train the index (K-means for IVF centroids + PQ codebooks)
    // Requires a representative set of vectors
    EXPORT void ivf_pq_train(IVFPQIndex* index, int n_samples, const float* samples);
    
    // Insert vector (assigns to nearest centroid, computes residual, quantizes residual)
    EXPORT void ivf_pq_insert(IVFPQIndex* index, long long id, const float* vector);
    
    // Search
    EXPORT int ivf_pq_search(IVFPQIndex* index, const float* query, int k, int n_probes, long long* out_ids, float* out_dists);

    EXPORT int ivf_pq_save(IVFPQIndex* index, const char* filename);
    EXPORT IVFPQIndex* ivf_pq_load(const char* filename);
}










