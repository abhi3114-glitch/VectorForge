#include "ivf_pq.h"
#include "distance.h"
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <cstring>
#include <iostream>
#include <cstdio>
#include <mutex>

// Constants
static const int MAX_ITER = 20;

struct KMeans {
    int k;
    int dim;
    std::vector<float> centroids; // k * dim
    
    KMeans(int _k, int _dim) : k(_k), dim(_dim) {
        centroids.resize(k * dim);
    }
    
    void train(const float* data, int n) {
        if (n < k) return; // Not enough data
        
        // Init centroids randomly from data
        // Better: KMeans++ (omitted for brevity, using random sample)
        std::vector<int> perm(n);
        for(int i=0; i<n; ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), std::mt19937{std::random_device{}()});
        
        for(int i=0; i<k; ++i) {
            std::memcpy(&centroids[i*dim], &data[perm[i]*dim], dim * sizeof(float));
        }
        
        std::vector<int> counts(k);
        std::vector<float> new_centroids(k * dim);
        
        for(int iter=0; iter<MAX_ITER; ++iter) {
            std::fill(counts.begin(), counts.end(), 0);
            std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
            
            // Assign
            bool changed = false;
            // TODO: Parallelize
            for(int i=0; i<n; ++i) {
                const float* vec = &data[i*dim];
                int best_c = -1;
                float best_dist = std::numeric_limits<float>::max();
                
                // Find nearest centroid
                // Optimization: Use separate function/SIMD
                for(int c=0; c<k; ++c) {
                    float d = dist_l2_sq(vec, &centroids[c*dim], dim);
                    if (d < best_dist) {
                        best_dist = d;
                        best_c = c;
                    }
                }
                
                if (best_c != -1) {
                    counts[best_c]++;
                    for(int d=0; d<dim; ++d) new_centroids[best_c*dim + d] += vec[d];
                }
            }
            
            // Update
            for(int c=0; c<k; ++c) {
                if (counts[c] > 0) {
                    float inv = 1.0f / counts[c];
                    for(int d=0; d<dim; ++d) new_centroids[c*dim + d] *= inv;
                } else {
                    // Re-init empty cluster? Keep old?
                    // Keep old for now or pick random
                }
            }
            
            // Check convergence (simple check: if centroids didn't move much)
            // For now just fixed iters
            centroids = new_centroids;
        }
    }
    
    int closest(const float* vec) const {
        int best_c = -1;
        float best_dist = std::numeric_limits<float>::max();
        for(int c=0; c<k; ++c) {
            float d = dist_l2_sq(vec, &centroids[c*dim], dim);
            if (d < best_dist) {
                best_dist = d;
                best_c = c;
            }
        }
        return best_c;
    }
};

struct ProductQuantizer {
    int m; // number of subquantizers
    int dsub; // dimension of each subvector
    int ksub; // centroids per subquantizer (2^nbits, usually 256)
    
    // Codebooks: m * ksub * dsub
    std::vector<float> codebooks;
    
    ProductQuantizer(int dim, int _m, int nbits=8) : m(_m) {
        dsub = dim / m;
        ksub = 1 << nbits;
        codebooks.resize(m * ksub * dsub);
    }
    
    void train(const float* data, int n) {
        // Train m k-means solvers
        std::vector<float> sub_data(n * dsub);
        
        for(int i=0; i<m; ++i) {
            // Extract sub-vectors
            for(int j=0; j<n; ++j) {
                const float* src = data + j * (m * dsub) + i * dsub;
                std::memcpy(&sub_data[j*dsub], src, dsub * sizeof(float));
            }
            
            KMeans km(ksub, dsub);
            km.train(sub_data.data(), n);
            
            // Store centroids
            std::memcpy(&codebooks[i * ksub * dsub], km.centroids.data(), ksub * dsub * sizeof(float));
        }
    }
    
    // Encode vector to m bytes (indices)
    void encode(const float* vec, uint8_t* codes) {
        for(int i=0; i<m; ++i) {
            const float* sub_vec = vec + i * dsub;
            const float* book = &codebooks[i * ksub * dsub];
            
            int best_k = -1;
            float best_d = std::numeric_limits<float>::max();
            
            for(int k=0; k<ksub; ++k) {
                float d = dist_l2_sq(sub_vec, book + k*dsub, dsub);
                if (d < best_d) {
                    best_d = d;
                    best_k = k;
                }
            }
            codes[i] = (uint8_t)best_k;
        }
    }
    
    // Precompute distance table for a query: m * ksub table
    void compute_distance_table(const float* query, float* table) {
        for(int i=0; i<m; ++i) {
            const float* sub_q = query + i * dsub;
            const float* book = &codebooks[i * ksub * dsub];
            
            for(int k=0; k<ksub; ++k) {
                table[i*ksub + k] = dist_l2_sq(sub_q, book + k*dsub, dsub);
            }
        }
    }
};

struct IVFPQIndex {
    int dim;
    int n_centroids; // coarse
    int m_pq;
    
    KMeans* coarse_quantizer;
    ProductQuantizer* pq;
    
    struct InvertedList {
        std::vector<long long> ids;
        std::vector<uint8_t> codes; // flattened: ids.size() * m_pq
    };
    
    std::vector<InvertedList> lists;
    bool is_trained = false;
    std::mutex mutex;
    
    IVFPQIndex(int d, int nc, int m, int nbits) : dim(d), n_centroids(nc), m_pq(m) {
        coarse_quantizer = new KMeans(nc, d);
        pq = new ProductQuantizer(d, m, nbits);
        lists.resize(nc);
    }
    
    ~IVFPQIndex() {
        delete coarse_quantizer;
        delete pq;
    }
    
    void train(int n, const float* samples) {
        // 1. Train coarse quantizer on all samples
        coarse_quantizer->train(samples, n);
        
        // 2. Compute residuals for PQ training
        std::vector<float> residuals(n * dim);
        for(int i=0; i<n; ++i) {
            const float* vec = samples + i*dim;
            int c = coarse_quantizer->closest(vec);
            const float* centroid = &coarse_quantizer->centroids[c*dim];
            
            for(int j=0; j<dim; ++j) {
                residuals[i*dim + j] = vec[j] - centroid[j];
            }
        }
        
        // 3. Train PQ on residuals
        pq->train(residuals.data(), n);
        is_trained = true;
    }
    
    void insert(long long id, const float* vector) {
        if (!is_trained) return; // Error
        std::unique_lock<std::mutex> lck(mutex);
        
        // 1. Find nearest coarse centroid
        int c = coarse_quantizer->closest(vector);
        
        // 2. Compute residual
        std::vector<float> residual(dim);
        const float* centroid = &coarse_quantizer->centroids[c*dim];
        for(int j=0; j<dim; ++j) {
            residual[j] = vector[j] - centroid[j];
        }
        
        // 3. PQ Encode residual
        std::vector<uint8_t> code(m_pq);
        pq->encode(residual.data(), code.data());
        
        // 4. Store
        lists[c].ids.push_back(id);
        lists[c].codes.insert(lists[c].codes.end(), code.begin(), code.end());
    }
    
    int search(const float* query, int k, int n_probes, long long* out_ids, float* out_dists) {
         if (!is_trained) return 0;
         std::unique_lock<std::mutex> lck(mutex);
         
         // 1. Coarse search: find n_probes nearest centroids
         // We reuse KMeans closest logic but need top-n.
         std::vector<std::pair<float, int>> coarse_matches;
         for(int c=0; c<n_centroids; ++c) {
             float d = dist_l2_sq(query, &coarse_quantizer->centroids[c*dim], dim);
             coarse_matches.push_back({d, c});
         }
         std::sort(coarse_matches.begin(), coarse_matches.end()); // partial sort better
         
         // 2. PQ Search in selected cells
         // Precompute distance table for residuals
         // Query for residual in cell C is (query - centroid_C).
         // Distance(q, y) approx = ||q - C - (y - C)||^2 = || (q-C) - r ||^2
         // Where r is the reconstructed residual.
         // Actually, ADC (Asymmetric Distance Calculation):
         // term1: ||q - C||^2 (computed above) -> Not needed if we only rank? 
         // Real L2 approx: || q - y ||^2 ~ || q - (C + r_dec) ||^2 = || (q - C) - r_dec ||^2
         // So for EACH visited cell, the "query residual" is different (q - C).
         // This means we need to recompute the PQ distance table for EACH cell,
         // OR we use the simplified "multi-index" appraoch?
         // Optim: Many implementations assume PQ on raw vectors or specialized ADC.
         // Standard IVFADC:
         // Dist = || (q - C) - r ||^2.
         // Yes, we must compute distance table for (q - C) for each visited centroid.
         // This is expensive if n_probes is high.
         
         // Heap for top-k
         std::priority_queue<std::pair<float, long long>> top_k;
         
         int probes = std::min(n_probes, n_centroids);
         
         // Reuse table buffer
         int ksub = pq->ksub;
         std::vector<float> dist_table(pq->m * ksub);
         
         std::vector<float> query_residual(dim);
         
         for(int p=0; p<probes; ++p) {
             int c_idx = coarse_matches[p].second;
             // float coarse_dist = coarse_matches[p].first; // Not directly addable unless using specific math
             
             // Compute (q - C)
             const float* centroid = &coarse_quantizer->centroids[c_idx * dim];
             for(int j=0; j<dim; ++j) query_residual[j] = query[j] - centroid[j];
             
             // Compute PQ Distance Table for this residual
             pq->compute_distance_table(query_residual.data(), dist_table.data());
             
             // Scan list
             const auto& list = lists[c_idx];
             size_t num_vecs = list.ids.size();
             
             for(size_t i=0; i<num_vecs; ++i) {
                 float dist = 0.0f;
                 const uint8_t* code = &list.codes[i * m_pq];
                 
                 // Sum sub-quantizer distances
                 for(int m=0; m<m_pq; ++m) {
                     dist += dist_table[m*ksub + code[m]];
                 }
                 
                 top_k.push({dist, list.ids[i]}); // Max heap keeps largest distances?
                 // Wait, we want smallest distances. PriorityQueue is Max Heap by default.
                 // So top() is largest distance (worst).
                 // We keep 'k' smallest. 
                 
                 if (top_k.size() > (size_t)k) {
                     top_k.pop(); // Remove worst
                 }
             }
         }
         
         // Extract
         std::vector<std::pair<float, long long>> results;
         while(!top_k.empty()) {
             results.push_back(top_k.top());
             top_k.pop();
         }
         // Reverse to get best first
         int actual_k = std::min((int)results.size(), k);
         for(int i=0; i<actual_k; ++i) {
             out_ids[i] = results[results.size() - 1 - i].second;
             out_dists[i] = results[results.size() - 1 - i].first;
         }
         return actual_k;
    }
};

// C Wrappers
EXPORT IVFPQIndex* ivf_pq_create(int dim, int n_centroids, int m_subquantizers, int n_bits) {
    return new IVFPQIndex(dim, n_centroids, m_subquantizers, n_bits);
}

EXPORT void ivf_pq_destroy(IVFPQIndex* index) {
    if (index) delete index;
}

EXPORT void ivf_pq_train(IVFPQIndex* index, int n_samples, const float* samples) {
    if (index) index->train(n_samples, samples);
}

EXPORT void ivf_pq_insert(IVFPQIndex* index, long long id, const float* vector) {
    if (index) index->insert(id, vector);
}

EXPORT int ivf_pq_search(IVFPQIndex* index, const float* query, int k, int n_probes, long long* out_ids, float* out_dists) {
    if (index) return index->search(query, k, n_probes, out_ids, out_dists);
    return 0;
}

EXPORT int ivf_pq_save(IVFPQIndex* index, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) return 0;
    
    // Header
    int magic = 0x56781234;
    fwrite(&magic, sizeof(int), 1, f);
    fwrite(&index->dim, sizeof(int), 1, f);
    fwrite(&index->n_centroids, sizeof(int), 1, f);
    fwrite(&index->m_pq, sizeof(int), 1, f);
    
    // Coarse Centroids
    fwrite(index->coarse_quantizer->centroids.data(), sizeof(float), index->n_centroids * index->dim, f);
    
    // PQ Codebooks
    size_t pq_size = index->pq->codebooks.size();
    fwrite(index->pq->codebooks.data(), sizeof(float), pq_size, f);
    
    // Inverted Lists
    for(int i=0; i<index->n_centroids; ++i) {
        size_t list_size = index->lists[i].ids.size();
        fwrite(&list_size, sizeof(size_t), 1, f);
        if (list_size > 0) {
            fwrite(index->lists[i].ids.data(), sizeof(long long), list_size, f);
            fwrite(index->lists[i].codes.data(), sizeof(uint8_t), list_size * index->m_pq, f);
        }
    }
    
    fclose(f);
    return 1;
}

EXPORT IVFPQIndex* ivf_pq_load(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return nullptr;
    
    int magic = 0x56781234;
    if (fread(&magic, sizeof(int), 1, f) != 1 || magic != 0x56781234) {
        fclose(f);
        return nullptr;
    }
    
    int dim, nc, mpq;
    fread(&dim, sizeof(int), 1, f);
    fread(&nc, sizeof(int), 1, f);
    fread(&mpq, sizeof(int), 1, f);
    
    IVFPQIndex* index = new IVFPQIndex(dim, nc, mpq, 8);
    index->is_trained = true;
    
    // Coarse
    fread(index->coarse_quantizer->centroids.data(), sizeof(float), nc * dim, f);
    
    // PQ
    size_t pq_size = index->pq->codebooks.size();
    fread(index->pq->codebooks.data(), sizeof(float), pq_size, f);
    
    // Lists
    for(int i=0; i<nc; ++i) {
        size_t list_size;
        fread(&list_size, sizeof(size_t), 1, f);
        if (list_size > 0) {
            index->lists[i].ids.resize(list_size);
            index->lists[i].codes.resize(list_size * mpq);
            
            fread(index->lists[i].ids.data(), sizeof(long long), list_size, f);
            fread(index->lists[i].codes.data(), sizeof(uint8_t), list_size * mpq, f);
        }
    }
    
    fclose(f);
    return index;
}





