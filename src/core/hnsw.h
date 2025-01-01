#pragma once
#include <mutex>
#include <random>
#include <cstring> 

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

// Forward declaration
struct HNSWIndex;

extern "C" {
    EXPORT HNSWIndex* hnsw_create(int dim, int M, int ef_construction);
    EXPORT void hnsw_destroy(HNSWIndex* index);
    EXPORT void hnsw_insert(HNSWIndex* index, long long id, const float* vector);
    EXPORT int hnsw_search(HNSWIndex* index, const float* query, int k, long long* out_ids, float* out_dists);
    EXPORT void hnsw_set_storage(HNSWIndex* index, const char* type, const char* path);
}

// Internal structures - RAW POINTERS for safety
struct HNSWNode {
    long long id;
    float* vector; 
    int** links; 
    int* link_counts; // current count per level
    int max_level;
    
    HNSWNode(long long _id, const float* vec, int dim, int _max_level, int M, int M0) 
        : id(_id), max_level(_max_level), link_counts(new int[_max_level + 1]) {
        
        vector = new float[dim];
        if (vec) std::memcpy(vector, vec, dim * sizeof(float));
        
        links = new int*[_max_level + 1];
        
        for(int i=0; i<=_max_level; ++i) {
             int cap = (i == 0) ? M0 : M;
             links[i] = new int[cap]; 
             link_counts[i] = 0;
        }
    }
    
    ~HNSWNode() {
        delete[] vector;
        if (links) {
            for(int i=0; i<=max_level; ++i) {
                if (links[i]) delete[] links[i];
            }
            delete[] links;
        }
        if (link_counts) delete[] link_counts;
    }
};

struct HNSWIndex {
    int dim;
    int M;
    int ef_construction;
    
    // We can use std::vector for the main list if we are careful, 
    // or use a fixed size array / linked list.
    // std::vector<HNSWNode*> is typically safe if HNSWNode is pointer.
    // The issue before was inside HNSWNode constructor std::vector resize.
    
    std::vector<HNSWNode*> nodes;
    int enterpoint_node = -1;
    int max_level = -1;
    
    double level_mult;
    std::mt19937 rng;
    std::mutex mutex; 
    
    HNSWIndex(int d, int m, int ef);
    ~HNSWIndex();
    
    int get_random_level();
};







