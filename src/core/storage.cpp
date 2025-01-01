#include "storage.h"
#include <cstring>
#include <iostream>
#include <stdexcept>

// --- RAM Storage ---


// --- MMap Storage ---
MMapStorage::MMapStorage(int d, const char* fname) : dim(d), filename(fname) {
#ifdef _WIN32
    // Open/Create file
    hFile = CreateFileA(filename.c_str(), GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Failed to open MMap file");
    }
    
    // Get current size
    DWORD sizeHigh;
    DWORD sizeLow = GetFileSize(hFile, &sizeHigh);
    size_t fileSize = ((size_t)sizeHigh << 32) | sizeLow;
    
    size_t vec_size = sizeof(long long) + dim*sizeof(float); // ID + Vector
    count = fileSize / vec_size;
    capacity = count; 
    
    if (capacity < 1000) capacity = 1000; // Min capacity
    
    expand(capacity);
#endif
}

MMapStorage::~MMapStorage() {
#ifdef _WIN32
    if (pData) UnmapViewOfFile(pData);
    if (hMap) CloseHandle(hMap);
    if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
#endif
}

void MMapStorage::expand(size_t new_cap) {
#ifdef _WIN32
    if (pData) UnmapViewOfFile(pData);
    if (hMap) CloseHandle(hMap);
    
    capacity = new_cap;
    size_t vec_size = sizeof(long long) + dim*sizeof(float);
    size_t total_bytes = capacity * vec_size;
    
    DWORD sizeHigh = (DWORD)(total_bytes >> 32);
    DWORD sizeLow = (DWORD)(total_bytes & 0xFFFFFFFF);
    
    hMap = CreateFileMappingA(hFile, NULL, PAGE_READWRITE, sizeHigh, sizeLow, NULL);
    if (!hMap) throw std::runtime_error("CreateFileMapping failed");
    
    pData = (float*)MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, 0); // Map entire file
    if (!pData) throw std::runtime_error("MapViewOfFile failed");
#endif
}

void MMapStorage::insert(long long id, const float* vector) {
    std::unique_lock<std::mutex> lck(mutex);
    
    if (count >= capacity) {
        expand(capacity * 2);
    }
    
    // Layout: [ID (int64)][Vector (float*dim)] ...
    // Calculate offset in floats. 
    // ID is 64bit = 2 floats.
    // Stride = 2 + dim floats
    size_t stride_floats = 2 + dim;
    float* ptr = pData + count * stride_floats;
    
    // Store ID (cast to float buffer, bitwise copy)
    memcpy(ptr, &id, sizeof(long long));
    
    // Store Vector
    memcpy(ptr + 2, vector, dim * sizeof(float));
    
    count++;
}

const float* MMapStorage::get_vector_ptr(long long id) const {
    // id here must be INDEX (0..N). 
    // If it is external ID, we can't find it instantly without an index.
    // Limitation: MMapStorage assumes sequential append. 'id' arg here is 'index'.
    // The HNSW Graph nodes use 'index' (internal id) to reference storage.
    
    size_t stride_floats = 2 + dim;
    if (id >= 0 && (size_t)id < count) {
        float* ptr = pData + id * stride_floats;
        return ptr + 2; // Skip ID
    }
    return nullptr;
}

void MMapStorage::get_vector(long long id, float* out_vector) const {
    const float* ptr = get_vector_ptr(id);
    if (ptr) {
        memcpy(out_vector, ptr, dim * sizeof(float));
    }
}

size_t MMapStorage::size() const { return count; }
int MMapStorage::get_dim() const { return dim; }




