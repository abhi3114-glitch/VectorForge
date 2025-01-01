#pragma once
#include <vector>
#include <string>
#include <mutex>
#include <cstring>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// Abstract base class for Vector Storage
class VectorStorage {
public:
    virtual ~VectorStorage() {}
    virtual void insert(long long id, const float* vector) = 0;
    virtual void get_vector(long long id, float* out_vector) const = 0;
    virtual const float* get_vector_ptr(long long id) const = 0; // Return pointer if possible (RAM), or internal buffer
    virtual size_t size() const = 0;
    virtual int get_dim() const = 0;
};

// RAM Storage (Original)
// RAM Storage (Original)
class RAMStorage : public VectorStorage {
    int dim;
    struct Record {
        long long id;
        std::vector<float> vec;
    };
    std::vector<Record> data;
    mutable std::mutex mutex;

public:
    RAMStorage(int d) : dim(d) {
        data.reserve(10000);
    }

    void insert(long long id, const float* vector) override {
        // std::unique_lock<std::mutex> lck(mutex);
        Record r;
        r.id = id;
        r.vec.resize(dim);
        std::memcpy(r.vec.data(), vector, dim * sizeof(float));
        data.push_back(std::move(r));
    }

    void get_vector(long long id, float* out_vector) const override {
        // std::unique_lock<std::mutex> lck(mutex);
        if (id >= 0 && id < (long long)data.size()) {
            std::memcpy(out_vector, data[id].vec.data(), dim * sizeof(float));
        }
    }

    const float* get_vector_ptr(long long id) const override {
        // std::unique_lock<std::mutex> lck(mutex);
        if (id >= 0 && id < (long long)data.size()) {
            return data[id].vec.data();
        }
        return nullptr;
    }

    size_t size() const override {
        // std::unique_lock<std::mutex> lck(mutex);
        return data.size();
    }

    int get_dim() const override { return dim; }
};

// MMap Storage (Disk backed)
class MMapStorage : public VectorStorage {
    int dim;
    std::string filename;
    size_t count = 0;
    size_t capacity = 0;
    
#ifdef _WIN32
    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMap = NULL;
    float* pData = nullptr;
#endif
    
    mutable std::mutex mutex;
    
    void expand(size_t new_cap);

public:
    MMapStorage(int d, const char* fname);
    ~MMapStorage();
    
    void insert(long long id, const float* vector) override;
    void get_vector(long long id, float* out_vector) const override;
    const float* get_vector_ptr(long long id) const override;
    
    size_t size() const override;
    int get_dim() const override;
};









