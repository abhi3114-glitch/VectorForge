#include <fstream>
#include <mutex>
#include <string>
#include <vector>

// Log format:
// [type:1byte][size:4bytes][data...]
// Types: 0=Insert, 1=Delete(future)

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

class WAL {
    std::ofstream log_file;
    std::mutex mtx;
    std::string filename;
    
public:
    WAL(const std::string& fname) : filename(fname) {
        log_file.open(filename, std::ios::out | std::ios::app | std::ios::binary);
    }
    
    ~WAL() {
        if (log_file.is_open()) log_file.close();
    }
    
    void log_insert(long long id, const float* vector, int dim) {
        std::lock_guard<std::mutex> lock(mtx);
        if (!log_file.is_open()) return;
        
        uint8_t type = 0; // Insert
        log_file.write((char*)&type, 1);
        
        int size = sizeof(long long) + dim * sizeof(float);
        log_file.write((char*)&size, sizeof(int));
        
        log_file.write((char*)&id, sizeof(long long));
        log_file.write((char*)vector, dim * sizeof(float));
        
        log_file.flush(); // Ensure durability
    }
    
    // Simple Replay iterator would be complex to expose to C API simply.
    // For now, we will just provide append support for safety.
    // Replay logic is usually done by loading the WAL file manually during startup in app.
};

// Global WAL instance (singleton for prototype simplicity)
static WAL* g_wal = nullptr;

extern "C" {
    EXPORT void wal_init(const char* filename) {
        if (g_wal) delete g_wal;
        g_wal = new WAL(filename);
    }
    
    EXPORT void wal_log_insert(long long id, const float* vector, int dim) {
        if (g_wal) g_wal->log_insert(id, vector, dim);
    }
    
    EXPORT void wal_close() {
        if (g_wal) {
            delete g_wal;
            g_wal = nullptr;
        }
    }
}





