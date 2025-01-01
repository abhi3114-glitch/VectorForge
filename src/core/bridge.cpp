#include "hnsw.h"
#include "ivf_pq.h"

// This file simply ensures the extern "C" functions are compiled 
// and available in the DLL. The declarations are in hnsw.h 
// but we need to ensure the linker sees them if they were just in headers.
// However, hnsw.cpp implements them directly.
// We can use this file for additional bridge logic if needed, 
// e.g., helper functions for array conversions or errors.

extern "C" {
    EXPORT int vectorforge_version() {
        return 1;
    }
}






