# VectorForge

VectorForge is a high-performance, embedded vector database built from scratch in C++ with Python bindings. It is designed for real-time Approximate Nearest Neighbor (ANN) search, offering sub-millisecond query latency and high-throughput ingestion.

The system features a hybrid architecture: a highly optimized C++17 core handles all heavy lifting (SIMD distance calculations, graph traversal, memory management), while a lightweight Python layer manages the API and orchestration.

## Key Features

*   **Hybrid Architecture**: C++ Core for performance, Python for ease of use.
*   **HNSW Indexing**: Hierarchical Navigable Small World graphs for fast, accurate in-memory search.
*   **IVF-PQ Indexing**: Inverted File System with Product Quantization for compressing large datasets (4x-16x memory reduction).
*   **SIMD Acceleration**: Hand-optimized AVX2/FMA intrinsics for L2 and Inner Product distance calculations.
*   **Manual Memory Management**: The Core engine uses raw pointers and manual allocation strategies to ensure absolute stability and alignment on Windows/MinGW systems, eliminating STL ABI incompatibilities.
*   **Thread Safety**: Fine-grained Read-Writer locks (`std::shared_mutex`) allow concurrent searching while inserting.
*   **Durability**: Synchronous Write-Ahead Logging (WAL) ensures zero data loss in the event of a crash.
*   **Persistence**: Binary serialization for saving and loading indices to disk.

## System Architecture

The project consists of three main layers:

1.  **C++ Core (`src/core`)**:
    *   `hnsw.cpp`: The graph-based index implementation using manual memory management.
    *   `ivf_pq.cpp`: K-Means clustering and Product Quantization logic.
    *   `distance.cpp`: Hardware-accelerated distance metrics.
    *   `wal.cpp`: Append-only transaction log for durability.
    *   `bridge.cpp`: `extern "C"` interface exporting functionality to the shared library (`vectorforge_core.dll`).

2.  **Python Bindings (`src/api/wrapper.py`)**:
    *   Loads the DLL using `ctypes`.
    *   Provides high-level, Pythonic classes (`VectorForgeIndex`, `IVFPQIndex`) that handle memory lifecycle and pointer marshaling.

3.  **REST API (`src/api/server.py`)**:
    *   A production-ready FastAPI server.
    *   Manages the index instance.
    *   Offloads CPU-bound search tasks to a thread pool to maintain event loop responsiveness.

## Prerequisites

*   **Operating System**: Windows (Tested on Windows 10/11).
*   **Compiler**: MinGW-w64 (G++) or MSVC with C++17 support.
*   **Python**: Python 3.8 or higher.
*   **Dependencies**:
    *   `numpy`
    *   `fastapi`
    *   `uvicorn`
    *   `requests` (for testing)

## Installation and Build

1.  **Clone the Repository**:
    Ensure you have the full source code structure.

2.  **Build the C++ Core**:
    Execute the provided batch script to compile the shared library.
    ```powershell
    .\build.bat
    ```
    This will create `build\vectorforge_core.dll`.
    *Note: The build script applies `-O3 -mavx2` optimizations automatically.*

3.  **Install Python Dependencies**:
    ```powershell
    pip install numpy fastapi uvicorn requests
    ```

## Usage: Python API

You can use VectorForge directly in your Python scripts for maximum performance.

### HNSW Index (In-Memory, High Precision)

```python
import numpy as np
from src.api.wrapper import VectorForgeIndex

# Initialize Index (Dimension=128)
index = VectorForgeIndex(dim=128, M=16, ef_construction=200)

# Insert Vectors
data = np.random.rand(1000, 128).astype(np.float32)
for i in range(1000):
    index.insert(id=i, vector=data[i])

# Search
query = np.random.rand(128).astype(np.float32)
ids, dists = index.search(query, k=5)

print("Nearest Neighbors:", ids)
print("Distances:", dists)

# Save to Disk
index.save("my_index.bin")

# Load from Disk
loaded_index = VectorForgeIndex.load("my_index.bin")
```

### IVF-PQ Index (Large Datasets, Compressed)

```python
from src.api.wrapper import IVFPQIndex

# Initialize
index = IVFPQIndex(dim=128, n_centroids=256, m_subquantizers=16, n_bits=8)

# Train (Requires a representative dataset)
training_data = np.random.rand(5000, 128).astype(np.float32)
index.train(training_data)

# Insert
index.insert(id=1, vector=training_data[0])

# Search (n_probes determines accuracy vs speed trade-off)
ids, dists = index.search(query=training_data[0], k=5, n_probes=16)
```

## Usage: REST API

VectorForge includes a standalone API server for microservice deployments.

1.  **Start the Server**:
    ```powershell
    python src/api/server.py
    # OR
    uvicorn src.api.server:app --port 8000
    ```

2.  **API Endpoints**:

    *   **POST /insert**: Add a vector.
        ```json
        {
          "id": 123,
          "vector": [0.1, 0.5, ...]
        }
        ```

    *   **POST /search**: Find nearest neighbors.
        ```json
        {
          "vector": [0.1, 0.5, ...],
          "k": 10
        }
        ```

    *   **GET /stats**: Check index status.

## Technical Details

### Manual Memory Management
Stability was prioritized over convenience. The core HNSW implementation explicitly avoids `std::vector` and complex STL containers within the node structure (`HNSWNode`). Instead, it uses raw pointers (`float*`, `int**`) and `new`/`delete`. This design choice solves persistent "Access Violation" (0xc00000ff) errors common when mixing MinGW STL binaries with Python ctypes on Windows.

### Persistence and Durability
*   **WAL**: All insert operations are synchronously written to `vectorforge.wal` in a binary format before being applied to the memory index. This ensures data is recoverable even if the process is terminated abruptly.
*   **Checkpoints**: The `save()` method serializes the entire graph structure (nodes, links, level info) to a compact binary file, allowing for fast startups.

## Performance Benchmarks

Internal tests on an AVX2-compatible CPU show the following performance metrics for a 128-dimensional dataset:

*   **Insertion Throughput**: ~60,000 vectors/second.
*   **Search Latency**: < 0.2 milliseconds (p95) for top-10 retrieval.

## Tests

The project includes a comprehensive regression suite:

*   `test_basic.py`: Verifies core HNSW logic, persistence, and basic recall.
*   `test_ivf.py`: Verifies IVF-PQ training, quantization accuracy, and retrieval.
*   `test_server.py`: Verifies API endpoints, threading, and concurrency handling.



