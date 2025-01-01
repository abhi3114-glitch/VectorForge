import sys
import os

# Add src/api to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "api"))

try:
    from wrapper import VectorForgeIndex
except ImportError:
    # Try local import if running from root
    sys.path.append("src/api")
    from wrapper import VectorForgeIndex

import numpy as np
import time

def main():
    print("Initializing Index...")
    dim = 4
    idx = VectorForgeIndex(dim=dim, M=16, ef_construction=100)
    
    # Insert some data
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.9, 0.1, 0.0, 0.0]
    ]
    ids = [1, 2, 3, 4, 5]
    
    print("Inserting vectors...")
    for i, v in zip(ids, vectors):
        idx.insert(i, v)
        
    # Search
    query = [1.0, 0.0, 0.0, 0.0]
    print(f"Searching for {query}...")
    
    ids, dists = idx.search(query, k=3)
    
    print("Results:")
    for i, d in zip(ids, dists):
        print(f"ID: {i}, Dist: {d}")
        
    print("\n--- BENCHMARK ---")
    # Synthetic benchmark
    N = 10000
    D = 128
    print(f"Generating {N} vectors of dim {D}...")
    data = np.random.rand(N, D).astype(np.float32)
    
    idx_bench = VectorForgeIndex(dim=D)
    
    start = time.time()
    for i in range(N):
        idx_bench.insert(i, data[i])
    dur = time.time() - start
    print(f"Insertion time: {dur:.4f}s ({N/dur:.2f} QPS)")
    
    query = np.random.rand(D).astype(np.float32)
    start = time.time()
    idx_bench.search(query, k=10)
    end = time.time()
    print(f"Search time: {(end-start)*1000:.4f}ms")

    print("\n--- PERSISTENCE TEST ---")
    save_path = "test_index.bin"
    print(f"Saving index to {save_path}...")
    idx.save(save_path)
    
    print("Loading index...")
    idx_loaded = VectorForgeIndex.load(save_path)
    
    print("Searching loaded index...")
    ids_l, dists_l = idx_loaded.search(query=[1.0, 0.0, 0.0, 0.0], k=3)
    
    print("Loaded Results:")
    for i, d in zip(ids_l, dists_l):
        print(f"ID: {i}, Dist: {d}")
        
    # Verify match
    # Original results
    ids_o, dists_o = idx.search(query=[1.0, 0.0, 0.0, 0.0], k=3)
    
    if ids_l == ids_o:
        print("SUCCESS: Persistence working (results match).")
    else:
        print("FAILURE: Persistence results mismatch.")
        print(f"Original: {ids_o}")
        print(f"Loaded:   {ids_l}")
    
    # Cleanup
    if os.path.exists(save_path):
        os.remove(save_path)

if __name__ == "__main__":
    main()














