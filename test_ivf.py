import sys
import os
import numpy as np
import time

# Add src/api to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "api"))
try:
    from wrapper import IVFPQIndex, VectorForgeIndex
except ImportError:
    sys.path.append("src/api")
    from wrapper import IVFPQIndex, VectorForgeIndex

def main():
    print("--- IVF-PQ TEST ---")
    dim = 128
    n = 2000
    n_centroids = 16
    m = 16 # 16 subquantizers (128/16 = 8 dim per sub)
    nbits = 8 # 256 centroids per sub
    
    print(f"Generating {n} vectors of dim {dim}...")
    np.random.seed(42)
    # Generate clustered data to verify recall
    centers = np.random.rand(n_centroids, dim).astype(np.float32)
    data = []
    ids = []
    
    for i in range(n):
        c = centers[i % n_centroids]
        noise = np.random.randn(dim).astype(np.float32) * 0.1
        vec = c + noise
        data.append(vec)
        ids.append(i)
        
    data = np.array(data, dtype=np.float32)
    
    print("Initializing IVF-PQ Index...")
    idx = IVFPQIndex(dim, n_centroids=n_centroids, m_subquantizers=m, n_bits=nbits)
    
    print("Training...")
    start = time.time()
    idx.train(data) # Train on all data for simplicity
    print(f"Training took {time.time()-start:.4f}s")
    
    print("Inserting...")
    start = time.time()
    for i in range(n):
        idx.insert(ids[i], data[i])
    print(f"Insertion took {time.time()-start:.4f}s")
    
    print("Searching...")
    query = data[0] # Search for first vector (should be close to ID 0)
    
    start = time.time()
    res_ids, res_dists = idx.search(query, k=5, n_probes=4)
    end = time.time()
    
    print(f"Search took {(end-start)*1000:.4f}ms")
    print(f"Query ID: 0")
    print(f"Results: {res_ids}")
    print(f"Dists: {res_dists}")
    
    if 0 in res_ids:
        print("SUCCESS: Found target vector.")
    else:
        print("FAILURE: Target not found.")
        
    # Persistence
    print("\n--- Persistence ---")
    idx.save("ivf.bin")
    idx2 = IVFPQIndex.load("ivf.bin")
    res_ids2, _ = idx2.search(query, k=5)
    print(f"Loaded Results: {res_ids2}")
    if res_ids == res_ids2:
         print("SUCCESS: Persistence match.")
    else:
         print("FAILURE: Persistence mismatch.")

if __name__ == "__main__":
    main()

