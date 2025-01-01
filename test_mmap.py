import sys
import os
import numpy as np
import time

# Add src/api to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "api"))
try:
    from wrapper import VectorForgeIndex
except ImportError:
    sys.path.append("src/api")
    from wrapper import VectorForgeIndex

def main():
    print("--- MMAP STORAGE TEST ---")
    dim = 64
    n = 5000
    
    print(f"Creating Index with MMAP Storage (dim={dim})...")
    idx = VectorForgeIndex(dim=dim)
    
    # Configure storage to MMAP
    # This must be done BEFORE insertion
    storage_file = "vectors_mmap.bin"
    if os.path.exists(storage_file):
        os.remove(storage_file)
        
    print(f"Setting storage backend to mmap at {storage_file}...")
    idx.set_storage("mmap", storage_file)
    
    print(f"Generating {n} vectors...")
    np.random.seed(42)
    data = []
    ids = []
    
    # Generate data
    for i in range(n):
        vec = np.random.rand(dim).astype(np.float32)
        data.append(vec)
        ids.append(i * 10) # Using non-sequential IDs to verify
        
    print("Inserting...")
    start = time.time()
    for i in range(n):
        idx.insert(ids[i], data[i])
    dur = time.time() - start
    print(f"Insertion took {dur:.4f}s ({n/dur:.0f} QPS)")
    
    print("Verifying Storage...")
    # Since HNSW graph uses storage to compute distance, a successful search implies storage works.
    query = data[123]
    target_id = ids[123]
    
    start = time.time()
    res_ids, res_dists = idx.search(query, k=5)
    end = time.time()
    
    print(f"Search took {(end-start)*1000:.4f}ms")
    print(f"Query ID: {target_id}")
    print(f"Results: {res_ids}")
    
    if target_id in res_ids:
        print("SUCCESS: Found target vector (Data retrieved from Disk correctly).")
    else:
        print("FAILURE: Target not found.")
        
    # Verify file size
    if os.path.exists(storage_file):
        size = os.path.getsize(storage_file)
        expected_min = n * dim * 4
        print(f"Storage File Size: {size/1024/1024:.2f} MB (Expected > {expected_min/1024/1024:.2f} MB)")
        if size > expected_min:
             print("SUCCESS: File size matches expected storage usage.")
        else:
             print("FAILURE: File too small.")
    else:
        print("FAILURE: File not created.")

if __name__ == "__main__":
    main()





