from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import time
from fastapi.concurrency import run_in_threadpool
import ctypes

# Import wrapper
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "api"))
try:
    from wrapper import VectorForgeIndex, lib
except ImportError:
    # Try importing directly if packaging structure differs
    sys.path.append(os.path.dirname(__file__)) 
    from wrapper import VectorForgeIndex, lib

app = FastAPI(title="VectorForge API", version="1.0")

# Global Index
INDEX_FILE = "vectorforge_index.bin"
DIM = 128 # Default dimension
index: Optional[VectorForgeIndex] = None

class InsertRequest(BaseModel):
    id: int
    vector: List[float]

class SearchRequest(BaseModel):
    vector: List[float]
    k: int = 10

class SearchResponse(BaseModel):
    id: int
    distance: float

@app.on_event("startup")
def startup_event():
    global index
    if os.path.exists(INDEX_FILE):
        print(f"Loading index from {INDEX_FILE}...")
        try:
            index = VectorForgeIndex.load(INDEX_FILE)
            print(f"Index loaded. Dim: {index.dim}")
        except Exception as e:
            print(f"Failed to load index: {e}")
            index = VectorForgeIndex(dim=DIM)
    else:
        print(f"Creating new index with dim={DIM}...")
        index = VectorForgeIndex(dim=DIM)
        
    # Init WAL
    lib.wal_init("vectorforge.wal".encode('utf-8'))

@app.on_event("shutdown")
def shutdown_event():
    global index
    if index:
        print(f"Saving index to {INDEX_FILE}...")
        # Since run_in_threadpool is for async routes, shutdown is sync usually or we just call save
        index.save(INDEX_FILE)
    lib.wal_close()

@app.post("/insert")
async def insert_vector(req: InsertRequest):
    global index
    if not index:
        raise HTTPException(status_code=500, detail="Index not initialized")
    
    if len(req.vector) != index.dim:
        raise HTTPException(status_code=400, detail=f"Vector dimension mismatch. Expected {index.dim}, got {len(req.vector)}")
        
    # Log to WAL (Sync is fine for durability, or threadpool if slow IO)
    vec_arr = (ctypes.c_float * index.dim)(*req.vector)
    lib.wal_log_insert(req.id, vec_arr, index.dim)
    
    # Insert to memory (Thread safe now due to C++ locks)
    await run_in_threadpool(index.insert, req.id, req.vector)
    
    return {"status": "ok", "id": req.id}

@app.post("/search", response_model=List[SearchResponse])
async def search_vector(req: SearchRequest):
    global index
    if not index:
        raise HTTPException(status_code=500, detail="Index not initialized")
        
    if len(req.vector) != index.dim:
        raise HTTPException(status_code=400, detail=f"Vector dimension mismatch. Expected {index.dim}, got {len(req.vector)}")
        
    start = time.time()
    # Run heavy search in threadpool to avoid blocking event loop
    ids, dists = await run_in_threadpool(index.search, req.vector, k=req.k)
    end = time.time()
    
    results = []
    for i, d in zip(ids, dists):
        results.append(SearchResponse(id=i, distance=d))
        
    # print(f"Search took {(end-start)*1000:.4f}ms")
    return results

@app.post("/save")
def save_index():
    global index
    if index:
        index.save(INDEX_FILE)
        return {"status": "saved", "path": INDEX_FILE}
    return {"status": "error", "message": "No index"}

@app.get("/stats")
def get_stats():
    global index
    if index:
        return {
            "dim": index.dim,
            "backend": "HNSW+SIMD",
            "status": "active"
        }
    return {"status": "inactive"}







