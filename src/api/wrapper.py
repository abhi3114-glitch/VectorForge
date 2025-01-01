import ctypes
import os
import numpy as np
from ctypes import c_int, c_float, c_void_p, c_longlong, POINTER

# Load the DLL
# Try to find it in build/ relative to this file or cwd
dll_path = os.path.join(os.path.dirname(__file__), "..", "..", "build", "vectorforge_core.dll")
if not os.path.exists(dll_path):
    dll_path = os.path.abspath("build/vectorforge_core.dll")

try:
    lib = ctypes.CDLL(dll_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find vectorforge_core.dll at {dll_path}")

# Check version
try:
    lib.vectorforge_version.restype = c_int
    print(f"VectorForge Core Version: {lib.vectorforge_version()}")
except:
    print("Warning: Could not get version from DLL")

# HNSW Bindings
lib.vectorforge_version.restype = c_int
# ...
lib.hnsw_create.argtypes = [c_int, c_int, c_int]
lib.hnsw_create.restype = c_void_p

lib.hnsw_destroy.argtypes = [c_void_p]
lib.hnsw_destroy.restype = None

lib.hnsw_insert.argtypes = [c_void_p, c_longlong, POINTER(c_float)]
lib.hnsw_insert.restype = None

lib.hnsw_search.argtypes = [c_void_p, POINTER(c_float), c_int, POINTER(c_longlong), POINTER(c_float)]
lib.hnsw_search.restype = c_int

lib.hnsw_set_storage.argtypes = [c_void_p, ctypes.c_char_p, ctypes.c_char_p]
lib.hnsw_set_storage.restype = None

lib.hnsw_set_storage.argtypes = [c_void_p, ctypes.c_char_p, ctypes.c_char_p]
lib.hnsw_set_storage.restype = None

lib.hnsw_save.argtypes = [c_void_p, ctypes.c_char_p]
lib.hnsw_save.restype = c_int

lib.hnsw_load.argtypes = [ctypes.c_char_p]
lib.hnsw_load.restype = c_void_p

lib.ivf_pq_create.argtypes = [c_int, c_int, c_int, c_int]
lib.ivf_pq_create.restype = c_void_p

lib.ivf_pq_destroy.argtypes = [c_void_p]
lib.ivf_pq_destroy.restype = None

lib.ivf_pq_train.argtypes = [c_void_p, c_int, POINTER(c_float)]
lib.ivf_pq_train.restype = None

lib.ivf_pq_insert.argtypes = [c_void_p, c_longlong, POINTER(c_float)]
lib.ivf_pq_insert.restype = None

lib.ivf_pq_search.argtypes = [c_void_p, POINTER(c_float), c_int, c_int, POINTER(c_longlong), POINTER(c_float)]
lib.ivf_pq_search.restype = c_int

lib.ivf_pq_save.argtypes = [c_void_p, ctypes.c_char_p]
lib.ivf_pq_save.restype = c_int

lib.ivf_pq_load.argtypes = [ctypes.c_char_p]
lib.ivf_pq_load.restype = c_void_p

lib.wal_init.argtypes = [ctypes.c_char_p]
lib.wal_init.restype = None

lib.wal_log_insert.argtypes = [c_longlong, POINTER(c_float), c_int]
lib.wal_log_insert.restype = None

lib.wal_close.argtypes = []
lib.wal_close.restype = None

class VectorForgeIndex:
    def __init__(self, dim, M=16, ef_construction=200, handle=None):
        self.dim = dim
        if handle:
            self.obj = handle
        else:
            self.obj = lib.hnsw_create(dim, M, ef_construction)
            print(f"DEBUG: VectorForgeIndex created handle: {self.obj}")
        
            print(f"DEBUG: VectorForgeIndex created handle: {self.obj}")

    def __del__(self):
        if hasattr(self, 'obj') and self.obj:
            lib.hnsw_destroy(self.obj)
            
    def set_storage(self, type: str, path: str):
        lib.hnsw_set_storage(self.obj, type.encode('utf-8'), path.encode('utf-8'))
            
    def insert(self, id: int, vector):
        if len(vector) != self.dim:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dim}, got {len(vector)}")
        
        # Ensure numpy array of float32
        v_arr = np.array(vector, dtype=np.float32)
        v_ptr = v_arr.ctypes.data_as(POINTER(c_float))
        lib.hnsw_insert(self.obj, id, v_ptr)
        
    def search(self, query, k=10):
        if len(query) != self.dim:
            raise ValueError(f"Query dimension mismatch. Expected {self.dim}, got {len(query)}")
            
        q_arr = np.array(query, dtype=np.float32)
        q_ptr = q_arr.ctypes.data_as(POINTER(c_float))
        
        ids = (c_longlong * k)()
        dists = (c_float * k)()
        
        count = lib.hnsw_search(self.obj, q_ptr, k, ids, dists)
        
        result_ids = []
        result_dists = []
        for i in range(count):
            result_ids.append(ids[i])
            result_dists.append(dists[i])
            
        return result_ids, result_dists

    def save(self, filename: str):
        b_filename = filename.encode('utf-8')
        ret = lib.hnsw_save(self.obj, b_filename)
        if ret == 0:
            raise RuntimeError("Failed to save index")

    @staticmethod
    def load(filename: str):
        b_filename = filename.encode('utf-8')
        handle = lib.hnsw_load(b_filename)
        if not handle:
            raise RuntimeError("Failed to load index")
        
        # We need to know dim to create the wrapper properly. 
        # But dim is inside the object. HNSWIndex struct is opaque here.
        # We can read it from file header manually in python or just expose a getter.
        # For this MVP, let's just cheat and assume the user knows dim or we read the header.
        # Actually, let's add a getter for dim in C++?
        # Or, just assume the user passes dim to `load` wrapper or we read 2nd int from file.
        # Let's read the file header in Python to get dim.
        
        with open(filename, 'rb') as f:
            import struct
            f.read(4) # magic
            dim = struct.unpack('i', f.read(4))[0]
            
        with open(filename, 'rb') as f:
            import struct
            f.read(4) # magic
            dim = struct.unpack('i', f.read(4))[0]
            
        return VectorForgeIndex(dim, handle=handle)

class IVFPQIndex:
    def __init__(self, dim, n_centroids=16, m_subquantizers=4, n_bits=8, handle=None):
        self.dim = dim
        if handle:
            self.obj = handle
        else:
            self.obj = lib.ivf_pq_create(dim, n_centroids, m_subquantizers, n_bits)
            
    def __del__(self):
        if hasattr(self, 'obj') and self.obj:
            lib.ivf_pq_destroy(self.obj)
            
    def train(self, samples):
        # samples: numpy array
        n = len(samples)
        flat = np.array(samples, dtype=np.float32).flatten()
        ptr = flat.ctypes.data_as(POINTER(c_float))
        lib.ivf_pq_train(self.obj, n, ptr)
        
    def insert(self, id: int, vector):
        if len(vector) != self.dim:
            raise ValueError(f"Dim mismatch")
        v_arr = np.array(vector, dtype=np.float32)
        v_ptr = v_arr.ctypes.data_as(POINTER(c_float))
        lib.ivf_pq_insert(self.obj, id, v_ptr)
        
    def search(self, query, k=10, n_probes=4):
        q_arr = np.array(query, dtype=np.float32)
        q_ptr = q_arr.ctypes.data_as(POINTER(c_float))
        
        ids = (c_longlong * k)()
        dists = (c_float * k)()
        
        count = lib.ivf_pq_search(self.obj, q_ptr, k, n_probes, ids, dists)
        
        result_ids = []
        result_dists = []
        for i in range(count):
            result_ids.append(ids[i])
            result_dists.append(dists[i])
        return result_ids, result_dists

    def save(self, filename: str):
        b_filename = filename.encode('utf-8')
        lib.ivf_pq_save(self.obj, b_filename)

    @staticmethod
    def load(filename: str):
        b_filename = filename.encode('utf-8')
        handle = lib.ivf_pq_load(b_filename)
        # Read dim
        with open(filename, 'rb') as f:
            import struct
            f.read(4)
            dim = struct.unpack('i', f.read(4))[0]
        return IVFPQIndex(dim, handle=handle)





