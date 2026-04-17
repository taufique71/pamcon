import os
import time
import numpy as np
import scipy.sparse as sp
import scipy.io
import pamcon
import pamcon._core as _core

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "prototype-tests/LFR/n5000")
PREFIX     = os.path.join(DATA_DIR, "LFR_n5000_mu04_gamma30_beta11")
K          = 16   # number of input clusterings to use (.0 through .15)

# --- Load graph as scipy sparse matrix ---
print("Loading LFR benchmark graph (n=5000, mu=0.4) ...")
A = scipy.io.mmread(PREFIX + ".mtx")
A = sp.csc_matrix(A, dtype=np.float64)
n = A.shape[0]
print(f"  {n} nodes, {A.nnz} edges")

# --- Load input clusterings ---
# Each clustering file: one cluster per line, space-separated 0-indexed vertex IDs
# Convert to 1D array of cluster assignments, then stack into [n x k] array
def read_cluster_file(fname, n):
    clust_asn = np.zeros(n, dtype=np.uint32)
    with open(fname) as f:
        for cid, line in enumerate(f):
            for node in line.split():
                clust_asn[int(node)] = cid
    return clust_asn

print(f"Loading {K} input clusterings ...")
C = np.zeros((n, K), dtype=np.uint32)
for i in range(K):
    fname = f"{PREFIX}.{i}"
    C[:, i] = read_cluster_file(fname, n)
    print(f"  Clustering {i}: {C[:, i].max() + 1} clusters")

# --- Run consensus: sequential (1 thread) ---
print(f"\nRunning consensus with 1 thread ...")
_core.set_num_threads(1)
t0 = time.perf_counter()
result_seq = pamcon.consensus(A, C, niter=100, verbose=False)
t1 = time.perf_counter()
print(f"  Consensus: {max(result_seq.values()) + 1} clusters")
print(f"  Runtime (1 thread):  {t1 - t0:.3f} seconds")

# --- Run consensus: parallel (8 threads) ---
print(f"\nRunning consensus with 8 threads ...")
_core.set_num_threads(8)
t0 = time.perf_counter()
result_par = pamcon.consensus(A, C, niter=100, verbose=False)
t1 = time.perf_counter()
print(f"  Consensus: {max(result_par.values()) + 1} clusters")
print(f"  Runtime (8 threads): {t1 - t0:.3f} seconds")
