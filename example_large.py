import time
import networkx as nx
import pamcon
import pamcon._core as _core

# --- Generate LFR benchmark graph ---
print("Generating LFR benchmark graph (n=5000, mu=0.3) ...")
G = nx.LFR_benchmark_graph(
    n=5000,
    tau1=3,           # power-law exponent for degree distribution
    tau2=1.5,         # power-law exponent for community size distribution
    mu=0.3,           # mixing parameter (fraction of inter-community edges)
    average_degree=10,
    min_community=20,
    seed=42,
)
G = nx.Graph(G)  # strip node attributes — pamcon only needs the graph structure
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

# --- Generate clusterings using Louvain with different resolution parameters ---
# Resolution > 1 favours smaller communities, < 1 favours larger communities
resolutions = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
clusterings = []
print("Generating input clusterings (Louvain, varying resolution):")
for i, r in enumerate(resolutions):
    communities = nx.community.louvain_communities(G, resolution=r, seed=i)
    clustering = {node: cid for cid, c in enumerate(communities) for node in c}
    clusterings.append(clustering)
    print(f"  resolution={r:.2f} -> {max(clustering.values()) + 1} clusters")

# --- Run consensus: sequential (1 thread) ---
print(f"\nRunning consensus with 1 thread ...")
_core.set_num_threads(1)
t0 = time.perf_counter()
result_seq = pamcon.consensus(G, clusterings, niter=100, verbose=False)
t1 = time.perf_counter()
print(f"  Consensus: {max(result_seq.values()) + 1} clusters")
print(f"  Runtime (1 thread):  {t1 - t0:.3f} seconds")

# --- Run consensus: parallel (8 threads) ---
print(f"\nRunning consensus with 8 threads ...")
_core.set_num_threads(8)
t0 = time.perf_counter()
result_par = pamcon.consensus(G, clusterings, niter=100, verbose=False)
t1 = time.perf_counter()
print(f"  Consensus: {max(result_par.values()) + 1} clusters")
print(f"  Runtime (8 threads): {t1 - t0:.3f} seconds")
