import networkx as nx
import pamcon

# Load the karate club graph (34 nodes, 78 edges)
G = nx.karate_club_graph()

# Create 5 slightly different clusterings using label propagation
clusterings = []
for _ in range(5):
    communities = nx.community.label_propagation_communities(G)
    clustering = {}
    for cid, community in enumerate(communities):
        for node in community:
            clustering[node] = cid
    clusterings.append(clustering)

# Print input clusterings
for i, c in enumerate(clusterings):
    n_clusters = max(c.values()) + 1
    print(f"Clustering {i}: {n_clusters} clusters")

# Run consensus
result = pamcon.consensus(G, clusterings, niter=100, verbose=False)

print(f"\nConsensus: {max(result.values()) + 1} clusters")
print(f"Result: {result}")
