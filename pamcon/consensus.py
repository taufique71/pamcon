import numpy as np
import scipy.sparse as sp

def consensus(graph, clusterings, niter=100, verbose=False):
    """
    Compute consensus clustering from multiple input clusterings.

    Parameters
    ----------
    graph : networkx.Graph or scipy.sparse matrix
        The input graph. If a NetworkX graph, nodes can have any hashable IDs.
        If a scipy sparse matrix, nodes are assumed to be 0-indexed integers.
    clusterings : list of dict or numpy.ndarray of shape (n, k)
        Input clusterings. Two formats are accepted:
        - If graph is a NetworkX graph: list of k dicts, each mapping node ID -> cluster ID (int).
        - If graph is a scipy sparse matrix: 2D numpy array of shape (n, k) where each
          column is one clustering and values are integer cluster IDs.
    niter : int, optional
        Maximum number of iterations (default: 100).
    verbose : bool, optional
        Print per-iteration progress (default: False).

    Returns
    -------
    dict
        Consensus clustering as a dict mapping node ID -> cluster ID.
    """
    from . import _core

    if niter < 1:
        raise ValueError("niter must be >= 1.")

    # --- Handle NetworkX graph ---
    try:
        import networkx as nx
        if isinstance(graph, nx.Graph):
            nodes = list(graph.nodes())
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            idx_to_node = nodes  # idx_to_node[i] = original node
            n = len(nodes)
            A = nx.to_scipy_sparse_array(graph, nodelist=nodes, format="csc", dtype=np.float64)
        else:
            node_to_idx = None
            idx_to_node = None
            A = graph
    except ImportError:
        node_to_idx = None
        idx_to_node = None
        A = graph

    # --- Handle scipy sparse matrix ---
    if not isinstance(A, sp.csc_matrix) and not isinstance(A, sp.csc_array):
        A = sp.csc_matrix(A)

    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Graph adjacency matrix must be square, got shape {A.shape}.")

    # Symmetrize (make undirected) — algorithm requires undirected graph
    A = A + A.T
    A = sp.csc_matrix(A)

    col_ptr = A.indptr.astype(np.uint32)
    row_ids = A.indices.astype(np.uint32)
    values  = A.data.astype(np.uint32)

    # --- Handle clusterings ---
    if isinstance(clusterings, np.ndarray):
        # scipy sparse path: clusterings is a 2D array of shape (n, k)
        if clusterings.ndim != 2:
            raise ValueError(
                f"When passing a numpy array, clusterings must be 2D of shape (n, k), "
                f"got shape {clusterings.shape}."
            )
        if clusterings.shape[0] != n:
            raise ValueError(
                f"clusterings has {clusterings.shape[0]} rows but graph has {n} nodes."
            )
        if clusterings.shape[1] < 2:
            raise ValueError("At least 2 input clusterings are required.")
        C = clusterings.astype(np.uint32)
    else:
        # NetworkX path: clusterings is a list of k dicts
        if len(clusterings) < 2:
            raise ValueError("At least 2 input clusterings are required.")
        k = len(clusterings)
        C = np.zeros((n, k), dtype=np.uint32)
        for j, clustering in enumerate(clusterings):
            if len(clustering) != n:
                raise ValueError(
                    f"Clustering {j} has {len(clustering)} entries but graph has {n} nodes."
                )
            for node, cluster_id in clustering.items():
                idx = node_to_idx[node] if node_to_idx is not None else int(node)
                C[idx, j] = np.uint32(cluster_id)

    # --- Call C++ binding ---
    result = _core.consensus(col_ptr, row_ids, values, n, n, C, niter, verbose)

    # --- Map result back to original node IDs ---
    if idx_to_node is not None:
        return {idx_to_node[i]: int(result[i]) for i in range(n)}
    else:
        return {i: int(result[i]) for i in range(n)}
