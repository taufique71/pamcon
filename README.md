# pamcon: Parallel Median Consensus Clustering in Complex Networks

Implementation of our work published in Nature Scientific Reports:
- Paper: https://www.nature.com/articles/s41598-025-87479-6
- ArXiv preprint: https://arxiv.org/pdf/2408.11331

---

## What is this?

Community detection algorithms are often non-deterministic — running the same algorithm multiple times, or different algorithms on the same graph, produces different results. **Consensus clustering** combines k input clusterings into a single stable result that minimizes the total Rand distance to all inputs.

**pamcon** is the first algorithm designed specifically for graphs that actively optimizes this objective. It uses a greedy local search heuristic that exploits graph connectivity structure, parallelized with OpenMP. The parallelism is essential — without it the algorithm is computationally intractable for large graphs.

See the paper for full algorithmic details and benchmarks.

---

## Python Package (Recommended)

### Installation

```bash
git clone https://github.com/taufique71/pamcon.git
cd pamcon
pip install -e .
```

**Requirements:** Python >= 3.8, numpy, scipy, a C++ compiler with OpenMP support.

| Platform | Support | Notes |
|----------|---------|-------|
| Linux    | Full    | Works out of the box. `g++` and OpenMP are standard. |
| macOS    | Full    | Requires `brew install libomp` before installing pamcon. |
| Windows  | Not supported | — |

For NetworkX support:
```bash
pip install -e ".[networkx]"
```

### Usage

```python
import pamcon
import networkx as nx

G = nx.karate_club_graph()

# Each clustering is a dict mapping node -> cluster ID
clusterings = [clustering_1, clustering_2, ..., clustering_k]

result = pamcon.consensus(G, clusterings)
# result is a dict mapping node -> consensus cluster ID
```

The `graph` argument accepts:
- A **NetworkX graph** (any node ID type) — pass clusterings as a list of k dicts mapping node ID → cluster ID
- A **scipy sparse matrix** (nodes assumed to be 0-indexed integers) — pass clusterings as a 2D numpy array of shape `(n, k)` where each column is one clustering

For large graphs using scipy sparse:

```python
import scipy.sparse as sp
import numpy as np
import pamcon

# A is a scipy sparse adjacency matrix of shape (n, n)
# C is a numpy array of shape (n, k) — each column is one clustering
result = pamcon.consensus(A, C)
```

### Controlling parallelism

```python
import pamcon._core as _core

_core.set_num_threads(8)   # use 8 threads
result = pamcon.consensus(G, clusterings)
```

### Examples

- **`example.py`** — Simple example using the karate club graph with label propagation and Louvain clusterings
- **`example_large.py`** — Demonstrates the scipy sparse interface on an LFR benchmark graph (n=5000, mu=0.4) with 16 input clusterings. Also shows the parallel speedup benefit:

```
Loading LFR benchmark graph (n=5000, mu=0.4) ...
  5000 nodes, 102571 edges
Loading 16 input clusterings ...
  Clustering 0: 1 clusters
  Clustering 1: 55 clusters
  Clustering 2: 309 clusters
  Clustering 3: 4999 clusters
  Clustering 4: 10 clusters
  Clustering 5: 87 clusters
  Clustering 6: 49 clusters
  Clustering 7: 61 clusters
  Clustering 8: 52 clusters
  Clustering 9: 56 clusters
  Clustering 10: 75 clusters
  Clustering 11: 71 clusters
  Clustering 12: 94 clusters
  Clustering 13: 95 clusters
  Clustering 14: 184 clusters
  Clustering 15: 156 clusters

Running consensus with 1 thread ...
  Consensus: 219 clusters
  Runtime (1 thread):  1.837 seconds

Running consensus with 8 threads ...
  Consensus: 219 clusters
  Runtime (8 threads): 0.202 seconds
```

9x speedup with 8 threads — same consensus result.

```bash
python example.py
python example_large.py
```

---

## Command-Line Interface

For users who prefer a CLI or work outside Python.

### Building

Requirements: `g++` with OpenMP support (no other dependencies).

```bash
make consensus
```

### Usage

```bash
./consensus --graph-file <graph.mtx> \
            --input-prefix <clustering_prefix> \
            --k <number_of_clusterings> \
            --output-prefix <output_prefix> \
            [--niter 100] \
            [--verbose]
```

### Parameters

| Argument | Required | Description |
|----------|----------|-------------|
| `--graph-file` | Yes | Input graph in Matrix Market format (`.mtx`) |
| `--input-prefix` | Yes | Path prefix for input clustering files (see format below) |
| `--k` | Yes | Number of input clusterings |
| `--output-prefix` | Yes | Path prefix for output file |
| `--niter` | No | Maximum number of iterations (default: 100) |
| `--verbose` | No | Print detailed progress per iteration |
| `--help` | No | Show usage message |

### Input clustering file format

Clustering files must be named `<prefix>.0`, `<prefix>.1`, ..., `<prefix>.(k-1)`.

Each file represents one clustering. Each **line** corresponds to one cluster and contains the **space-separated 0-indexed vertex IDs** belonging to that cluster:

```
0 4 7 12
1 3 9
2 5 6 8 10 11
```

### Output format

The consensus clustering is written to `<output-prefix>.soln` in the same format — one cluster per line, space-separated vertex IDs.

### Example

```bash
# Run consensus on a graph with 10 input clusterings using 8 threads
OMP_NUM_THREADS=8 ./consensus \
    --graph-file graph.mtx \
    --input-prefix clusterings/run \
    --k 10 \
    --output-prefix results/consensus
```

This expects files `clusterings/run.0` through `clusterings/run.9` and writes the result to `results/consensus.soln`.

---

## Research / Advanced CLI

The full research pipeline (with preprocessing, clustering grouping, and alternative algorithms) is available as a separate binary:

```bash
make consensus-research
```

This requires [igraph](https://igraph.org/c/doc/igraph-Installation.html) — update the paths in `Makefile` before building.

---

## Project Structure

```
pamcon/          ← Python package (pip installable)
consensus.h      ← Core algorithm: parallel_consensus_v8
main_user.cpp    ← CLI entry point (no external dependencies)
main.cpp         ← Research CLI entry point (requires igraph)
example.py       ← Simple usage example
example_large.py ← Large-scale benchmark example
```
