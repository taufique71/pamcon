# pamcon: Parallel Median Consensus Clustering in Complex Networks

Implementation of our work published in Nature Scientific Reports:
- Paper: https://www.nature.com/articles/s41598-025-87479-6
- ArXiv preprint: https://arxiv.org/pdf/2408.11331

---

## Python Package (Recommended)

### Installation

```bash
git clone https://github.com/taufique71/pamcon.git
cd pamcon
pip install -e .
```

**Requirements:** Python >= 3.8, numpy, scipy, a C++ compiler with OpenMP support.

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
- **`example_large.py`** — Large-scale example: LFR benchmark graph (n=5000, mu=0.3), Louvain with varying resolution parameters, sequential vs parallel runtime comparison

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

- `--graph-file`: Input graph in Matrix Market format (`.mtx`)
- `--input-prefix`: Prefix for clustering files, expected as `<prefix>.0`, `<prefix>.1`, ..., `<prefix>.(k-1)`
- `--k`: Number of input clusterings
- `--output-prefix`: Output file prefix — consensus written to `<prefix>.soln`
- `--niter`: Maximum iterations (default: 100)
- `--verbose`: Print per-iteration progress

Control threads via `OMP_NUM_THREADS`:
```bash
OMP_NUM_THREADS=8 ./consensus --graph-file graph.mtx ...
```

Run `./consensus --help` for full usage.

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
