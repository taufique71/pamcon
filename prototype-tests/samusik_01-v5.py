import networkx as nx
import networkx.algorithms.community as nx_comm
from networkx.generators.community import LFR_benchmark_graph
from networkx.algorithms import bipartite
import numpy as np
import scipy as sp
from scipy.sparse import coo_array
from scipy import sparse
from cdlib import algorithms
from cdlib import evaluation
import sklearn
from utils import *
from distances import *
from consensus import *
import math
import itertools
import random
import time
from pathlib import Path

cons_name = "v5"

def v5_consensus(P_list, niter=10, starting_partition=None, verbose=False):
    G = nx.Graph(P_list[0]["graph"])
    n = len(list(G.nodes()))
    k = len(P_list)
    print("Number of edges in G:", len(list(G.edges())))

    t1 = time.time()
    A = nx.to_scipy_sparse_array(G, format="coo")
    t2 = time.time()
    print("Time to get sparse matrix of the graph:", t2-t1)

    nz_rows = A.row 
    nz_cols = A.col
    
    t1 = time.time()
    P_list_asn = []
    c = np.zeros((n,k))
    for i in range(k):
        clust_lst = P_list[i]["partition"]
        clust_asn = clust_lst_to_asn(clust_lst)
        c[:,i] = np.array(clust_asn)
    t2 = time.time()
    print("Time to generate cluster assignment matrix:", t2-t1)
    
    Aw_rows = A.row
    Aw_cols = A.col
    Aw_vals = A.data
    nz_elems = []
    t1 = time.time()
    for i in range(len(nz_rows)):
        Aw_vals[i] = np.sum( c[nz_rows[i],:] == c[nz_cols[i],:] )
        nz_elems.append((nz_rows[i], nz_cols[i], Aw_vals[i]))
    Gw = nx.from_scipy_sparse_array(coo_array((Aw_vals, (Aw_rows, Aw_cols)), shape=(n, n)))
    nz_elems = sorted(nz_elems, key=lambda x: x[2], reverse=True)
    t2 = time.time()
    print("Time to generate weighted consensus graph:", t2-t1)
    print("Number of edges in Gw:", len(list(Gw.edges())))
    
    t1 = time.time()
    refined_partition = None
    if starting_partition:
        refined_partition = list(starting_partition)
    else:
        refined_partition = []
        for i in range(n):
            refined_partition.append([str(i)])
    
    refined_partition_map = clust_lst_to_map(refined_partition)
    items = list(refined_partition_map.keys())
    t2 = time.time()
    print("Time to initialize:", t2-t1)
    
    """
    tSearch = 0
    tUpdate = 0
    tMovement = 0
    count = 0
    it = 1
    while(it <= niter):
        opt_item = items[0]
        opt_deltaS = 0
        opt_a = refined_partition_map[items[0]]
        opt_b = refined_partition_map[items[0]]
        opt_x = int(opt_item)
        flag = False
        for i in range(len(nz_elems)):
            if verbose:
                print("nz_elems[", i, "]", nz_elems[i])
            t1 = time.time()
            
            x1 = nz_elems[i][0]
            x2 = nz_elems[i][1]
            
            a1 = refined_partition_map[str(x1)]
            a2 = refined_partition_map[str(x2)]
            
            Mx1a1 = 0
            Mx2a1 = 0
            Mx1a2 = 0
            Mx2a2 = 0
            for elem in refined_partition[a1]:
                if str(elem) != str(x1):
                    Mx1a1 = Mx1a1 + (k - 2 * np.sum( (c[int(x1),:] == c[int(elem),:]) ) )
                if str(elem) != str(x2):
                    Mx2a1 = Mx2a1 + (k - 2 * np.sum( (c[int(x2),:] == c[int(elem),:]) ) )
            for elem in refined_partition[a2]:
                if str(elem) != str(x1):
                    Mx1a2 = Mx1a2 + (k - 2 * np.sum( (c[int(x1),:] == c[int(elem),:]) ) )
                if str(elem) != str(x2):
                    Mx2a2 = Mx2a2 + (k - 2 * np.sum( (c[int(x2),:] == c[int(elem),:]) ) )
                    
            deltaS1 = Mx1a2 - Mx1a1
            deltaS2 = Mx2a1 - Mx2a2
            x = None
            item = None
            b = None
            deltaS = None
            if (deltaS1 < deltaS2):
                x = x1
                item = str(x1)
                a = a1
                b = a2
                deltaS = deltaS1
            else:
                x = x2
                item = str(x2)
                a = a2
                b = a1
                deltaS = deltaS2
            t2 = time.time()
            tSearch = tSearch + t2-t1
            if (deltaS is not None) and (deltaS < 0) and (a != b):
                opt_item = item
                opt_deltaS = deltaS
                opt_a = a
                opt_b = b
                opt_x = x

                if verbose:
                    print("---")
                    print("Move Count:", count+1, "Optimum move results in", opt_deltaS)
                    print("Move:", opt_item)
                    print("From", opt_a, ":", refined_partition[opt_a])
                    print("To", opt_b, ":", refined_partition[opt_b])
                
                t1 = time.time()
                refined_partition[opt_a].remove(opt_item)
                refined_partition[opt_b].append(opt_item)
                refined_partition_map[opt_item] = opt_b
                t2 = time.time()
                tMovement = tMovement + (t2-t1)
                if verbose:
                    print("---")
            
                count = count + 1
                flag = True
        if flag == False:
            break
        print("Iteration:", it)
        print("Move count:", count)
        it = it + 1
    print("Time to search moves:", tSearch)
    print("Time to update M:", tUpdate)
    """
    
    t1 = time.time()
    empty_clusters = []
    for i in range(len(refined_partition)):
        if len(refined_partition[i]) == 0:
            empty_clusters.append(i)
            
    empty_clusters.sort(reverse=True)
    for e in empty_clusters:
        del refined_partition[e]
    t2 = time.time()
    print("Time to delete empty partitions:", t2-t1)
    
    return {"graph": nx.Graph(Gw), "partition": list(refined_partition)}

n = 53173
expected_clusters = []
for i in range(4):
    expected_clusters.append(random.randint(int(n ** (1. / 3)),3*int(n ** (1. / 2))))
    
alg_params = {
    "label_propagation": None,
    "leiden": None,
    "significance_communities": None,
    "surprise_communities": None,
    "greedy_modularity": None,
    "paris": None,
    "louvain": {
        "resolution": [0.75, 1.0, 1.25, 1.5],
        "randomize": [314159, 2718]
    },
    "infomap": None,
    "walktrap": None,
    "markov_clustering": {
        "inflation": [1.2, 1.5, 2, 2.5],
        "pruning_threshold": [0.01, 0.001],
        "convergence_check_frequency": [100]
    },
    "em": {
        "k": list(expected_clusters)
    },
    "sbm_dl": None,
    "spinglass": {
        "spins": list(expected_clusters)
    },
    "ricci_community": {
        "alpha": [0.3, 0.5, 0.6, 0.75]
    }
}
clustering_enumeration = []
count = 0
for alg, params in alg_params.items():
    param_combinations = []
    param_names = []
    if params is not None:
        iterables = []
        param_names = []
        for param in params.keys():
            iterables.append(list(params[param]))
            param_names.append(param)
        param_combinations = list(itertools.product(*iterables))
    if len(param_combinations) > 0:
        for param_combination in param_combinations:
            expr = "algorithms."+alg+"(G"
            for i in range(len(param_names)):
                expr = expr + "," + param_names[i] + "=" + str(param_combination[i])
            expr = expr + ")"
            clustering_enumeration.append((expr,count))
            count = count + 1      
    else:
        expr = "algorithms."+alg+"(G)"
        clustering_enumeration.append((expr,count))
        count = count + 1
        
print(clustering_enumeration)

fileprefix = "/home/mth/Data/UNC DATASET/Metis Format/"
fname = "Samusik_01NetworkMetis"
#graph_file = fileprefix + fname + ".edgelist"
graph_file = fileprefix + fname + ".mtx"
G = None
print(graph_file)
P_list = []
if Path(graph_file).is_file():
    with open(graph_file) as f:
        G = nx.from_scipy_sparse_array(spio.mmread(f), create_using=nx.Graph)
        coms = None
        for k in clustering_enumeration:
            clust_file = fileprefix + fname + "." + str(k[1])
            if Path(clust_file).is_file():
                partition = read_clust_lst(clust_file)
                P_list.append({"graph": nx.Graph(G), "partition": list(partition)})
        t1 = time.time()
        P_star = v5_consensus(P_list, niter=1, starting_partition=None, verbose=False)
        t2 = time.time()
        print("Number of clusters", len(P_star["partition"]))
        print("Time:", t2-t1)
        #write_clust_lst(P_star["partition"], fileprefix + fname + "." + cons_name)
