import os
from pathlib import Path
import networkx as nx
import networkx.algorithms.community as nx_comm
from networkx.generators.community import LFR_benchmark_graph
from networkx.algorithms import bipartite
import numpy as np
from scipy.sparse import coo_array
from scipy import sparse
import scipy as sp
from scipy import io as spio
import json
import subprocess
import time
import sys
import math
import itertools
import random
import pyintergraph

from cdlib import algorithms
from cdlib import evaluation

def read_clust_lst(filename, integer_node_id=False):
    lst = []
    with open(filename, mode="r") as infile:
        lines = infile.readlines()
        for l in lines:
            cl = []
            toks = l.split(" ")
            for t in toks:
                if t.isspace() == False:
                    if integer_node_id:
                        cl.append(int(t))
                    else:
                        cl.append(t)
            lst.append(cl)
    return lst

def write_clust_lst(clust_lst, filename):
    with open(filename, mode="w") as outfile:
        for lst in clust_lst:
            if len(lst) > 0:
                for e in lst:
                    outfile.write(str(e))
                    outfile.write(" ")
                outfile.write("\n")

def write_clust_asn(clust_asn, filename):
    with open(filename, mode="w") as outfile:
        for s in clust_asn:
            outfile.write("%s\n" % s)

def write_label_map(label_map, filename):
    with open(filename, mode="w") as outfile:
        for k in label_map.keys():
            line = str(k) + " " + str(label_map[k]) + "\n"
            outfile.write(line)

def read_label_map(filename, reverse=False):
    label_map = {}
    with open(filename, mode="r") as infile:
        lines = infile.readlines()
        for l in lines:
            l = l.strip()
            toks = l.split(" ")
            # print(toks)
            if reverse:
                label_map[toks[1]] = toks[0]
            else:
                label_map[toks[0]] = toks[1]
    return label_map

def clust_asn_to_lst(clust_asn):
    clust_lst = {}
    for i in range(len(clust_asn)):
        if clust_asn[i] not in clust_lst.keys():
            clust_lst[clust_asn[i]] = []
        clust_lst[clust_asn[i]].append(i)
    clust_lst = list(clust_lst.values())
    for i in range(len(clust_lst)):
        clust_lst[i] = set(clust_lst[i])
    return clust_lst

# print(clust_asn_to_lst([0, 0, 1, 0, 2, 1]))

def clust_lst_to_asn(clust_lst, nelem=None):
    
    if nelem == None:
        nelem = 0
        for l in clust_lst:
            nelem = nelem + len(l)   
    
    clust_map = clust_lst_to_map(clust_lst)
    keys = list(clust_map.keys())
    keys.sort(key=int)
    
    clust_asn = [-1] * nelem
    i = 0
    while i < nelem:
        clust_asn[i] = clust_map[keys[i]]
        i = i + 1
        
    return clust_asn

# print(clust_lst_to_asn([[0,1,3], [2,5], [4]]))

def clust_lst_to_map(clust_lst, nelem=None):
    clust_map = {}
    for l in range(len(clust_lst)):
        for e in clust_lst[l]:
            clust_map[e] = l
    return clust_map

# print(clust_lst_to_map([[0,1,3], [2,5], [4]]))

def read_matrix_market(filename):
    G = None
    with open(filename) as f:
        G = nx.from_scipy_sparse_matrix(spio.mmread(f))
    return G

def write_matrix_market(filename, G):
    A = nx.to_numpy_array(G)
    SA = csr_matrix(A)
    spio.mmwrite(filename, SA)
    return

def prep_consensus_graph(partitions):
    n = len(partitions[0])
    k = len(partitions)
    #print("Number of nodes", n)
    
    row = []
    col = []
    val = []

    for x in partitions:
        partition = clust_asn_to_lst(x)
        for elemset in partition:
            cluster = list(elemset)
            for i in range(len(cluster)):
                for j in range(i+1, len(cluster)):
                    item_1 = cluster[i]
                    item_2 = cluster[j]
                    row.append(int(item_1))
                    col.append(int(item_2))
                    val.append(int(1))
                    
    r = coo_array((val, (row, col)), shape=(n, n))
    rDense = r.toarray()
    threshold = k / 2
    rDense[np.abs(rDense) < threshold] = 0
    
    G = nx.from_numpy_array(rDense)
    return G

# alg_params = {
    # "louvain": {
        # "resolution": [0.75, 1.0, 1.25, 1.5],
        # "randomize": [314159, 2718, 98765, 12345]
    # }
# }

alg_params = {
    "louvain": {
        "resolution": [1.0],
        "randomize": [314159, 
                      2718, 
                      98765, 
                      12345, 
                      20181218,
                      20181228,
                      20190103, 
                      20190107,
                      20190322,
                      20190831,
                      20200311,
                      20200329,
                      20210507,
                      20220104,
                      20230225,
                      20240402
                     ]
    }
}

def lf_louvain(partitions):
    G = prep_consensus_graph(partitions)

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
            
    # print(clustering_enumeration)
    
    new_adj_mat = nx.to_numpy_array(G)
    old_adj_mat = np.zeros(new_adj_mat.shape)
    diff_mat = old_adj_mat - new_adj_mat
    old_adj_mat = np.array(new_adj_mat)
    norm = np.linalg.norm(diff_mat)

    for it in range(20):
        print(it, "norm:", norm)
        partitions = []
        for k in clustering_enumeration:
            try:
                coms = eval(k[0])
                coms_asn = clust_lst_to_asn(list(coms.communities))
                partitions.append(coms_asn)
                # partitions.append(list(coms.communities))
                # print(coms.communities)
            except:
                print("UNSUCCESSFUL", k[0])
        G = prep_consensus_graph(partitions)
        new_adj_mat = nx.to_numpy_array(G)
        diff_mat = old_adj_mat - new_adj_mat
        norm = np.linalg.norm(diff_mat)
        old_adj_mat = np.array(new_adj_mat)
        if norm < 1e-3:
            # print("Converged")
            # print("---")
            break
    
    return partitions[0]

if __name__=="__main__":
    i = 1
    graphfile = ""
    input_clustering_prefix = ""
    k = 0
    output_prefix = ""
    
    for i in range(1, len(sys.argv) ):
        if sys.argv[i] == "--graph-file":
            graphfile = str(sys.argv[i+1])
        elif sys.argv[i] == "--input-prefix":
            input_clustering_prefix = str(sys.argv[i+1])
        elif sys.argv[i] == "--k":
            k = int(sys.argv[i+1])
        elif sys.argv[i] == "--output-prefix":
            output_prefix = str(sys.argv[i+1])

    print("Graph:", graphfile)
    print("Input clustering prefix:", input_clustering_prefix)
    print("Number of input clusterings:", k)
    print("Output file prefix:", output_prefix)
    
    G = nx.from_scipy_sparse_array(spio.mmread(graphfile), create_using=nx.Graph)
    G_igraph = pyintergraph.nx2igraph(G)
    partitions = []
    energies = []
    for i in range(k):
        clust_file = input_clustering_prefix + "." + str(i)
        partition_lst = read_clust_lst(clust_file)
        partition_asn = clust_lst_to_asn(partition_lst)
        partitions.append(partition_asn)
        # partition_lst = clust_asn_to_lst(partition_asn)
        # Q = nx.community.modularity(G, partition_lst)
        # energies.append(Q)
    cons_asn = lf_louvain(partitions)
    cons_lst = clust_asn_to_lst(cons_asn)
    # print(cons_lst)
    write_clust_lst(cons_lst, output_prefix + ".soln-0")

