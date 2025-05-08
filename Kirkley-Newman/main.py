import os
from pathlib import Path
import networkx as nx
import scipy as sp
from scipy import io as spio
import json
import subprocess
import time
import sys
import pyintergraph

# import pyximport; pyximport.install()
import kn

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
        partition_lst = clust_asn_to_lst(partition_asn)
        Q = nx.community.modularity(G, partition_lst)
        energies.append(Q)
    
    t0 = time.time()
    results = kn.partition_clustering(G_igraph,partitions,energies, num_mcs = 30,num_init_modes = 1,initial_runs = 10,\
                                    manual_mode_scale = -1.,dist_percentile = -1.,fix_K = 0,consecutive_rejects = 100,\
                                                            max_local_moves = 10,Lambda = 0.,graph_plots = False,node_size = 5,Kprior=1,MItype=1,mode_info_type=0)
    t1 = time.time()

    print("Time to find representatives:", t1-t0, "seconds")
    
    component_ranking = []
    for i in range(len(results[0])):
        component_id = i
        component_size = 0
        component_members = []
        for j in range(len(results[1])):
            if results[1][j] == results[0][i]:
                component_size = component_size + 1
                component_members.append(j)
        component_ranking.append( (component_id, component_size, component_members) )
    component_ranking.sort(key=lambda a: a[1], reverse=True)

    for i in range(len(results[0])):
        component_id = component_ranking[i][0]
        component_size = component_ranking[i][1]
        component_members = component_ranking[i][2]
        soln_idx = results[0][component_id]
        soln_asn = partitions[soln_idx]
        soln_lst = clust_asn_to_lst(soln_asn)
        write_clust_lst( soln_lst, output_prefix + "." + "soln-" + str(i) )
        print(len(soln_lst), "clusters in solution", i, "brings consensus of", component_size, "partitions");
        print("soln-"+str(i), "[", ", ".join(str(e) for e in component_members), "]")

