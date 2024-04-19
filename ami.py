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
from sklearn.metrics.cluster import adjusted_mutual_info_score

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

if __name__=="__main__":
    file_1 = None
    file_2 = None
    for i in range(1, len(sys.argv) ):
        if sys.argv[i] == "--file-1":
            file_1 = str(sys.argv[i+1])
        elif sys.argv[i] == "--file-2":
            file_2 = str(sys.argv[i+1])
    clust_1_lst = read_clust_lst(file_1)
    clust_2_lst = read_clust_lst(file_2)

    clust_1_asn = clust_lst_to_asn(clust_1_lst)
    clust_2_asn = clust_lst_to_asn(clust_2_lst)
    print(adjusted_mutual_info_score(clust_1_asn, clust_2_asn))

