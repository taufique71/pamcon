import random
from scipy.sparse import csr_matrix, csc_matrix
from numpy import linalg as LA
import networkx as nx
import networkx.algorithms.community as nx_comm
from networkx.generators.community import LFR_benchmark_graph
from networkx.algorithms import bipartite
import numpy as np
import scipy as sp
from scipy import sparse
import pandas as pd
import uunet.multinet as ml
#import ClusterEnsembles as CE
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from pathlib import Path
import json
from subprocess import Popen, PIPE
import subprocess
import time
import sys
import itertools
import random
import math
from utils import *
from distances import *

def gen_batches(P_list, batch_size):
    P_list_shuffled = list(P_list)
    random.shuffle(P_list_shuffled)
    n_batches = math.ceil(len(P_list_shuffled) * 1.0 / batch_size)
    batches = []
    for i in range(n_batches):
        if i < n_batches - 1:
            batches.append(P_list_shuffled[i * batch_size : (i+1) * batch_size])
        else:
            batches.append(P_list_shuffled[i * batch_size : ])
    return batches

def remove_inter_cluster_edges(G, clust_lst):
    clust_map = clust_lst_to_map(clust_lst)
    #print(clust_map)
    ebunch = []
    for e in G.edges:
        if e[0] in clust_map:
            if clust_map[e[0]] != clust_map[e[1]]:
                ebunch.append(e)
        else:
            if clust_map[str(e[0])] != clust_map[str(e[1])]:
                ebunch.append(e)
    G.remove_edges_from(ebunch)
    return G

def col_stochastic(A):
    col_sum = np.sum(A, axis=0)
    A = A / col_sum
    return A
# col_stochastic(np.array([[1,2,3], [4,5,6], [7,8,9]]))

def chaos(A):
    colmaxs = np.max(A, axis=0)
    A = np.multiply(A,A)
    colssqs = np.sum(A, axis=0)
    colmaxs = colmaxs - colssqs;
    A[A.nonzero()] = 1.0
    nnz_col = np.sum(A, axis=0)
    colmaxs = np.multiply(colmaxs, nnz_col)
    colmaxs[colmaxs < 0.0] = 0.0
    return np.sum(colmaxs)


def markov_diffusion(A, r=2.0, prune_threshold=0.001, n_iter=1000):
    colmaxs = np.max(A, axis=0) # Get maximum of all the edge weights going from each vertex
    Diag = np.diag(colmaxs)
    A = np.maximum.reduce([A, Diag])
    c = 1.0;
    EPS = 0.001
    for it in range(n_iter):
        A_sparse = csr_matrix(A)
        #print(A_sparse.getnnz())
        ##A_sparse = A_sparse.toarray()
        ##print("Delta(A, A_sparse)", LA.norm(A-A_sparse))
        A_sparse = A_sparse @ A_sparse
        A = A_sparse.toarray()
        #A = np.matmul(A,A) # Expand flows
        #print("Delta(A, A_sparse)", LA.norm(A-A_sparse))
        A = col_stochastic(A) # Normalize
        c = chaos(A) 
        A = np.power(A, r) # Inflate flows
        A = col_stochastic(A) # Normalize
        A[A < prune_threshold] = 0.0
        if c < EPS:
            break
    return A

def markov_consensus(P_star, P_list):
    #global axes
    #global axr
    # Accumulate input graphs and clustering structures to form a new accumulated graph
    G_star_temp = remove_inter_cluster_edges(P_star["graph"], P_star["partition"]) # Remove inter-cluster edges
    A_star_temp = nx.to_numpy_array(G_star_temp) # Entry at (i,j) position will have weight of edge i to j
    
    node_list = list(G_star_temp.nodes())
    node_map = {}
    for i in range(len(node_list)):
        node_map[i] = node_list[i]
        
    A_star_temp = A_star_temp.transpose() # Entry at (i,j) position will have weight of edge j to i
    A_star_temp = col_stochastic(A_star_temp) # Normalize edge weights going from each vertex
    
    #axes[axr][0].imshow(A_star_temp)
    #axes[axr][0].set_title("A_markov(it-1)")
    
    Acc = P_star["weight"] * A_star_temp # Scale each edge with provided weights
    for i in range(len(P_list)):
        G_temp = remove_inter_cluster_edges(P_list[i]["graph"], P_list[i]["partition"]) # Remove inter-cluster edges
        A_temp = nx.to_numpy_array(G_temp) # Entry at (i,j) position will have weight of edge i to j
        A_temp = A_temp.transpose() # Entry at (i,j) position will have weight of edge j to i
        A_temp = col_stochastic(A_temp) # Normalize edge weights going from each vertex
        
        #axes[axr][1].imshow(A_temp)
        #axes[axr][1].set_title("A(it)")
        
        Acc = Acc + P_list[i]["weight"] * A_temp # Scale each edge with provided weights
    Acc = col_stochastic(Acc)
    
    #axes[axr][2].imshow(Acc)
    #axes[axr][2].set_title("Acc(it)")
    
    A_markov = markov_diffusion(Acc, 1.2, 0.0001, 5)
    A_markov = A_markov.transpose()
    #G_markov = nx.from_numpy_array(A_markov, create_using=nx.DiGraph)
    G_markov = nx.DiGraph()
    G_markov.add_nodes_from(node_list)
    ebunch = []
    x = np.nonzero(A_markov)
    for xi in range(len(x[0])):
        ri = x[0][xi]
        ci = x[1][xi]
        ebunch.append((node_map[ri], node_map[ci], {"weight": A_markov[ri,ci]}))
    G_markov.add_edges_from(ebunch)
    
    #nx.relabel_nodes(G_markov, node_map)
    cc_list = nx.weakly_connected_components(G_markov)
    
    #axr = axr + 1
    
    P = {}
    P["graph"] = G_markov
    P["partition"] = []
    for c in cc_list:
        cluster = []
        for elem in c:
            cluster.append(str(elem))
        P["partition"].append(cluster)
    return P

def iterative_consensus(P_list, n_iter=10, batch_size=1, distance=split_joint_distance, batch_consensus=markov_consensus):
    k = len(P_list)
    n = 0
    for cluster in P_list[0]["partition"]:
        n =  n + len(cluster)
    
    # Convert each input graph to a directed graph
    # Reason: Dynamics of directed graphs would be employed for consensus generation
    for i in range(k):
        if P_list[i]["graph"].is_directed() == False:
            P_list[i]["graph"] = P_list[i]["graph"].to_directed()
     
    # If no initial solution is given randomly pick an item from C as initial solution
    random.seed(123)
    P_star = random.choice(P_list)
    
    P_prev = P_star
        
    for it in range(n_iter):
        #global axr
        #axr = 0
        print("***")
        print("Iteration:", it)
        print("***")
        
        P_list_working = []
        dist = []
        # Calculate the distance between current solution and all inputs
        for i in range(k):
            d = distance(P_list[i]["partition"], P_star["partition"])
            item = P_list[i]
            item["dist"] = d
            dist.append(d)
            P_list_working.append(item)
        
        dist = np.array(dist)
        dist_total = np.sum(dist)
        dist_mean = np.mean(dist)
        dist_std = np.std(dist)
        dist_med = np.median(dist)
        
        #w_star = ( (iter+1) * 1.0 ) / n_iter # Give this weight to current solution
        #w_rest = 1 - w_star
        
        #w_star = np.exp(-1 * dist_std)
        #w_rest = 1 - w_star
        
        #w_star = 1.0 / (k + 1)
        #w_rest = k * 1.0 / (k + 1)
        
        w_star = (1.0) / (k + 1) + (k * 1.0 / (k + 1)) * np.exp(-1 * dist_std)
        w_rest = 1.0 - w_star
        
        #dist_rstrd = np.array(dist)
        #for i in range(len(dist)):
        #    dist_rstrd[i] = np.exp(-1 * (np.absolute(dist[i] - dist_med))) 
        #dist_rstrd_total = np.sum(dist_rstrd)
        #w_star = (1.0) / (k + 1) + (k * 1.0 / (k + 1)) * np.exp(-1 * dist_std)
        #w_rest = 1.0 - w_star
        
        print("dist_total:", dist_total, ", dist_mean:", dist_mean, ", dist_std:", dist_std, ", w_star:", w_star, ", w_rest:", w_rest)
        #print("dist_rstrd_total:", dist_rstrd_total)
        # Normalize all the distances to get weight
        for i in range(len(P_list_working)):
            P_list_working[i]["weight"] = (P_list_working[i]["dist"] / dist_total) * w_rest
            
        # Normalize all the distances to get weight
        #for i in range(len(P_list_working)):
        #    P_list_working[i]["weight"] = (dist_rstrd[i] / dist_rstrd_total) * w_rest
        
        batches = gen_batches(P_list_working, batch_size)
        for b in range(len(P_list_working)):
            P_list_batch = batches[b]
            P_star["weight"] = w_star
            P_star = batch_consensus(P_star, P_list_batch)
            d = distance(P_star["partition"], P_prev["partition"])
            P_prev = P_star
            print("batch: ", b, ": distance to previous solution:", d, ", #cluster:", len(P_star["partition"]))
    return P_star

def lf_consensus(P_list):
    k = len(P_list)
    n = 0
    for cluster in P_list[0]["partition"]:
        n =  n + len(cluster)
    adj_mat = np.zeros((n,n))
    for P in P_list:
        for clust in P["partition"]:
            for i in range(len(clust)):
                for j in range(i+1, len(clust)):
                    idxi = int(clust[i])
                    idxj = int(clust[j])
                    adj_mat[idxi, idxj] = adj_mat[idxi, idxj] + 1
    Ga = nx.from_numpy_matrix(adj_mat)
    clust_lst = nx_comm.louvain_communities(Ga, seed=123)
    P_star = { "graph": nx.Graph(Ga), "partition": list(clust_lst)}
    return P_star

"""
def hbgf_solution(P_list):
    k = len(P_list)
    n = 0
    label_matrix = np.full((n, k), -1)
    for e in range(k):
        P = P_list[e]
        clust_asn = clust_lst_to_asn(P["partition"], nelem=n)
        label_matrix[:,e] = np.array(clust_asn)
    cons_asn = CE.cluster_ensembles(np.transpose(label_matrix), solver="hbgf")
    cons_lst = clust_asn_to_lst(cons_asn)
    P_star = {"graph": None, "partition": list(cons_lst)}
    return P_star
"""

