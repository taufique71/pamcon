import math
from emd import emd
import numpy as np
import scipy as sp
from sklearn.metrics.cluster import rand_score
import pyvoi # pip install python-voi
from utils import *

"""
F-score calculation in the same way as Wikipedia page: https://en.wikipedia.org/wiki/F-score
Pair of datapoints is considered to determine TP, TN, FN and FP \
- TP: Number of pairs that have been clustered together both in ground-truth and candidate
- FN: Number of pairs that have been cluster together in ground-truth but not in candidate
- FP: Number of pairs that have been clustered together in candidate but not in ground-truth
- TN: Number of pairs that have been clustered separately both in ground-truth and candidate
"""
def fscore(ground_truth, cluster_to_compare):
    gt_clust_asn = clust_lst_to_asn(ground_truth)
    cand_clust_asn = clust_lst_to_asn(cluster_to_compare)
    n = len(gt_clust_asn)
    TP = 0 
    TN = 0 
    FN = 0 
    FP = 0 
    
    for i in range(n):
        for j in range(i+1, n):
            if (gt_clust_asn[i] == gt_clust_asn[j]) and (cand_clust_asn[i] == cand_clust_asn[j]):
                # Number of pairs that have been clustered together both in ground-truth and candidate
                TP = TP + 1
            elif (gt_clust_asn[i] == gt_clust_asn[j]) and (cand_clust_asn[i] != cand_clust_asn[j]):
                # Number of pairs that have been cluster together in ground-truth but not in candidate
                FN = FN + 1
            elif (gt_clust_asn[i] != gt_clust_asn[j]) and (cand_clust_asn[i] == cand_clust_asn[j]):
                # Number of pairs that have been clustered together in candidate but not in ground-truth
                FP = FP + 1
            elif (gt_clust_asn[i] != gt_clust_asn[j]) and (cand_clust_asn[i] != cand_clust_asn[j]):
                # Number of pairs that have been clustered separately both in ground-truth and candidate
                TN = TN + 1
            j = j + 1
        i = i + 1
        
    precision = 0.0
    recall = 0.0
    F = 0
    if TP > 0:
        precision = TP * 1.0 / (TP + FP)
        recall = TP * 1.0 / (TP + FN)
        F = 2 * TP * 1.0 / (2 * TP * 1.0 + FP * 1.0 + FN * 1.0)
    return (F, precision, recall)

def earth_movers_distance(clust_lst_1, clust_lst_2):
    p1_n = 0
    p1_temp = {}
    for i in range(len(clust_lst_1)):
        p1_n = p1_n + len(clust_lst_1[i])
        p1_temp["p1_"+str(i)] = list(clust_lst_1[i])
        
    emd_X = []
    emd_X_weight = []
    for k, v in p1_temp.items():
        emd_X.append(k)
        emd_X_weight.append(len(v)*1.0/p1_n)
        
    p2_n = 0
    p2_temp = {}
    for i in range(len(clust_lst_2)):
        p2_n = p2_n + len(clust_lst_2[i])
        p2_temp["p2_"+str(i)] = list(clust_lst_2[i])
    emd_Y = []
    emd_Y_weight = []
    for k, v in p2_temp.items():
        emd_Y.append(k)
        emd_Y_weight.append(len(v)*1.0/p2_n)
        
    d = []
    for kx, vx in p1_temp.items():
        arr = []
        for ky, vy in p2_temp.items():
            set_1 = set(vx)
            set_2 = set(vy)
            isec = set_1.intersection(set_2)
            un = set_1.union(set_2)
            d_vx_vy = (len(un)-len(isec))*1.0
            arr.append(d_vx_vy)
        d.append(arr)
    a, f = emd(emd_X, emd_Y, X_weights=np.array(emd_X_weight), Y_weights=np.array(emd_Y_weight), distance='precomputed',
        D=np.array(d), return_flows=True)
    return a

"""
- Split-Joint distance proposed by Stijn van Dongen in his MCL thesis. 
- This distance is a true metric that signifies the number of elements needed to move to convert one clustering to another.
- Paper: Performance criteria for graph clustering and Markov cluster
experiments (https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.26.9783&rep=rep1&type=pdf)
"""
def split_joint_distance(A, B):
    n = 0
    C = []
    for a in A:
        n = n + len(a)
        for b in B:
            set_a = set(a)
            set_b = set(b)
            isec = set_a.intersection(set_b)
            if len(isec) != 0:
                C.append(list(isec))
                
    proj_A_B = 0
    for a in A:
        max_isec_card = 0
        for c in C:
            set_a = set(a)
            set_c = set(c)
            isec = set_a.intersection(set_c)
            if len(isec) > max_isec_card:
                max_isec_card = len(isec)
        proj_A_B = proj_A_B + max_isec_card
    
    proj_B_A = 0
    for b in B:
        max_isec_card = 0
        for c in C:
            set_b = set(b)
            set_c = set(c)
            isec = set_b.intersection(set_c)
            if len(isec) > max_isec_card:
                max_isec_card = len(isec)
        proj_B_A = proj_B_A + max_isec_card
    
    dist_A_B = 2 * n - proj_A_B - proj_B_A
    return dist_A_B
# print(split_joint_distance([[1,2,3,4], [5,6,7], [8,9,10,11,12]], [[2,4,6,8,10], [3,9,12], [1,5,7], [11]]))

def mirkin_distance(A, B):
    A_clust_asn = clust_lst_to_asn(A)
    B_clust_asn = clust_lst_to_asn(B)
    n = len(A_clust_asn)
    R = rand_score(A_clust_asn, B_clust_asn)
    M = (n * (n-1))*(1-R)
    return M
# print(mirkin_distance([[1,2,3,4], [5,6,7], [8,9,10,11,12]], [[2,4,6,8,10], [3,9,12], [1,5,7], [11]]))

def variation_of_info_distance(A, B):
    A_clust_asn = clust_lst_to_asn(A)
    B_clust_asn = clust_lst_to_asn(B)
    n = len(A_clust_asn)
    vi,vi_split,vi_merge=pyvoi.VI(A_clust_asn,B_clust_asn)
    return vi.item() #vi is a tensor
#print(variation_of_info_distance([[1,2,3,4], [5,6,7], [8,9,10,11,12]], [[2,4,6,8,10], [3,9,12], [1,5,7], [11]]))
