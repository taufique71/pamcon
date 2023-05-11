import math
from emd import emd
import numpy as np
import scipy as sp

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
