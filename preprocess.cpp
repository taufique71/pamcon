#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <igraph.h>

#include "CONPAS/MersenneTwister.h"
#include "CONPAS/RefinedClustering.h"
#include "CONPAS/SetPartition.h"
#include "CONPAS/SetPartitionVector.h"
#include "CONPAS/Utility.h"
#include "CONPAS/BestOfK.h"
#include "CONPAS/AdjBestOfK.h"
#include "CONPAS/MajorityRule.h"
#include "CONPAS/AverageLink.h"
#include "CONPAS/Matrix.h"
#include "CONPAS/CCPivot.h"
#include "CONPAS/CCAverageLink.h"

#include "GAP/pvector.h"
#include "GAP/timer.h"

#include "utils.h"
#include "COO.h"
#include "CSC.h"
#include "CSC_adder.h"

using namespace std;

/*
 * Elementwise equality operation
 * Returns a boolean vector
 * */
template<typename T>
vector<bool> ewise_equal(vector<T>& a, vector<T>& b){
    vector<bool> eq(a.size());
    for(auto i = 0; i < a.size(); i++){
        eq[i] = (a[i] == b[i]) ? true : false ;
    }
    return eq;
}

/*
 * Elementwise equality operation
 * Returns a boolean vector - true if the corresponding elements are not equal, false otherwise
 * */
template<typename T>
vector<bool> ewise_not_equal(vector<T>& a, vector<T>& b){
    vector<bool> eq(a.size());
    for(auto i = 0; i < a.size(); i++){
        eq[i] = (a[i] == b[i]) ? false : true ;
    }
    return eq;
}


/*
 * Reads clustering stored in the form of cluster list
 * Reads to a vector representing cluster assignment
 * Return number of clusters
 * */
uint32_t read_clust_lst(string fname, vector<uint32_t>& clust_asn){
    ifstream infile(fname);
    string line;
    uint32_t clust_id = 0;
    vector < vector<uint32_t> > clust_lst;
    uint32_t nelem = 0;
    while(getline(infile, line)){
        clust_lst.push_back( vector<uint32_t>() );
        istringstream iss{line};
        uint32_t elem;
        
        while (iss >> elem) {
            clust_lst[clust_id].push_back(elem);
            nelem++;
        }
        clust_id++;
    }

    if (clust_asn.size() != nelem){
        clust_asn.resize(nelem);
    }

    for (clust_id = 0; clust_id < clust_lst.size(); clust_id++){
        for (auto i = 0; i < clust_lst[clust_id].size(); i++){
            uint32_t elem = clust_lst[clust_id][i];
            clust_asn[elem] = clust_id;
        }
    }
    //printf("[read_clust_lst]\t %d clusters\n", clust_id);
    return clust_id;
}

/*  
 * Reads clustering stored in the form of cluster assignment
 * Reads to a vector representing cluster assignment
 * Return number of clusters
 * */
int read_clust_asn(string fname, vector<uint32_t>& clust_asn){
    int clust_id = 0;
    ifstream infile(fname);
    for(int i = 0; i < clust_asn.size(); i++){
        infile >> clust_asn[i];
        clust_id = max((uint32_t)clust_id, (uint32_t)clust_asn[i]);
    }
    return clust_id+1;
}

/*
 * Convert cluster assignment vector to cluster list
 * */
vector< vector<uint32_t> > clust_asn_to_lst(vector<uint32_t>& clust_asn){
    uint32_t max_clust_id = *(std::max_element(clust_asn.begin(), clust_asn.end()));
    //printf("[clust_asn_to_lst] max_clust_id %d\n", max_clust_id);
    vector< vector<uint32_t> > clust_lst(max_clust_id+1);
    for (uint32_t i = 0; i < clust_asn.size(); i++){
        clust_lst[clust_asn[i]].push_back(i);
    }

    return clust_lst;
}

/*
 * Relabel cluster assignment vector in such a way
 * that every cluster label is in the range [0,k)
 * where k is the number of unique clusters
 * */
void relabel_clust_asn(vector<uint32_t>& clust_asn){
    vector<uint32_t> label_map(clust_asn.size(), -1);
    uint32_t count = 0;
    for(uint32_t i = 0; i < clust_asn.size(); i++){
        if( label_map[clust_asn[i]] == -1 ){
            label_map[clust_asn[i]] = count;
            count++;
        }
        clust_asn[i] = label_map[clust_asn[i]];
    }
}

/*
 * Write given cluster list to a file
 * */
void write_clust_lst( string fname, vector<uint32_t> &clust_asn){
    vector< vector<uint32_t> > clust_lst = clust_asn_to_lst(clust_asn);
    ofstream outfile;
    outfile.open(fname, ofstream::trunc);
    for(auto i = 0; i < clust_lst.size(); i++){
        for (auto j = 0; j < clust_lst[i].size(); j++){
            outfile << clust_lst[i][j] << " ";
        }
        outfile << "\n";
    }

    outfile.close();
}

int main(int argc, char* argv[]){
    // Timers
    double t0, t1;

    string graphfile;
    string input_clustering_prefix; // Directory containing input clusterings
    int k; // Number of input clusterings
    int n;
    string output_prefix;
    string alg;
    double pre_proc_threshold = 1.0; // Default

    for(int i = 1; i < argc; i++){
        if (strcmp(argv[i], "--input-prefix") == 0){
            input_clustering_prefix = string(argv[i+1]);
        }
        else if (strcmp(argv[i], "--n") == 0){
            n = atoi(argv[i+1]);
        }
        else if (strcmp(argv[i], "--k") == 0){
            k = atoi(argv[i+1]);
        }
        else if (strcmp(argv[i], "--pre-proc-threshold") == 0){
            int t = atoi(argv[i+1]);
            pre_proc_threshold = (double)t / 100.00;
        }
    }
    
    /*
     * Read input clusterings of the graph
     * k clustering: vector of k elements
     *      - Each element is a vector of n elements - cluster assignment vector
     * */
    t0 = omp_get_wtime();
    vector< vector<uint32_t> > input_clusterings(k);
    for (int i = 0 ; i < k; i++){
        input_clusterings[i].resize(n);
        string cluster_assignment_fname = input_clustering_prefix + std::string(".") + to_string(i);
        read_clust_lst(cluster_assignment_fname, input_clusterings[i]);
        //clust_asn_to_lst(input_clusterings[i]);
    }
    t1 = omp_get_wtime();
    //printf("Time to read input clusterings: %lf\n", t1-t0);


   /*
    * Preprocessing
    * Find clustering of clustering
    * */
    igraph_matrix_t distance_kk;
    igraph_matrix_init(&distance_kk, k, k);
    igraph_matrix_fill(&distance_kk, 0);
    for(int i = 0; i < k; i++){
        for (int j = i; j < k; j++){
            igraph_vector_int_t partition_i_ig;
            igraph_vector_int_init(&partition_i_ig, input_clusterings[i].size());

            for(int ii = 0; ii < input_clusterings[i].size(); ii++){
                VECTOR(partition_i_ig)[ii] = input_clusterings[i][ii];
            }

            igraph_vector_int_t partition_j_ig;
            igraph_vector_int_init(&partition_j_ig, input_clusterings[j].size());

            for(int ii = 0; ii < input_clusterings[j].size(); ii++){
                VECTOR(partition_j_ig)[ii] = input_clusterings[j][ii];
            }

            igraph_real_t dist;
            igraph_compare_communities(&partition_i_ig, &partition_j_ig, &dist, IGRAPH_COMMCMP_SPLIT_JOIN);
            dist = (dist / 2) / double(n);
            igraph_matrix_set(&distance_kk, i, j, (dist));
            igraph_matrix_set(&distance_kk, j, i, (dist));
            int ki = *(std::max_element(input_clusterings[i].begin(), input_clusterings[i].end())) + 1;
            int kj = *(std::max_element(input_clusterings[j].begin(), input_clusterings[j].end())) + 1;
            //printf("%d cliusters in %d; %d clusters in %d; dist(%d, %d): %0.2lf\n", ki, i, kj, j, i, j, (dist));
        }
    }

    //for(int i = 0; i < k; i++){
        //for (int j = 0; j < k; j++){
            //printf("%0.2lf ", MATRIX(distance_kk,i,j) );
        //}
        //printf("\n");
    //}
    
    int count1 = 0;
    int count2 = 0;
    for(int i = 0; i < k; i++){
        for (int j = 0; j < k; j++){
            if(MATRIX(distance_kk, i, j) > pre_proc_threshold){
                igraph_matrix_set(&distance_kk, i, j, 0);
                count1++;
            }
            else{
                igraph_matrix_set(&distance_kk, i, j, 1);
                count2++;
            }
            //printf("%0.2lf ", MATRIX(distance_kk,i,j) );
            //printf("drop count: %d, keep count: %d\n", count1, count2 );
        }
        //printf("\n");
    }

    //for(int i = 0; i < k; i++){
        //for (int j = 0; j < k; j++){
            //if(MATRIX(distance_kk, i, j) == 0){
                //printf("     ");
            //}
            //else printf("%0.2lf ", MATRIX(distance_kk,i,j) );
        //}
        //printf("\n");
    //}
    igraph_t contingency_ig;
    igraph_adjacency(&contingency_ig, &distance_kk, IGRAPH_ADJ_DIRECTED, IGRAPH_NO_LOOPS);

    igraph_integer_t n_components;
    igraph_vector_int_t membership;
    igraph_vector_int_init(&membership, k);
    igraph_vector_int_t component_sizes;
    igraph_vector_int_init(&component_sizes, k);
    igraph_connected_components(&contingency_ig, &membership, &component_sizes, &n_components, IGRAPH_WEAK);
    //printf("%d components\n", n_components);
    //for(int  i = 0; i < k; i++){
        //printf("Partition %d: Contains %d clusters. Belongs to component %d\n", i, *(std::max_element(input_clusterings[i].begin(), input_clusterings[i].end())) + 1, VECTOR(membership)[i] );
    //}

    //printf("%s,%d,%ld\n", input_clustering_prefix.c_str(), k, n_components);
    printf("%lf,%d,%ld\n", pre_proc_threshold, k, n_components);

	return 0;
}
