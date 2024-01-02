#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "CSC.h"
#include "COO.h"
#include "GAP/pvector.h"
#include "GAP/timer.h"
#include "utils.h"

using namespace std;

/*
 * Elementwise equality operation
 * Returns a boolean vector
 * */
vector<bool> ewise_equal(vector<uint32_t>& a, vector<uint32_t>& b){
    vector<bool> eq(a.size());
    for(int i = 0; i < a.size(); i++){
        eq[i] = (a[i] == b[i]) ? true : false ;
    }
    return eq;
}

/*
 *  Return number of clusters
 * */
uint32_t read_cluster_list(string fname, vector<uint32_t>& clust_asn){
    ifstream infile(fname);
    string line;
    uint32_t clust_id = 0;
    while(getline(infile, line)){
        //cout << line << endl;
        istringstream iss{line};
        uint32_t vtx;

        //printf("%d: ", clust_id);
        while (iss >> vtx) {
            clust_asn[vtx] = clust_id;
            //printf("%d ", vtx);
        }
        //printf("\n");
        clust_id++;
    }
    return clust_id;
}

/*  
 *  Return number of clusters
 * */
int read_cluster_assignment(string fname, vector<uint32_t>& clust_asn){
    int clust_id = 0;
    ifstream infile(fname);
    for(int i = 0; i < clust_asn.size(); i++){
        infile >> clust_asn[i];
        clust_id = max((uint32_t)clust_id, (uint32_t)clust_asn[i]);
    }
    return clust_id+1;
}

int main(int argc, char* argv[]){
    string graphfile(argv[1]);
    string input_clustering_prefix(argv[2]); // Directory containing input clusterings
    int k = atoi(argv[3]); // Number of input clusterings
    //string outputfile(argv[4]);

    cout << graphfile << endl;
    cout << input_clustering_prefix << endl;
    cout << k << endl;

    COO<uint32_t, uint32_t, uint32_t> coo;
    coo.ReadMM(graphfile);
    CSC<uint32_t, uint32_t, uint32_t> graph(coo);
    graph.PrintInfo();
    
    //printf("After CSC->COO conversion\n");
    //COO<uint32_t, uint32_t, uint32_t> coo_cv(graph);
    //coo_cv.PrintInfo();
    
    uint32_t n = graph.get_nrows(); // Number of vertices in the graph

    /*
     * Vector of k elements
     *      - Each element is a vector of n elements - cluster assignment vector
     * */
    vector< vector<uint32_t> > input_clusterings(k);
    for (int i = 0 ; i < k; i++){
        input_clusterings[i].resize(n);
        string cluster_assignment_fname = input_clustering_prefix + to_string(i);
        read_cluster_list(cluster_assignment_fname, input_clusterings[i]);
    }

    /*
     * Vector of n elements
     *      - Each element is a vector of k elements
     * */
    vector< vector<uint32_t> > C(n);
    for (int i = 0 ; i < n; i++){
        C[i].resize(k);
        for(int j = 0; j < k; j++){
            C[i][j] = input_clusterings[j][i];
        }
    }
    
    /*
     * Prepare weighted graph, weighted by the agreement value
     * */
    pvector<uint32_t> nzRows(graph.get_nnz());
    pvector<uint32_t> nzCols(graph.get_nnz());
    pvector<uint32_t> nzVals(graph.get_nnz());
    
    uint32_t idx = 0;
    for(int col = 0; col < graph.get_ncols(); col++){
        uint32_t colStart = (*graph.get_colPtr())[col];
        uint32_t colEnd = (*graph.get_colPtr())[col+1];
        for(int i = colStart; i < colEnd; i++){
            uint32_t row = (*graph.get_rowIds())[i];
            vector<bool> eq = ewise_equal(C[col], C[row]);
            uint32_t sum = accumulate(eq.begin(), eq.begin(), 0);

            nzRows[idx] = row;
            nzCols[idx] = col;
            nzVals[idx] = sum;
            idx++;
        }
    }

    COO<uint32_t, uint32_t, uint32_t>consensus_coo(graph.get_nrows(), graph.get_ncols(), graph.get_nnz(), &nzRows, &nzCols, &nzVals, true);
    CSC<uint32_t, uint32_t, uint32_t>consensus_csc(consensus_coo);
    consensus_csc.PrintInfo();

	return 0;
}
