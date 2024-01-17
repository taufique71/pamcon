#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>

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

#include "utils.h"
#include "COO.h"
#include "CSC.h"
#include "CSC_adder.h"
#include "GAP/pvector.h"
#include "GAP/timer.h"

using namespace std;

class potential_moves{
public:
    vector<uint32_t> to;
    vector<uint32_t> from;
    vector<uint32_t> attractor;
    vector<int> deltaS;
    vector<bool> valid;

    potential_moves(uint32_t n){
        // Resize
        to.resize(n);
        from.resize(n);
        attractor.resize(n);
        deltaS.resize(n);
        valid.resize(n);

        // Initialize
        for(int i = 0; i < n; i++){
            to[i] = i;
            from[i] = i;
            attractor[i] = i;
            valid[i] = false; // Intially all moves are marked as invalid
            deltaS[i] = 0; // Initially no improvement is sum of distances for moving any vertex
        }
    }
};

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

/*
 * Consensus implementation
 * Input: graph and cluster assignment matrix
 * */
vector<uint32_t> consensus_v8(CSC<uint32_t, uint32_t, uint32_t> &graph, vector< vector<uint32_t> > &C, int niter=10){
    double t0, t1, t2, t3, t4, t5, t6, t7, t8, t9;
    double tInit = 0;
    double tSearch = 0;
    double tValidate = 0;
    double tMove = 0;

    uint32_t n = graph.get_nrows(); // Number of vertices in the graph
    uint32_t nnz = graph.get_nnz(); // Number of edges in the graph
    uint32_t k = C[0].size(); // Number of input clustering
    
    t2 = omp_get_wtime();

    /*
     * Prepare weighted graph, weighted by the cluster agreement value
     * We call it consensus graph
     * */
    t0 = omp_get_wtime();
    
    CSC<uint32_t, uint32_t, uint32_t>csc;
    {
        // Need to be pvector because COO and CSC data structures require it
        pvector<uint32_t> nzRows(graph.get_nnz());
        pvector<uint32_t> nzCols(graph.get_nnz());
        pvector<uint32_t> nzVals(graph.get_nnz());
        
        uint32_t idx = 0;
        for(int col = 0; col < graph.get_ncols(); col++){
            uint32_t colStart = (*graph.get_colPtr())[col];
            uint32_t colEnd = (*graph.get_colPtr())[col+1];
            for(int i = colStart; i < colEnd; i++){
                uint32_t row = (*graph.get_rowIds())[i];
                vector<bool> eq = ewise_equal<uint32_t>(C[col], C[row]);
                uint32_t sum = accumulate(eq.begin(), eq.end(), 0);

                nzRows[idx] = row;
                nzCols[idx] = col;
                nzVals[idx] = sum;
                idx++;
            }
        }

        COO<uint32_t, uint32_t, uint32_t>coo(
                graph.get_nrows(), graph.get_ncols(), graph.get_nnz(), 
                &nzRows, &nzCols, &nzVals, 
                true);

        csc = CSC<uint32_t, uint32_t, uint32_t>(coo);
    }

    t1 = omp_get_wtime();
    printf("[consensus_v8] Time to prepare the consensus graph: %lf\n", t1-t0);

    csc.PrintInfo();
    const pvector<uint32_t>* colPtr = csc.get_colPtr();
    const pvector<uint32_t>* rowIds = csc.get_rowIds();
    const pvector<uint32_t>* nzVals = csc.get_nzVals();

    /* 
     * Initialize singleton clusters
     * */
    t0 = omp_get_wtime();
    vector<uint32_t> clust_asn(n);
    for(uint32_t i = 0; i < n; i++){
        clust_asn[i] = i;
    }
    vector< vector<uint32_t> > clust_lst = clust_asn_to_lst(clust_asn);
    t1 = omp_get_wtime();
    tInit = t1-t0;
    
    // A vector containing valid moves
    // Used to store the moves in previous iteration
    // Helps to detect one-step circular moves
    vector<bool> last_valid(n, false);
    vector<int> last_deltaS(n, 0);

    for(int it = 1; it <= niter; it++){
        printf("[consensus_v8] >>> Iteration: %d\n", it);

        potential_moves pm(n);
        
        t0 = omp_get_wtime();
        // Figure out edges to be probed
        // Maintain a list of edges to be probed for each vertex
        // Effectively a vector of vectors of tuples
        uint32_t nnz_to_probe=0;
        vector< vector< tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> > > nz_to_probe(n);
        for(uint32_t u = 0; u < n; u++){
            uint32_t a = clust_asn[u];
            for(uint32_t j = (*colPtr)[u]; j < (*colPtr)[u+1]; j++ ){
                uint32_t v = (*rowIds)[j];
                uint32_t w = (*nzVals)[j];
                uint32_t b = clust_asn[v];

                if(a!=b){
                    nnz_to_probe++;
                    nz_to_probe[u].push_back( make_tuple(u, a, v, b, w) );
                }
            }
        }
        t1 = omp_get_wtime();
        tSearch += (t1-t0);
        printf("[consensus_v8]\tTime to figure out edges to be probed %lf\n", t1-t0);
        printf("[consensus_v8]\t\tNumber of edges to probe %d\n", nnz_to_probe );
        
        // Compute Mua for all target u in current iteration
        // Equation for computation comes from https://web.cs.ucdavis.edu/~filkov/papers/ictai03.pdf
        t0 = omp_get_wtime();
        vector<int> Mua(n, 0);
        for(uint32_t u = 0; u < n; u++){
            if(nz_to_probe[u].size() > 0){
                uint32_t a = clust_asn[u];
                for(auto i = 0; i < clust_lst[a].size(); i++){
                    uint32_t e = clust_lst[a][i];
                    if(u != e){
                        vector<bool> eq = ewise_equal<uint32_t>(C[u], C[e]);
                        uint32_t sum = accumulate(eq.begin(), eq.end(), 0);
                        Mua[u] += (k - 2 * sum);
                    }
                }
            }
        }
        t1 = omp_get_wtime();
        tSearch += (t1-t0);
        printf("[consensus_v8]\tTime to compute Mua for all vertices %lf\n", t1-t0);

        // Compute Mub for potential moves and check feasibility with deltaS
        // Equation for computation comes from https://web.cs.ucdavis.edu/~filkov/papers/ictai03.pdf
        t0 = omp_get_wtime();
        for(uint32_t i = 0; i < n; i++){
            sort(nz_to_probe[i].begin(), nz_to_probe[i].end(), [] (tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> &lhs, tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> &rhs){
                auto lhs_b = get<3>(lhs);
                auto lhs_w = get<4>(lhs);
                auto rhs_b = get<3>(rhs);
                auto rhs_w = get<4>(rhs);
                if (lhs_b < rhs_b){
                    return true;
                }
                else{
                    if (lhs_b > rhs_b){
                        return false;
                    }
                    else return(lhs_w > rhs_w);
                }
            });
            for(auto j = 0 ; j < nz_to_probe[i].size(); ){
                auto u = get<0>(nz_to_probe[i][j]);
                auto a = get<1>(nz_to_probe[i][j]);
                auto v = get<2>(nz_to_probe[i][j]);
                auto b = get<3>(nz_to_probe[i][j]);
                auto w = get<4>(nz_to_probe[i][j]);
                
                int Mub = 0;
                for(auto jj = 0; jj < clust_lst[b].size(); jj++){
                    auto e = clust_lst[b][jj];
                    vector<bool> eq = ewise_equal<uint32_t>(C[u], C[e]);
                    uint32_t sum = accumulate(eq.begin(), eq.end(), 0);
                    Mub += (k - 2 * sum);
                }

                int deltaS = Mub - Mua[u];
                // If current potential move gives best potential reduction in distance
                if (deltaS < pm.deltaS[u]){
                    pm.from[u] = a; // Mark potential move for u as moving from cluster a
                    pm.to[u] = b; // Mark potential move for u as moving to cluster b
                    pm.attractor[u] = v; // Mark potential move for u as it is attracted by v
                    pm.deltaS[u] = deltaS; // Keep track of potential reduction in distance if this move takes place 
                    pm.valid[u] = true; // Initially mark this move to be a valid. It may become invalid after validation test
                }

                while(get<3>(nz_to_probe[i][j]) == b) j++;
            }
        }
        t1 = omp_get_wtime();
        tSearch += (t1-t0);
        printf("[consensus_v8]\tTime to figure out potential moves for all vertices %lf\n", t1-t0);
        printf("[consensus_v8]\t\tNumber of valid moves %d\n", accumulate(pm.valid.begin(), pm.valid.end(), 0) );

        // First validation pass:
        // Is a move such that u moves to some cluster attracted by v while v moves to the cluster of u attracted by it?
        // In that case one of those two moves would be invalid
        t0 = omp_get_wtime();
        for(uint32_t u = 0; u < n; u++){
            if(pm.valid[u] == true){
                auto v = pm.attractor[u];
                if( (pm.valid[v] == true) && (pm.attractor[v] == u) ){
                    if (pm.deltaS[u] <= pm.deltaS[v] ){
                        pm.valid[v] = false;
                        pm.deltaS[v] = 0;
                    }
                    else{
                        pm.valid[u] = false;
                        pm.deltaS[u] = 0;
                    }
                }

            }
        }
        t1 = omp_get_wtime();
        tValidate += (t1-t0);
        printf("[consensus_v8]\tTime for first validation pass %lf\n", t1-t0);
        printf("[consensus_v8]\t\tNumber of valid moves %d\n", accumulate(pm.valid.begin(), pm.valid.end(), 0) );

        // Second validation pass:
        // If same set of vertices are moving then total potential reduction in distance should be lower than the previous iteration
        // Main a flag to denote if any move passed the validation tests
        bool flag = false; // Initially mark as none
        t0 = omp_get_wtime();
        vector<bool> neq = ewise_not_equal<bool>(pm.valid, last_valid);
        uint32_t sum = accumulate(neq.begin(), neq.end(), 0);
        if(sum == 0){
            // Same set of vertices are being moved as in the previous iteration
            if( accumulate(pm.deltaS.begin(), pm.deltaS.end(), 0) < accumulate(last_deltaS.begin(), last_deltaS.end(), 0) ){
                last_valid = pm.valid;
                last_deltaS = pm.deltaS;
                flag = true;
            }
            else{
                //Keep the flag false to denote no valid move this iteration
                flag = false;
            }
        }
        else{
            last_valid = pm.valid;
            last_deltaS = pm.deltaS;
            flag = true;
        }
        t1 = omp_get_wtime();
        tValidate += (t1-t0);
        printf("[consensus_v8]\tTime for second validation pass %lf\n", t1-t0);
        printf("[consensus_v8]\t\tNumber of valid moves %d\n", accumulate(pm.valid.begin(), pm.valid.end(), 0) );

        // Perform moves
        if(flag == true){
            t0 = omp_get_wtime();
            for(uint32_t u = 0; u < n; u++){
                if (pm.valid[u] == true){
                    auto b = pm.to[u];
                    clust_asn[u] = b;
                }
            }
            relabel_clust_asn(clust_asn);
            clust_lst = clust_asn_to_lst(clust_asn);
            t1 = omp_get_wtime();
            tMove += t1-t0;
            printf("[consensus_v8]\tTime for vertex movement %lf\n", t1-t0);
        }
        else{
            break;
        }
        printf("[consensus_v8]\tNumber of clusters %d\n", clust_lst.size() );
    }

    t3 = omp_get_wtime();
    printf("[consensus_v8]\tTime to reach consensus %lf\n", t3-t2 );
    printf("[consensus_v8]\t\ttInit %lf\n", tInit );
    printf("[consensus_v8]\t\ttSearch %lf\n", tSearch );
    printf("[consensus_v8]\t\ttValidate %lf\n", tValidate );
    printf("[consensus_v8]\t\ttMove %lf\n", tMove );

    return clust_asn;
}

int main(int argc, char* argv[]){
    // Timers
    double t0, t1;

    string graphfile;
    string input_clustering_prefix; // Directory containing input clusterings
    int k; // Number of input clusterings
    string outputfile;
    string alg;
    if(argc < 10){
        printf("Not enough arguments!!\n");
        return -1;
    }

    for(int i = 1; i < argc; i++){
        if (strcmp(argv[i], "--graph-file") == 0){
            graphfile = string(argv[i+1]);
        }
        else if (strcmp(argv[i], "--input-clustering-prefix") == 0){
            input_clustering_prefix = string(argv[i+1]);
        }
        else if (strcmp(argv[i], "--number-of-input-clustering") == 0){
            k = atoi(argv[i+1]);
        }
        else if (strcmp(argv[i], "--output-file") == 0){
            outputfile = string(argv[i+1]);
        }
        else if (strcmp(argv[i], "--alg") == 0){
            alg = string(argv[i+1]);
        }
    }

    cout << "Graph: " << graphfile << endl;
    cout << "Input clustering prefix: " << input_clustering_prefix << endl;
    cout << "Number of input clusterings: " << k << endl;
    cout << "Output clustering file: " << outputfile << endl;
    cout << "Algorithm to use: " << alg << endl;
    
    if ( (alg == "boem") || (alg == "saoem") ){
        // Run methods from CONPAS
        vector<string> fileNames;
        for (int i = 0; i < k; i++){
            string cluster_assignment_fname = input_clustering_prefix + to_string(i);
            fileNames.push_back(cluster_assignment_fname);
        }
        SetPartitionVector spv;
        spv.LoadFiles(fileNames);

        if(alg == "saoem"){
            // Run simulated annealing one element move
            SimulatedAnnealingOEM saoem;
            t0 = omp_get_wtime();
            auto saoem_sp = saoem.Run(spv);
            t1 = omp_get_wtime();
            printf("SAOEM: %lf seconds\n", t1-t0);

            saoem_sp.WriteClusterListFile(outputfile);
        }
        else if(alg == "boem"){
            // Run greedy best one element move
            BestOneElementMove boem;
            t0 = omp_get_wtime();
            auto boem_sp = boem.Run(spv);
            t1 = omp_get_wtime();
            printf("BOEM: %lf seconds\n", t1-t0);
            
            boem_sp.WriteClusterListFile(outputfile);
        }
    }
    else if(alg == "v8"){
        // Run prototype v8 for graphs
        /*
         * Read input graph
         * */
        t0 = omp_get_wtime();
        CSC<uint32_t, uint32_t, uint32_t> graph;
        {
            // Doing this in specific scope to force clean up of unnecessary memory blocks
            COO<uint32_t, uint32_t, uint32_t> coo;
            coo.ReadMM(graphfile);
            graph = CSC<uint32_t, uint32_t, uint32_t>(coo);
            graph.PrintInfo();
        }
        t1 = omp_get_wtime();
        printf("Time to read input graph: %lf\n", t1-t0);
        
        uint32_t n = graph.get_nrows(); // Number of vertices in the graph
        
        /*
         * Make undirected if not already
         * */
        t0 = omp_get_wtime();
        {
            COO<uint32_t, uint32_t, uint32_t> coo_cv = graph.to_COO<uint32_t>();
            coo_cv.transpose();
            CSC<uint32_t, uint32_t, uint32_t> graph_cv(coo_cv);
            if(graph == graph_cv){
                printf("Undirected\n");
            }
            else{
                printf("Directed\n");
                graph = SpAddRegular<uint32_t, uint32_t, uint32_t, uint32_t>(&graph, &graph_cv);
            }
        }
        t1 = omp_get_wtime();
        printf("Time to make undirected: %lf\n", t1-t0);

        /*
         * Read input clusterings of the graph
         * k clustering: vector of k elements
         *      - Each element is a vector of n elements - cluster assignment vector
         * */
        t0 = omp_get_wtime();
        vector< vector<uint32_t> > input_clusterings(k);
        for (int i = 0 ; i < k; i++){
            input_clusterings[i].resize(n);
            string cluster_assignment_fname = input_clustering_prefix + to_string(i);
            read_clust_lst(cluster_assignment_fname, input_clusterings[i]);
            //clust_asn_to_lst(input_clusterings[i]);
        }
        t1 = omp_get_wtime();
        printf("Time to read input clusterings: %lf\n", t1-t0);

        /*
         * Convert the input clusterings into a cluster assignment matrix
         * Vector of n elements
         *      - Each element is a vector of k elements
         * */
        t0 = omp_get_wtime();
        vector< vector<uint32_t> > C(n);
        for (int i = 0 ; i < n; i++){
            C[i].resize(k);
            for(int j = 0; j < k; j++){
                C[i][j] = input_clusterings[j][i];
            }
        }
        t1 = omp_get_wtime();
        printf("Time to prepare cluster assignment matrix: %lf\n", t1-t0);
        
        vector<uint32_t> cons_clust_asn = consensus_v8(graph, C, 100); // Maximum 100 iterations
        write_clust_lst(outputfile, cons_clust_asn);
    }

	return 0;
}
