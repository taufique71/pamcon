#include <iostream>
#include <vector>
#include <string>
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

#include "COO.h"
#include "CSC.h"
#include "CSC_adder.h"

#include "consensus.h"

using namespace std;

int main(int argc, char* argv[]){
    // Timers
    double t0, t1;

    string graphfile;
    string input_clustering_prefix; // Directory containing input clusterings
    int k; // Number of input clusterings
    string output_prefix;
    string alg;
    double pre_proc_threshold = 1.0; // Default
    if(argc < 10){
        printf("Not enough arguments!!\n");
        return -1;
    }

    for(int i = 1; i < argc; i++){
        if (strcmp(argv[i], "--graph-file") == 0){
            graphfile = string(argv[i+1]);
        }
        else if (strcmp(argv[i], "--input-prefix") == 0){
            input_clustering_prefix = string(argv[i+1]);
        }
        else if (strcmp(argv[i], "--k") == 0){
            k = atoi(argv[i+1]);
        }
        else if (strcmp(argv[i], "--output-prefix") == 0){
            output_prefix = string(argv[i+1]);
        }
        else if (strcmp(argv[i], "--alg") == 0){
            alg = string(argv[i+1]);
        }
        else if (strcmp(argv[i], "--pre-proc-threshold") == 0){
            pre_proc_threshold = atof(argv[i+1]);
        }
    }

    cout << "Graph: " << graphfile << endl;
    cout << "Input clustering prefix: " << input_clustering_prefix << endl;
    cout << "Number of input clusterings: " << k << endl;
    cout << "Output file prefix: " << output_prefix<< endl;
    cout << "Algorithm to use: " << alg << endl;
    cout << "Pre process threshold: " << pre_proc_threshold << endl;
    if ( (alg == "boem") || (alg == "saoem") ){
        cout << "!!! Pre process threshold has no effect for " << alg << endl;
    }
    
    if ( (alg == "boem") || (alg == "saoem") ){
        // Run methods from CONPAS
        vector<string> fileNames;
        for (int i = 0; i < k; i++){
            string cluster_assignment_fname = input_clustering_prefix + std::string(".") + to_string(i);
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
            //string outputfile = output_prefix + std::string(".") + alg;
            string outputfile = output_prefix + std::string(".soln-") + std::to_string(0);
            saoem_sp.WriteClusterListFile(outputfile);
        }
        else if(alg == "boem"){
            // Run greedy best one element move
            BestOneElementMove boem;
            t0 = omp_get_wtime();
            auto boem_sp = boem.Run(spv);
            t1 = omp_get_wtime();
            printf("BOEM: %lf seconds\n", t1-t0);
            
            //cout << boem_sp << endl;
            //boem_sp.PrintDistributionOfBlockSizes();
            string outputfile = output_prefix + std::string(".soln-") + std::to_string(0);
            boem_sp.WriteClusterListFile(outputfile);
        }
    }
    else if( (alg == "v8") || (alg == "v8-parallel") ) {
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
            string cluster_assignment_fname = input_clustering_prefix + std::string(".") + to_string(i);
            read_clust_lst(cluster_assignment_fname, input_clusterings[i]);
            //clust_asn_to_lst(input_clusterings[i]);
        }
        t1 = omp_get_wtime();
        printf("Time to read input clusterings: %lf\n", t1-t0);


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
            //printf("Partition %d: Contains %d clusters. Belongs to component %d: size %ld\n", i, *(std::max_element(input_clusterings[i].begin(), input_clusterings[i].end())) + 1, VECTOR(membership)[i], VECTOR(component_sizes)[VECTOR(membership)[i]] );
        //}
        std::vector< std::pair<int, int> > component_ranking(n_components);
        for(int c = 0; c < n_components; c++){
            component_ranking[c] = std::make_pair(c, (int)VECTOR(component_sizes)[c]);
        }

        std::sort(component_ranking.begin(), component_ranking.end(), 
                  [](std::pair<int, int> lhs, std::pair<int, int> rhs){
                    if(lhs.second < rhs.second){ return false; }
                    else return true;
                  });
        
        /*
         * Find consensus for each group/component
         * */
        for(int c = 0; c < n_components; c++){
            int cid = component_ranking[c].first;
            int csize = component_ranking[c].second;
            //printf("cid: %d, csize: %d\n", cid, csize);
            /*
             * Convert the input clusterings into a cluster assignment matrix
             * Vector of n elements
             *      - Each element is a vector of k elements
             * */
            t0 = omp_get_wtime();
            vector< vector<uint32_t> > C(n);
            for (int i = 0 ; i < n; i++){
                C[i].resize(csize);
                int cj = 0;
                for(int j = 0; j < k; j++){
                    if (VECTOR(membership)[j] == cid){
                        C[i][cj] = input_clusterings[j][i];
                        cj++;
                        //printf("%d ", j);
                    }
                }
            }
            t1 = omp_get_wtime();
            printf("Time to prepare cluster assignment matrix: %lf\n", t1-t0);

            vector<uint32_t> cons_clust_asn;
            t0 = omp_get_wtime();
            if(alg == "v8") cons_clust_asn = consensus_v8(graph, C, 100); // Maximum 100 iterations
            else if(alg == "v8-parallel") cons_clust_asn = parallel_consensus_v8(graph, C, 100, false); // Maximum 100 iterations, verbose false
            t1 = omp_get_wtime();
            printf("%s: %lf seconds\n", alg.c_str(), t1-t0);
            int kcons = *(std::max_element(cons_clust_asn.begin(), cons_clust_asn.end())) + 1;
            printf("%d clusters in solution %d, brings consensus of %d partitions\n", kcons, c, csize);
            printf("soln-%d: consensus of [", c);
            for(int j = 0; j < k; j++){
                if (VECTOR(membership)[j] == cid){
                    printf("%d, ", j);
                }
            }
            printf("]\n");

            string outputfile = output_prefix + std::string(".soln-") + std::to_string(c);
            write_clust_lst(outputfile, cons_clust_asn);
        }
    }

	return 0;
}
