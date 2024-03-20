#include <igraph.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
using namespace std;

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

int main(int argc, char* argv[]){
    // Timers
    double t0, t1;

    string consensus_file;
    string ground_truth_file;
    string input_clustering_prefix; // Directory containing input clusterings
    int k; // Number of input clusterings
    string outputfile;
    string distance_metric;
    if(argc < 10){
        printf("Not enough arguments!!\n");
        return -1;
    }

    for(int i = 1; i < argc; i++){
        if (strcmp(argv[i], "--consensus-file") == 0){
            consensus_file = string(argv[i+1]);
        }
        else if (strcmp(argv[i], "--ground-truth-file") == 0){
            ground_truth_file = string(argv[i+1]);
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
        else if (strcmp(argv[i], "--distance-metric") == 0){
            distance_metric = string(argv[i+1]);
        }
    }

    cout << "Consensus file: " << consensus_file << endl;
    cout << "Ground truth file: " << ground_truth_file << endl;
    cout << "Input clustering prefix: " << input_clustering_prefix << endl;
    cout << "Number of input clusterings: " << k << endl;
    cout << "Distance distribution file (output): " << outputfile << endl;
    cout << "Distance metric to use: " << distance_metric << endl;
    

    vector<uint32_t> consensus;
    read_clust_lst(consensus_file, consensus);
    igraph_vector_int_t consensus_ig;
    igraph_vector_int_init(&consensus_ig, consensus.size() );
    for(int i = 0; i < consensus.size(); i++){
        VECTOR(consensus_ig)[i] = consensus[i];
    }

    vector<uint32_t> ground_truth;
    igraph_vector_int_t ground_truth_ig;
    if(ground_truth_file != ""){
        read_clust_lst(ground_truth_file, ground_truth);
        igraph_vector_int_init(&ground_truth_ig, ground_truth.size() );
        for(int i = 0; i < ground_truth.size(); i++){
            VECTOR(ground_truth_ig)[i] = ground_truth[i];
        }
    }

    ofstream outfile;
    outfile.open(outputfile, ofstream::trunc);

    if(distance_metric == "nmi"){
        igraph_real_t dist;
        //cout << "---" << endl;
        //cout << "Comparison between:" << endl;
        //cout << ground_truth_file << endl;
        //cout << "and" << endl;
        //cout << consensus_file << endl;
        igraph_compare_communities(&ground_truth_ig, &consensus_ig, &dist, IGRAPH_COMMCMP_NMI);
        outfile << dist << "\n";
        //cout << "---" << endl;
    }
    else{
        vector< vector<uint32_t> > input_clusterings(k);
        for (int j = 0; j < k; j++){
            string clust_lst_fname = input_clustering_prefix + std::string(".") + to_string(j);
            read_clust_lst(clust_lst_fname, input_clusterings[j]);

            //printf("Comparing: %s with %s\n", consensus_file.c_str(), clust_lst_fname.c_str());

            igraph_vector_int_t ip_ig;
            igraph_vector_int_init(&ip_ig, input_clusterings[j].size() );
            for(int i = 0; i < input_clusterings[j].size(); i++){
                VECTOR(ip_ig)[i] = input_clusterings[j][i];
            }

            //cout << "---" << endl;
            //cout << "Comparison between:" << endl;
            //cout << clust_lst_fname << endl;
            //cout << "and" << endl;
            //cout << consensus_file << endl;

            igraph_real_t dist;
            if(distance_metric == "vi"){
                igraph_compare_communities(&ip_ig, &consensus_ig, &dist, IGRAPH_COMMCMP_VI);
            }
            else if(distance_metric == "split-join"){
                igraph_compare_communities(&ip_ig, &consensus_ig, &dist, IGRAPH_COMMCMP_SPLIT_JOIN);
                dist = (dist / 2) / double(consensus.size());
            }
            else if(distance_metric == "rand") {
                igraph_compare_communities(&ip_ig, &consensus_ig, &dist, IGRAPH_COMMCMP_RAND);
                dist = 1.0 - dist;
            }
            //printf("%s distance: %g\n", distance_metric.c_str(), dist);
            outfile << dist << "\n";
            //cout << "---" << endl;
            igraph_vector_int_destroy(&ip_ig);
        }
    }


    outfile.close();
    igraph_vector_int_destroy(&consensus_ig);
    if(ground_truth_file != "") igraph_vector_int_destroy(&ground_truth_ig);

	return 0;
}
