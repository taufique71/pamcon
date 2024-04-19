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

    string file_1;
    string file_2;
    string outputfile;
    string distance_metric;

    for(int i = 1; i < argc; i++){
        if (strcmp(argv[i], "--file-1") == 0){
            file_1 = string(argv[i+1]);
        }
        else if (strcmp(argv[i], "--file-2") == 0){
            file_2 = string(argv[i+1]);
        }
        else if (strcmp(argv[i], "--output-file") == 0){
            outputfile = string(argv[i+1]);
        }
        else if (strcmp(argv[i], "--distance-metric") == 0){
            distance_metric = string(argv[i+1]);
        }
    }

    cout << "File-1: " << file_1 << endl;
    cout << "File-2: " << file_2 << endl;
    cout << "Distance file (output): " << outputfile << endl;
    cout << "Distance metric to use: " << distance_metric << endl;
    
    vector<uint32_t> clust_1;
    read_clust_lst(file_1, clust_1);
    igraph_vector_int_t clust_1_ig;
    igraph_vector_int_init(&clust_1_ig, clust_1.size() );
    //printf("%d\n", clust_1.size());
    for(int i = 0; i < clust_1.size(); i++){
        VECTOR(clust_1_ig)[i] = clust_1[i];
    }

    vector<uint32_t> clust_2;
    read_clust_lst(file_2, clust_2);
    igraph_vector_int_t clust_2_ig;
    igraph_vector_int_init(&clust_2_ig, clust_2.size() );
    for(int i = 0; i < clust_2.size(); i++){
        VECTOR(clust_2_ig)[i] = clust_2[i];
    }

    ofstream outfile;
    outfile.open(outputfile, ofstream::trunc);

    if(distance_metric == "nmi"){
        igraph_real_t dist;
        igraph_compare_communities(&clust_2_ig, &clust_1_ig, &dist, IGRAPH_COMMCMP_NMI);
        outfile << dist << "\n";
    }
    else if(distance_metric == "vi"){
        igraph_real_t dist;
        igraph_compare_communities(&clust_2_ig, &clust_1_ig, &dist, IGRAPH_COMMCMP_VI);
        outfile << dist << "\n";
    }
    else if(distance_metric == "split-join"){
        igraph_real_t dist;
        igraph_compare_communities(&clust_2_ig, &clust_1_ig, &dist, IGRAPH_COMMCMP_SPLIT_JOIN);
        dist = (dist / 2) / double(clust_1.size());
        outfile << dist << "\n";
    }
    else if(distance_metric == "rand"){
        igraph_real_t dist;
        igraph_compare_communities(&clust_2_ig, &clust_1_ig, &dist, IGRAPH_COMMCMP_RAND);
        dist = 1.0 - dist;
        outfile << dist << "\n";
    }


    outfile.close();
    igraph_vector_int_destroy(&clust_1_ig);
    igraph_vector_int_destroy(&clust_2_ig);

	return 0;
}
