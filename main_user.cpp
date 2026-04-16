#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>

#include <omp.h>

#include "COO.h"
#include "CSC.h"
#include "CSC_adder.h"
#include "consensus.h"

using namespace std;

void print_help(const char* prog) {
    printf("Usage: %s --graph-file <file> --input-prefix <prefix> --k <int> --output-prefix <prefix> [options]\n\n", prog);
    printf("Required arguments:\n");
    printf("  --graph-file <file>       Input graph in Matrix Market format (.mtx)\n");
    printf("  --input-prefix <prefix>   Path prefix for input clustering files\n");
    printf("                            Files must be named <prefix>.0, <prefix>.1, ..., <prefix>.(k-1)\n");
    printf("  --k <int>                 Number of input clusterings\n");
    printf("  --output-prefix <prefix>  Path prefix for output file\n\n");
    printf("Optional arguments:\n");
    printf("  --niter <int>             Maximum number of iterations (default: 100)\n");
    printf("  --verbose                 Print detailed progress per iteration\n");
    printf("  --help, -h                Show this help message\n\n");
    printf("Parallelism:\n");
    printf("  Control threads via the OMP_NUM_THREADS environment variable.\n");
    printf("  Example: OMP_NUM_THREADS=8 %s --graph-file graph.mtx ...\n\n", prog);
    printf("Output:\n");
    printf("  Writes consensus clustering to <output-prefix>.soln\n");
    printf("  Each line in the output file lists the vertex IDs belonging to that cluster.\n");
}

int main(int argc, char* argv[]) {

    if (argc < 2) {
        print_help(argv[0]);
        return 0;
    }

    string graphfile;
    string input_prefix;
    int k = -1;
    string output_prefix;
    int niter = 100;
    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_help(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "--graph-file") == 0) {
            if (i+1 >= argc) { fprintf(stderr, "Error: --graph-file requires a value.\n"); return 1; }
            graphfile = string(argv[++i]);
        }
        else if (strcmp(argv[i], "--input-prefix") == 0) {
            if (i+1 >= argc) { fprintf(stderr, "Error: --input-prefix requires a value.\n"); return 1; }
            input_prefix = string(argv[++i]);
        }
        else if (strcmp(argv[i], "--k") == 0) {
            if (i+1 >= argc) { fprintf(stderr, "Error: --k requires a value.\n"); return 1; }
            k = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--output-prefix") == 0) {
            if (i+1 >= argc) { fprintf(stderr, "Error: --output-prefix requires a value.\n"); return 1; }
            output_prefix = string(argv[++i]);
        }
        else if (strcmp(argv[i], "--niter") == 0) {
            if (i+1 >= argc) { fprintf(stderr, "Error: --niter requires a value.\n"); return 1; }
            niter = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        }
        else {
            fprintf(stderr, "Error: Unknown argument '%s'.\n", argv[i]);
            fprintf(stderr, "Run '%s --help' for usage.\n", argv[0]);
            return 1;
        }
    }

    // Validate required arguments
    bool ok = true;
    if (graphfile.empty())    { fprintf(stderr, "Error: --graph-file is required.\n");  ok = false; }
    if (input_prefix.empty()) { fprintf(stderr, "Error: --input-prefix is required.\n"); ok = false; }
    if (k < 1)                { fprintf(stderr, "Error: --k is required and must be >= 1.\n"); ok = false; }
    if (output_prefix.empty()) { fprintf(stderr, "Error: --output-prefix is required.\n"); ok = false; }
    if (!ok) {
        fprintf(stderr, "Run '%s --help' for usage.\n", argv[0]);
        return 1;
    }
    if (niter < 1) {
        fprintf(stderr, "Error: --niter must be >= 1.\n");
        return 1;
    }

    // Read graph
    printf("Reading graph from '%s' ...\n", graphfile.c_str());
    CSC<uint32_t, uint32_t, uint32_t> graph;
    {
        COO<uint32_t, uint32_t, uint32_t> coo;
        coo.ReadMM(graphfile);
        graph = CSC<uint32_t, uint32_t, uint32_t>(coo);
    }
    uint32_t n = graph.get_nrows();
    printf("  %u vertices, %u edges\n", n, graph.get_nnz());

    // Make undirected if needed
    {
        COO<uint32_t, uint32_t, uint32_t> coo_cv = graph.to_COO<uint32_t>();
        coo_cv.transpose();
        CSC<uint32_t, uint32_t, uint32_t> graph_cv(coo_cv);
        if (!(graph == graph_cv)) {
            graph = SpAddRegular<uint32_t, uint32_t, uint32_t, uint32_t>(&graph, &graph_cv);
            printf("  Graph was directed — symmetrized to undirected.\n");
        }
    }

    // Read input clusterings
    printf("Reading %d input clusterings from '%s.0' ... '%s.%d' ...\n",
           k, input_prefix.c_str(), input_prefix.c_str(), k-1);
    vector<vector<uint32_t>> input_clusterings(k);
    for (int i = 0; i < k; i++) {
        input_clusterings[i].resize(n);
        string fname = input_prefix + "." + to_string(i);
        uint32_t nclusters = read_clust_lst(fname, input_clusterings[i]);
        if (nclusters == 0) {
            fprintf(stderr, "Error: Could not read clustering file '%s'.\n", fname.c_str());
            fprintf(stderr, "  Make sure the file exists and is non-empty.\n");
            return 1;
        }
        if (verbose) printf("  Clustering %d: %u clusters\n", i, nclusters);
    }
    printf("  Loaded %d clusterings.\n", k);

    // Build cluster assignment matrix C[vertex] = vector of k cluster labels
    vector<vector<uint32_t>> C(n);
    for (uint32_t i = 0; i < n; i++) {
        C[i].resize(k);
        for (int j = 0; j < k; j++) {
            C[i][j] = input_clusterings[j][i];
        }
    }

    // Run consensus
    printf("Running consensus (max %d iterations, %d thread(s)) ...\n",
           niter, omp_get_max_threads());
    double t0 = omp_get_wtime();
    vector<uint32_t> result = parallel_consensus_v8(graph, C, niter, verbose);
    double t1 = omp_get_wtime();

    int kcons = *(max_element(result.begin(), result.end())) + 1;
    printf("Done in %.3f seconds. Consensus clustering has %d clusters.\n", t1-t0, kcons);

    // Write output
    string outfile = output_prefix + ".soln";
    write_clust_lst(outfile, result);
    printf("Output written to '%s'\n", outfile.c_str());

    return 0;
}
