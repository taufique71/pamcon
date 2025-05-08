# pamcon: Parallel median consensus clustering in complex networks

Implementation of our work discussed in the Nature Scientific Report paper (https://www.nature.com/articles/s41598-025-87479-6).
ArXiv preprint is here: https://arxiv.org/pdf/2408.11331

### Project outline
- `consensus.h` contains our implementations
- `main.cpp` contains a cpp application that uses different implementations - our implementation and implementation from Filkov and Skienna
- `Makefile` contains compilation recipes of the application as well as other scripts for testing

### Dependency:
You would need igraph installed to compile the project, even though implementation of our consensus algorithm does not require igraph, the preprocessing does, it might help in certain scenarions.

igraph: https://igraph.org/c/doc/igraph-Installation.html

### Building:
You need to update the appropriate directory names on top of the Makefile -https://github.com/taufique71/pamcon/blob/main/Makefile. 
`make consensus` should be enough after that.

### Usage:
Usage can be found on this script (Line 83-91) -https://github.com/taufique71/pamcon/blob/main/test/runtime.sh#L83-L91
 
`./consensus --graph-file <graph_file> \\
                              --input-prefix <input_clustering_file_prefix> \\
                              --k <number_of_input_clustering> \\
                              --output-prefix <output_file_prefix> \\
                              --pre-proc-threshold <number between 0 and 1> \\
                              --alg <algorithm_of_choice>

You may specify OMP_NUM_THREADS for parallelism.

### Explanation of the parameters:
Graph file need to be in matrix market format.
value of k is self-explanatory. If you are making consensus of 20 community assignment, k would be 20.
It expects input clustering files to be named in the following way: input_clustering_file_prefix.0, input_clustering_file_prefix.1,...input_clustering_file_prefix.k-1
Don't worry about the pre-process threshold. You can keep it 0.99 in all cases
Algorithm should be specificly "v8-parallel". There are other algorithm implementations in the repo, you would want to use this one.
Output will be save in a file named in the following way: output_file_prefix.soln-0.v8-parallel

