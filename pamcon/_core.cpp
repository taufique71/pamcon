#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <cstdint>
#include <omp.h>

#include "../CSC.h"
#include "../COO.h"
#include "../consensus.h"

namespace py = pybind11;
using namespace std;

py::array_t<uint32_t> consensus_binding(
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> col_ptr,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> row_ids,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> values,
    uint32_t nrows,
    uint32_t ncols,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> C_array,
    int niter,
    bool verbose
) {
    auto col_ptr_buf = col_ptr.request();
    auto row_ids_buf = row_ids.request();
    auto values_buf  = values.request();
    auto C_buf       = C_array.request();

    uint32_t nnz = (uint32_t)row_ids_buf.shape[0];
    uint32_t k   = (uint32_t)C_buf.shape[1];

    uint32_t* col_ptr_data = static_cast<uint32_t*>(col_ptr_buf.ptr);
    uint32_t* row_ids_data = static_cast<uint32_t*>(row_ids_buf.ptr);
    uint32_t* values_data  = static_cast<uint32_t*>(values_buf.ptr);
    uint32_t* C_data       = static_cast<uint32_t*>(C_buf.ptr);

    // Build CSC directly from scipy sparse arrays
    // scipy sparse CSC is already column-sorted, so col_sort=true
    pvector<uint32_t> colPtr_v(ncols + 1);
    pvector<uint32_t> rowIds_v(nnz);
    pvector<uint32_t> nzVals_v(nnz);

    for (uint32_t i = 0; i <= ncols; i++) colPtr_v[i] = col_ptr_data[i];
    for (uint32_t i = 0; i < nnz; i++)    rowIds_v[i] = row_ids_data[i];
    for (uint32_t i = 0; i < nnz; i++)    nzVals_v[i] = values_data[i];

    CSC<uint32_t, uint32_t, uint32_t> graph(nrows, ncols, nnz, true, true);
    graph.cols_pvector(&colPtr_v);
    graph.nz_rows_pvector(&rowIds_v);
    graph.nz_vals_pvector(&nzVals_v);

    // Build C: vector<vector<uint32_t>> [nrows x k]
    // C_array is row-major (C-contiguous), so C_data[i * k + j] = C[i][j]
    vector<vector<uint32_t>> C(nrows, vector<uint32_t>(k));
    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < k; j++) {
            C[i][j] = C_data[i * k + j];
        }
    }

    // Run parallel consensus
    vector<uint32_t> result = parallel_consensus_v8(graph, C, niter, verbose);

    // Return result as 1D numpy array
    py::array_t<uint32_t> result_array(nrows);
    auto result_buf = result_array.request();
    uint32_t* result_data = static_cast<uint32_t*>(result_buf.ptr);
    for (uint32_t i = 0; i < nrows; i++) {
        result_data[i] = result[i];
    }

    return result_array;
}

void set_num_threads(int n) {
    omp_set_num_threads(n);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "pamcon C++ core: parallel consensus clustering";
    m.def("set_num_threads", &set_num_threads, py::arg("n"),
          "Set the number of OpenMP threads for parallel consensus.");
    m.def("consensus", &consensus_binding,
          py::arg("col_ptr"),
          py::arg("row_ids"),
          py::arg("values"),
          py::arg("nrows"),
          py::arg("ncols"),
          py::arg("C"),
          py::arg("niter") = 100,
          py::arg("verbose") = false
    );
}
