#ifndef _CSC_H_
#define _CSC_H_

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <tuple>
#include <random>
#include <vector>


#include "COO.h"
#include "utils.h"
#include "GAP/timer.h"
#include "GAP/util.h"
#include "GAP/pvector.h"
#include "GAP/platform_atomics.h"

#include <numeric>
//#include "gtest/gtest.h"



/*
 We heavily use pvector (equivalent to std::vector).
Since pvector uses size_t for indexing, we will also stick to  size_t for indexing. That means, nnz, nrows, ncols are all of size_t.
What should we do for value type for array of offsets such as colptr?
size_t is definitely safe because of the above argument.
However, it may take more space for tall and skinny matrices.
Hence, we use an optional type CPT for such cases. By default, CPT=size_t
*/



// RIT: Row Index Type
// VT: Value Type
// CPT: Column pointer type (use only for very tall and skinny matrices)
template <typename RIT, typename VT=double, typename CPT=size_t>
class CSC
{
public:
	CSC(): nrows_(0), ncols_(0), nnz_(0), isColSorted_(false) {}

	CSC(RIT nrows, size_t ncols, size_t nnz,bool col_sort_bool, bool isWeighted): nrows_(nrows), ncols_(ncols), nnz_(nnz), isColSorted_(col_sort_bool), isWeighted_(isWeighted) 
    {
		rowIds_.resize(nnz); 
        colPtr_.resize(ncols+1); 
        nzVals_.resize(nnz);
    }  // added by abhishek

	template <typename CIT>
	CSC(COO<RIT, CIT, VT> & cooMat);
	
	template <typename AddOp>
	void MergeDuplicateSort(AddOp binop);
	void PrintInfo();

	const pvector<CPT>* get_colPtr(); // added by abhishek
	const pvector<RIT>* get_rowIds(); 
	const pvector<VT>* get_nzVals();
    
    const CPT get_colPtr(size_t idx);

	CSC<RIT, VT, CPT>(CSC<RIT, VT, CPT> &&other): nrows_(other.nrows_),ncols_(other.ncols_),nnz_(other.nnz_),isWeighted_(other.isWeighted_),isColSorted_(other.isColSorted_)   // added by abhishek
	{
		rowIds_.resize(nnz_); colPtr_.resize(ncols_+1); nzVals_.resize(nnz_);
		colPtr_ = std::move(other.colPtr_);
		rowIds_ = std::move(other.rowIds_);
		nzVals_ = std::move(other.nzVals_);
	}

	CSC<RIT, VT, CPT>& operator= (CSC<RIT, VT, CPT> && other){ // added by abhishek
		nrows_ = other.nrows_;
		ncols_ = other.ncols_;
		nnz_ = other.nnz_;
		isWeighted_ = other.isWeighted_;
		isColSorted_ = other.isColSorted_;
		rowIds_.resize(nnz_); colPtr_.resize(ncols_+1); nzVals_.resize(nnz_);
		colPtr_ = std::move(other.colPtr_);
		rowIds_ = std::move(other.rowIds_);
		nzVals_ = std::move(other.nzVals_);
		return *this;
	}
    
    bool operator== (const CSC<RIT, VT, CPT> & other);

	size_t get_ncols();
	size_t get_nrows();
	size_t get_nnz() ;
    bool isWeighted(){ return isWeighted_; }
    bool isSorted(){ return isColSorted_; }

	void nz_rows_pvector(pvector<RIT>* row_pointer) { rowIds_ = std::move(*(row_pointer));} // added by abhishek
	void cols_pvector(pvector<CPT>* column_pointer) {colPtr_ = std::move(*(column_pointer));}
	void nz_vals_pvector(pvector<VT>* value_pointer) {nzVals_ = std::move(*(value_pointer));}
    
	void count_sort(pvector<std::pair<RIT, VT> >& all_elements, size_t expon); // added by abhishek
	void sort_inside_column();
    
    void column_split(std::vector< CSC<RIT, VT, CPT>* > &vec, int nsplit);

    template <typename CIT>
    COO<RIT, CIT, VT> to_COO();

private:
	size_t nrows_;
	size_t ncols_;
	size_t nnz_;
 
	pvector<CPT> colPtr_;
	pvector<RIT> rowIds_;
	pvector<VT> nzVals_;
	bool isWeighted_;
	bool isColSorted_;
};

template <typename RIT, typename VT, typename CPT>
const pvector<CPT>* CSC<RIT, VT, CPT>::get_colPtr()
{
	return &colPtr_;
}

template <typename RIT, typename VT, typename CPT>
const CPT CSC<RIT, VT, CPT>::get_colPtr(size_t idx)
{
    return colPtr_[idx];
}

template <typename RIT, typename VT, typename CPT>
const pvector<RIT>*  CSC<RIT, VT, CPT>::get_rowIds()
{
	return &rowIds_;
}

template <typename RIT, typename VT, typename CPT>
const pvector<VT>* CSC<RIT, VT, CPT>::get_nzVals()
{
	return &nzVals_;
} 



template <typename RIT, typename VT, typename CPT>
size_t CSC<RIT, VT, CPT>:: get_ncols()
{
	return ncols_;
}

template <typename RIT, typename VT, typename CPT>
size_t CSC<RIT, VT, CPT>:: get_nrows()
{
	return nrows_;
}

template <typename RIT, typename VT, typename CPT>
size_t CSC<RIT, VT, CPT>:: get_nnz()
{
	return nnz_;
}

//TODO: need parallel code
template <typename RIT, typename VT, typename CPT>
bool CSC<RIT, VT, CPT>::operator==(const CSC<RIT, VT, CPT> & rhs)
{
    if(nnz_ != rhs.nnz_ || nrows_  != rhs.nrows_ || ncols_ != rhs.ncols_) return false;
    bool same = std::equal(colPtr_.begin(), colPtr_.begin()+ncols_+1, rhs.colPtr_.begin());
    same = same && std::equal(rowIds_.begin(), rowIds_.begin()+nnz_, rhs.rowIds_.begin());
    ErrorTolerantEqual<VT> epsilonequal(EPSILON);
    same = same && std::equal(nzVals_.begin(), nzVals_.begin()+nnz_, rhs.nzVals_.begin(), epsilonequal );
    return same;
}

template <typename RIT, typename VT, typename CPT>
void CSC<RIT, VT, CPT>::PrintInfo()
{
	std::cout << "CSC matrix: " << " Rows= " << nrows_  << " Columns= " << ncols_ << " nnz= " << nnz_ << std::endl;
}
// Construct an CSC object from COO
// This will be a widely used function
// Optimize this as much as possible
template <typename RIT, typename VT, typename CPT>
template <typename CIT>
CSC<RIT, VT, CPT>::CSC(COO<RIT, CIT, VT> & cooMat)
{
	Timer t;
	t.Start();
	nrows_ = cooMat.nrows();
	ncols_ = cooMat.ncols();
	nnz_ = cooMat.nnz();
	isWeighted_ = cooMat.isWeighted();
	cooMat.BinByCol(colPtr_, rowIds_, nzVals_);
	MergeDuplicateSort(std::plus<VT>());
	isColSorted_ = true;
	t.Stop();
	//PrintTime("CSC Creation Time", t.Seconds());
}



template <typename RIT, typename VT, typename CPT>
template <typename AddOp>
void CSC<RIT, VT, CPT>::MergeDuplicateSort(AddOp binop)
{
	pvector<RIT> sqNnzPerCol(ncols_);
#pragma omp parallel
	{
		pvector<std::pair<RIT, VT>> tosort;
#pragma omp for
        for(size_t i=0; i<ncols_; i++)
        {
            size_t nnzCol = colPtr_[i+1]-colPtr_[i];
            sqNnzPerCol[i] = 0;
            
            if(nnzCol>0)
            {
                if(tosort.size() < nnzCol) tosort.resize(nnzCol);
                
                for(size_t j=0, k=colPtr_[i]; j<nnzCol; ++j, ++k)
                {
                    tosort[j] = std::make_pair(rowIds_[k], nzVals_[k]);
                }
                
                //TODO: replace with radix or another integer sorting
                sort(tosort.begin(), tosort.begin()+nnzCol);
                
                size_t k = colPtr_[i];
                rowIds_[k] = tosort[0].first;
                nzVals_[k] = tosort[0].second;
                
                // k points to last updated entry
                for(size_t j=1; j<nnzCol; ++j)
                {
                    if(tosort[j].first != rowIds_[k])
                    {
                        rowIds_[++k] = tosort[j].first;
                        nzVals_[k] = tosort[j].second;
                    }
                    else
                    {
                        nzVals_[k] = binop(tosort[j].second, nzVals_[k]);
                    }
                }
                sqNnzPerCol[i] = k-colPtr_[i]+1;
          
            }
        }
    }
    
    
    // now squeze
    // need another set of arrays
    // Think: can we avoid this extra copy with a symbolic step?
    pvector<CPT>sqColPtr;
    ParallelPrefixSum(sqNnzPerCol, sqColPtr);
    nnz_ = sqColPtr[ncols_];
    pvector<RIT> sqRowIds(nnz_);
    pvector<VT> sqNzVals(nnz_);

#pragma omp parallel for
	for(size_t i=0; i<ncols_; i++)
	{
		size_t srcStart = colPtr_[i];
		size_t srcEnd = colPtr_[i] + sqNnzPerCol[i];
		size_t destStart = sqColPtr[i];
		std::copy(rowIds_.begin()+srcStart, rowIds_.begin()+srcEnd, sqRowIds.begin()+destStart);
		std::copy(nzVals_.begin()+srcStart, nzVals_.begin()+srcEnd, sqNzVals.begin()+destStart);
	}
	
	// now replace (just pointer swap)
	colPtr_.swap(sqColPtr);
	rowIds_.swap(sqRowIds);
	nzVals_.swap(sqNzVals);
	
}


template <typename RIT, typename VT, typename CPT> 
void CSC<RIT, VT, CPT>::count_sort(pvector<std::pair<RIT, VT> >& all_elements, size_t expon)
{


	size_t num_of_elements = all_elements.size();

	pvector<std::pair<RIT, VT> > temp_array(num_of_elements);
	RIT count[10] = {0};
	size_t index_for_count;

	for(size_t i = 0; i < num_of_elements; i++){
		index_for_count = ((all_elements[i].first)/expon)%10;
		count[index_for_count]++;
	}

	for(int i = 1; i < 10; i++){
		count[i] += count[i-1];
	}

	for(size_t i = num_of_elements-1; i > 0; i--){
		index_for_count = ((all_elements[i].first)/expon)%10;
		temp_array[count[index_for_count] -1] = all_elements[i];
		count[index_for_count]--;
	}


	index_for_count = ((all_elements[0].first)/expon)%10;
	temp_array[count[index_for_count] -1] = all_elements[0];

	all_elements = std::move(temp_array);

	return;
}




// here rowIds vector is to be sorted piece by piece given that there are no row repititions in a single column or any zero element in nzVals
template <typename RIT, typename VT, typename CPT>
void CSC<RIT, VT, CPT>::sort_inside_column()
{
	if(!isColSorted_)
	{
		#pragma omp parallel for
				for(size_t i=0; i<ncols_; i++)
				{
					size_t nnzCol = colPtr_[i+1]-colPtr_[i];
					
					pvector<std::pair<RIT, VT>> tosort(nnzCol);
					RIT max_row = 0;
					if(nnzCol>0)
					{
						//if(tosort.size() < nnzCol) tosort.resize(nnzCol);
						
						for(size_t j=0, k=colPtr_[i]; j<nnzCol; ++j, ++k)
						{
							tosort[j] = std::make_pair(rowIds_[k], nzVals_[k]);
							max_row = std::max(max_row, rowIds_[k]);
						}

						//sort(tosort.begin(), tosort.end());
						for(size_t expon = 1; max_row/expon > 0; expon *= 10){
							count_sort(tosort, expon);
						}
						
						size_t k = colPtr_[i];
						rowIds_[k] = tosort[0].first;
						nzVals_[k] = tosort[0].second;
						
						// k points to last updated entry
						for(size_t j=1; j<nnzCol; ++j)
						{
								rowIds_[++k] = tosort[j].first;
								nzVals_[k] = tosort[j].second;
						}
					}
				}
			

			isColSorted_ = true;
	}
	else{
		return;
	}
}

template <typename RIT, typename VT, typename CPT>
void CSC<RIT, VT, CPT>::column_split(std::vector< CSC<RIT, VT, CPT>* > &vec, int nsplits)
{   
    vec.resize(nsplits);
    int ncolsPerSplit = ncols_ / nsplits;
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
#pragma omp for
        for(int s = 0; s < nsplits; s++){
            CPT colStart = s * ncolsPerSplit;
            CPT colEnd = (s < nsplits-1) ? (s + 1) * ncolsPerSplit: ncols_;
            pvector<CPT> colPtr(colEnd - colStart + 1);
            for(int i = 0; colStart+i < colEnd + 1; i++){
                colPtr[i] = colPtr_[colStart + i] - colPtr_[colStart];
            }
            pvector<RIT> rowIds(rowIds_.begin() + colPtr_[colStart], rowIds_.begin() + colPtr_[colEnd]);
            pvector<VT> nzVals(nzVals_.begin() + colPtr_[colStart], nzVals_.begin() + colPtr_[colEnd]);
            vec[s] = new CSC<RIT, VT, CPT>(nrows_, colEnd - colStart, colPtr_[colEnd] - colPtr_[colStart], false, true);
            vec[s]->cols_pvector(&colPtr);
            vec[s]->nz_rows_pvector(&rowIds);
            vec[s]->nz_vals_pvector(&nzVals);
        }
    }
}

template <typename RIT, typename VT, typename CPT>
template <typename CIT>
COO<RIT, CIT, VT> CSC<RIT, VT, CPT>::to_COO(){
        pvector<RIT> coo_nzRows(nnz_);
        pvector<CIT> coo_nzCols(nnz_);
        pvector<VT> coo_nzVals(nnz_);


        auto idx = 0;
        for (auto i = 0; i < ncols_; i++){
            for(auto j = colPtr_[i]; j < colPtr_[i+1]; j++){
                coo_nzRows[idx] = rowIds_[j];
                coo_nzVals[idx] = nzVals_[j];
                coo_nzCols[idx] = i;
                idx++;
            }
        }

        return COO<RIT,CIT,VT> (nrows_, ncols_, nnz_, &coo_nzRows, &coo_nzCols, &coo_nzVals, isWeighted_);
}

#endif
