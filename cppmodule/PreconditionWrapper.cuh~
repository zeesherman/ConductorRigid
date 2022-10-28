#include "ConductorRigid.cuh"
#include <stdio.h>
#include "TextureTools.h"
#include <cusp/linear_operator.h>

#ifndef __PreconditionWrapper_CUH__
#define __PreconditionWrapper_CUH__ 

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

// command to convert floats or doubles to integers
#ifdef SINGLE_PRECISION
#define __scalar2int_rd __float2int_rd
#else
#define __scalar2int_rd __double2int_rd
#endif

// Construct class wrapper to use the grand potential matrix as a matrix-free method in CUSP.
class cuspPrecondition : public cusp::linear_operator<float, cusp::device_memory>
{
public:

   	typedef cusp::linear_operator<float, cusp::device_memory> super; // Defines size of linear operator

	unsigned int *d_isolated_membership; // isolated body tag each particle belongs to (from plugin)
	unsigned int *d_rigid_members; // indices of beads belonging to each rigid body
	unsigned int *d_rigid_size; // number of beads in each rigid body

	int group_size; // number of particles in active group
	int N_rigid; // number of rigid bodies
	int N_isolated;  // number of isolated bodies
	int max_rigid_size; // maximum rigid body body

	int block_size; // number of threads to use per block

    	// constructor
    	cuspPrecondition( 	unsigned int *d_isolated_membership,
				unsigned int *d_rigid_members,
				unsigned int *d_rigid_size,
				int group_size,
				int N_rigid,
				int N_isolated,
				int max_rigid_size,
				int block_size)
                      	      : super(group_size+N_isolated, group_size+N_isolated), 
				d_isolated_membership(d_isolated_membership), 
				d_rigid_members(d_rigid_members), 
				d_rigid_size(d_rigid_size),  
				group_size(group_size),
				N_rigid(N_rigid),
				N_isolated(N_isolated),
				max_rigid_size(max_rigid_size), 
				block_size(block_size){}


    // Perform the linear matrix multiplication operation y = A*x
    template <typename VectorType1,
             typename VectorType2>
    void operator()( VectorType1& x, VectorType2& y )
    {

        // obtain a raw pointer to device memory for input and output arrays
        float *x_ptr = thrust::raw_pointer_cast(&x[0]);
        float *y_ptr = thrust::raw_pointer_cast(&y[0]);

        // run kernels to compute y = A*x
	Precondition(x_ptr, y_ptr, d_isolated_membership, d_rigid_members, d_rigid_size, group_size, N_rigid, N_isolated, max_rigid_size,
		     block_size);

    }
};

#endif
