#include <cusp/monitor.h>
#include <cusp/array1d.h>
#include <cusp/krylov/gmres.h>
#include <cusp/krylov/cg.h>
#include <cusp/multiply.h>

#include <stdio.h>
#include "ConductorRigid.cuh"
#include "PotentialWrapper.cuh"
#include "PreconditionWrapper.cuh"
#include "TextureTools.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

#ifdef SINGLE_PRECISION
#define __scalar2int_rd __float2int_rd
#else
#define __scalar2int_rd __double2int_rd
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define PI 3.1415926535897932f
#define NOT_MEMBER 0xffffffff

scalar2_tex_t phiq_table_tex;
scalar2_tex_t gradphiq_table_tex;
scalar4_tex_t pos_tex;

// Zero any one-dimensional array
__global__ void zero_array(Scalar *d_array, // pointer to array
			   int N) // size of array
{
	// Get thread index
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	// Set array element to zero
	if (tid < N) {
		d_array[tid] = 0.0;
	}
}

// Zero forces (or any Scalar4 array)
__global__ void zeroforce(int Ntotal, // total number of particles (size of array)
			  Scalar4 *d_force) // pointer to force array 
{
	// Get thread index
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	// Set array element to zero
	if (tid < Ntotal) {
		d_force[tid] = make_scalar4( 0.0, 0.0, 0.0, 0.0 );
	}
}


cudaError_t gpu_ZeroForce(int Ntotal, Scalar4 *d_force, int block_size) {

	// Use one thread per particle
    	dim3 grid( (Ntotal/block_size) + 1, 1, 1);
    	dim3 threads(block_size, 1, 1);

	zeroforce<<<grid,threads>>>(Ntotal, d_force);

	return cudaSuccess;
}

// Set the grid to zero
__global__ void initialize_grid(CUFFTCOMPLEX *grid, int N) {

	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid < N) {
		grid[tid] = make_scalar2( 0.0, 0.0 );
	}

}

// Initialize membership arrays.  One array is for indexing within the active group while the other is for indexing isolated bodies in the active group.
__global__ void initialize_membership( unsigned int *d_group_membership, // active group index list
				       unsigned int *d_isolated_membership, // isolated body tag list
				       int Ntotal) // total number of particles
{
	// Global particle index
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	// Flag every particle as not a member of the groups
	if (idx < Ntotal) {
		d_group_membership[idx] = NOT_MEMBER;
		d_isolated_membership[idx] = NOT_MEMBER;
	}
}

// Update membership arrays, containing indices for all of the particles in the simulation.
//
// Active group membership:
// A particle with global index i that is not a member of the active group has d_group_membership[i] = NOT_MEMBER.
// A particle with global index i that is a member of the active group has its active group-specific index in d_group_membership[i].
// That is, d_group_members[d_group_membership[i]] = i.
//
// Isolated body membership:
// A particle with global index i that is not a member of the active group has d_isolated_membership[i] = NOT_MEMBER.
// A particle with global index i that is a member of the active group has its isolated body's tag in d_isolated_membership[i].
// Note that each rigid body and isolated bead count as isolated bodies and get different indices.
__global__ void update_membership( unsigned int *d_group_membership, // particle membership and group index list
				   unsigned int *d_group_members, // group members
				   int group_size, // number of particles belonging to the group of interest
				   unsigned int *d_tag, // global tag of all particles
				   unsigned int *d_isolated_membership_unsrt, // isolated body tag each particle belongs to (unsorted)
				   unsigned int *d_isolated_membership) // isolated body tag each particle belongs to (sorted)
{
	// Group-specific particle index
	unsigned int group_idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (group_idx < group_size) {
		
		// Global particle index
		unsigned int idx = d_group_members[group_idx];

		// Set the group-specific index at the current particle's global index position in the group membership list
		d_group_membership[idx] = group_idx;

		// Global tag of the current bead
		unsigned int tag = d_tag[idx];

		// Get the index of the rigid body to which this bead belongs
		d_isolated_membership[idx] = d_isolated_membership_unsrt[tag];
	}
}

// Spread particle charges to a uniform grid.  Use a P-by-P-by-P block per particle with a thread per grid node. 
__global__ void spread( Scalar4 *d_pos, // particle positions
			Scalar *d_charge, // particle charges
			unsigned int *d_group_members, // particles in active group
			BoxDim box, // simulation box
			int group_size, // number of particles belonging to the group of interest	
			Scalar3 eta, // Spectral splitting parameter			
			int Nx, // number of grid nodes in x dimension 
			int Ny, // number of grid nodes in y dimension
			int Nz, // number of grid nodes in z dimension
			Scalar3 gridh, // grid spacing
			int P, // number of nodes to spread the particle dipole over
			CUFFTCOMPLEX *qgrid, // grid on which to spread the charges
			Scalar xiterm, // precomputed term 2*xi^2
			Scalar prefac ) // precomputed term (2*xi^2/PI)^(3/2)*1/(eta.x*eta.y*eta.z)^(1/2)
{

	// Setup shared memory for the particle position
	__shared__ float3 pos_shared_mem;
	float3 *pos_shared = &pos_shared_mem;	

	// Get the block index and linear index of the thread within the block
	int group_idx = blockIdx.x;
	int thread_offset = threadIdx.z + threadIdx.y*blockDim.z + threadIdx.x*blockDim.z*blockDim.y;

	// Global index of current particle
	int idx = d_group_members[group_idx];

	// Have the first thread fetch the particle position and store it in shared memory
	if (thread_offset == 0) {

		Scalar4 tpos = texFetchScalar4(d_pos, pos_tex, idx);
		pos_shared[0].x = tpos.x;
		pos_shared[0].y = tpos.y;
		pos_shared[0].z = tpos.z;
	}
	
	// Box size
	Scalar3 L = box.getL();
	Scalar3 halfL = L/2.0;

	// Current particle's charge
	Scalar qj = d_charge[idx];

	// Wait for the particle position to be written to shared memory before proceeding
	__syncthreads();

	// Retrieve position from shared memory
	Scalar3 pos = *pos_shared;

	// Fractional position within box (0 to 1)
	Scalar3 pos_frac = box.makeFraction(pos);

	// Particle position in units of grid spacing
	pos_frac.x *= (Scalar)Nx;
	pos_frac.y *= (Scalar)Ny;
	pos_frac.z *= (Scalar)Nz;

	// Determine index of the grid node immediately preceeding (in each dimension) the current particle
	int x = int(pos_frac.x);
	int y = int(pos_frac.y);
	int z = int(pos_frac.z);

	// Index of grid point associated with current thread.  For an even number of support nodes, support an equal number of nodes above and below the particle.  For an odd number of support nodes, support an equal number of nodes above and below the closest grid node (which is not necessarily node (x,y,z) ).   
	int halfP = P/2;
	int x_inp = x + threadIdx.x - halfP + 1 - (P % 2)*( (pos_frac.x-Scalar(x)) < 0.5 );
	int y_inp = y + threadIdx.y - halfP + 1 - (P % 2)*( (pos_frac.y-Scalar(y)) < 0.5 );
	int z_inp = z + threadIdx.z - halfP + 1 - (P % 2)*( (pos_frac.z-Scalar(z)) < 0.5 );

	// Position of the current grid node (w/o periodic boundaries)
	Scalar3 pos_grid;
	pos_grid.x = gridh.x*x_inp - halfL.x;
	pos_grid.y = gridh.y*y_inp - halfL.y;
	pos_grid.z = gridh.z*z_inp - halfL.z;

	// Distance from particle to grid node and r^2/eta
	Scalar3 r = pos_grid - pos;
	Scalar r2eta = r.x*r.x/eta.x + r.y*r.y/eta.y + r.z*r.z/eta.z;

	// Contribution to the current grid node from the current particle charge
	Scalar charge_inp = prefac*expf( -xiterm*r2eta )*qj;

	// Grid node index accounting for periodicity
	x_inp = (x_inp<0) ? x_inp+Nx : ( (x_inp>Nx-1) ? x_inp-Nx : x_inp);
	y_inp = (y_inp<0) ? y_inp+Ny : ( (y_inp>Ny-1) ? y_inp-Ny : y_inp);
	z_inp = (z_inp<0) ? z_inp+Nz : ( (z_inp>Nz-1) ? z_inp-Nz : z_inp);
	int grid_idx = x_inp*Ny*Nz + y_inp*Nz + z_inp;

	// Add particle contribution to the grid
	atomicAdd( &(qgrid[grid_idx].x), charge_inp);

}

// Scale the transformed gridded charges
__global__ void scale(  CUFFTCOMPLEX *qgrid, // transform of the gridded charges
			Scalar *d_scale_phiq,  // scaling factor associated with each grid point
			int Ngrid)  // total number of grid nodes
{
  	// Current thread's linear index
  	int tid = blockDim.x * blockIdx.x + threadIdx.x;

  	// Ensure thread index corresponds to a node within the grid
  	if ( tid < Ngrid ) {

    		// Read the gridded value from global memory
    		Scalar2 fqgrid = qgrid[tid];

    		// Current wave-space scaling
		Scalar f_phiq = d_scale_phiq[tid];

    		// Write the scaled grid value to global memory.
		qgrid[tid] = make_scalar2( f_phiq*fqgrid.x, f_phiq*fqgrid.y );

  	}
}

// Contract the grid to the particle centers
__global__ void contract(	Scalar4 *d_pos,  // particle positions
				unsigned int *d_group_members, // pointer to array of particles belonging to the group
				BoxDim box, // simulation box
				int group_size, // number of particles belonging to the group of interest
				Scalar xi, //  Ewald splitting parameter
				Scalar3 eta, // Spectral splitting parameter
				int Nx, // number of grid nodes in x dimension
				int Ny, // number of grid nodes in y dimension
				int Nz, // number of grid nodes in z dimension
				Scalar3 gridh, // grid spacing
				int P, // number of nodes to spread the particle dipole over
				CUFFTCOMPLEX *d_qgrid, // gridded potentials 
				Scalar *d_phiq, // pointer to the output array 
				Scalar xiterm, // precomputed term 2*xi^2
				Scalar prefac) // precomputed term (2*xi^2/PI)^(3/2)*1/(eta.x*eta.y*eta.z)^(1/2)*gridh.x*gridh.y*gridh.z
{
	// Setup shared memory for particle output and particle positions
	extern __shared__ float shared[];
	float *phiq = shared;
	float *pos_shared = &shared[P*P*P];

	// Get the block index, linear index of the thread within the block, and the number of threads in the block
	int group_idx = blockIdx.x;
	int thread_offset = threadIdx.z + threadIdx.y*blockDim.z + threadIdx.x*blockDim.z*blockDim.y;
	int blocksize = blockDim.x*blockDim.y*blockDim.z;

	// Global particle ID
    	int idx = d_group_members[group_idx];

	// Initialize the shared memory and have the first thread fetch the particle position and store it in shared memory
	phiq[thread_offset] = 0.0;
	if (thread_offset == 0){
		Scalar4 tpos = texFetchScalar4(d_pos, pos_tex, idx);
		pos_shared[0] = tpos.x;
		pos_shared[1] = tpos.y;
		pos_shared[2] = tpos.z;
	}
	
	// Box size
	Scalar3 L = box.getL();
	Scalar3 halfL = L/2.0;

	// Wait for the particle position to be written to shared memory before proceeding
	__syncthreads();

	// Fetch position from shared memory
	Scalar3 pos = make_scalar3(pos_shared[0], pos_shared[1], pos_shared[2]);

    	// Express the particle position in units of grid spacing.
    	Scalar3 pos_frac = box.makeFraction(pos);
    	pos_frac.x *= (Scalar)Nx;
    	pos_frac.y *= (Scalar)Ny;
    	pos_frac.z *= (Scalar)Nz;

    	// Determine index of the grid node immediately preceeding (in each dimension) the current particle
	int x = int(pos_frac.x);
	int y = int(pos_frac.y);
	int z = int(pos_frac.z);

	// Index of grid point associated with current thread.  For an even number of support nodes, support an equal number of nodes above and below the particle.  For an odd number of support nodes, support an equal number of nodes above and below the closest grid node (which is not necessarily node (x,y,z) ).
	int halfP = P/2;
	int x_inp = x + threadIdx.x - halfP + 1 - (P % 2)*( (pos_frac.x-Scalar(x)) < 0.5 );
	int y_inp = y + threadIdx.y - halfP + 1 - (P % 2)*( (pos_frac.y-Scalar(y)) < 0.5 );
	int z_inp = z + threadIdx.z - halfP + 1 - (P % 2)*( (pos_frac.z-Scalar(z)) < 0.5 );

	// Position of the current grid node (w/o periodic boundaries)
	Scalar3 pos_grid;
	pos_grid.x = gridh.x*x_inp - halfL.x;
	pos_grid.y = gridh.y*y_inp - halfL.y;
	pos_grid.z = gridh.z*z_inp - halfL.z;

	// Distance from particle to grid node and "r^2/eta"
	Scalar3 r = pos - pos_grid;
	Scalar r2eta = r.x*r.x/eta.x + r.y*r.y/eta.y + r.z*r.z/eta.z;

	// Grid node index accounting for periodicity
	x_inp = (x_inp<0) ? x_inp+Nx : ( (x_inp>Nx-1) ? x_inp-Nx : x_inp);
	y_inp = (y_inp<0) ? y_inp+Ny : ( (y_inp>Ny-1) ? y_inp-Ny : y_inp);
	z_inp = (z_inp<0) ? z_inp+Nz : ( (z_inp>Nz-1) ? z_inp-Nz : z_inp);
	int grid_idx = x_inp*Ny*Nz + y_inp*Nz + z_inp;

	// Contribution to the current particle from the current grid node.
	phiq[thread_offset] = prefac*expf( -xiterm*r2eta )*d_qgrid[grid_idx].x;	

	// Reduction to add all of the P^3 values
	int offs = blocksize;
	int oldoffs;
	while ( offs > 1){
		oldoffs = offs; // store the previous value of offs
		offs = (offs+1)/2; // the current value of offs is half of the previous (result is a rounded up integer)
		__syncthreads();
		if ( thread_offset + offs < oldoffs ){
			phiq[thread_offset] += phiq[thread_offset + offs];
		}
	}

	if (thread_offset == 0){
		// Store the current particle's output
		d_phiq[group_idx] = phiq[0];
	}
}

// Contract the grid to forces on particles
__global__ void contract_force(	Scalar4 *d_pos,  // particle positions
				Scalar *d_charge, // particle charges
				Scalar4 *d_force, // pointer to particle forces
				unsigned int *d_group_members, // pointer to array of particles belonging to the group
				BoxDim box, // simulation box
				int group_size, // number of particles in active group
				Scalar3 eta, // Spectral splitting parameter
				int Nx, // number of grid nodes in x dimension
				int Ny, // number of grid nodes in y dimension
				int Nz, // number of grid nodes in z dimension
				Scalar3 gridh, // grid spacing
				int P, // number of nodes to spread the particle dipole over
				CUFFTCOMPLEX *d_qgrid, // potential/charge grid
				Scalar xiterm, // precomputed term 2*xi^2
				Scalar prefac ) // precomputed term (2*xi^2/PI)^(3/2)*1/(eta.x*eta.y*eta.z)^(1/2)*gridh.x*gridh.y*gridh.z
{
	// Setup shared memory for particle forces and particle positions
	extern __shared__ float3 shared_force[];
	float3 *force = shared_force;
	float3 *pos_shared = &shared_force[P*P*P];

	// Get the block index, linear index of the thread within the block, and the number of threads in the block
	int group_idx = blockIdx.x;
	int thread_offset = threadIdx.z + threadIdx.y*blockDim.z + threadIdx.x*blockDim.z*blockDim.y;
	int blocksize = blockDim.x*blockDim.y*blockDim.z;

	// Global particle ID
    	int idx = d_group_members[group_idx];

	// Initialize the shared memory and have the first thread fetch the particle position and store it in shared memory
	force[thread_offset] = make_scalar3(0.0,0.0,0.0);
	if (thread_offset == 0){
		Scalar4 tpos = texFetchScalar4(d_pos, pos_tex, idx);
		pos_shared[0].x = tpos.x;
		pos_shared[0].y = tpos.y;
		pos_shared[0].z = tpos.z;
	}
	
	// Current particle's charge
	Scalar qi = d_charge[idx];

	// Box size
	Scalar3 L = box.getL();
	Scalar3 halfL = L/2.0;

	// Wait for the particle position to be written to shared memory before proceeding
	__syncthreads();

	// Fetch position from shared memory
	Scalar3 pos = pos_shared[0];

    	// Express the particle position in units of grid spacing.
    	Scalar3 pos_frac = box.makeFraction(pos);
    	pos_frac.x *= (Scalar)Nx;
    	pos_frac.y *= (Scalar)Ny;
    	pos_frac.z *= (Scalar)Nz;

    	// Determine index of the grid node immediately preceeding (in each dimension) the current particle
	int x = int(pos_frac.x);
	int y = int(pos_frac.y);
	int z = int(pos_frac.z);

	// Index of grid point associated with current thread.  For an even number of support nodes, support an equal number of nodes above and below the particle.  For an odd number of support nodes, support an equal number of nodes above and below the closest grid node (which is not necessarily node (x,y,z) ).
	int halfP = P/2;
	int x_inp = x + threadIdx.x - halfP + 1 - (P % 2)*( (pos_frac.x-Scalar(x)) < 0.5 );
	int y_inp = y + threadIdx.y - halfP + 1 - (P % 2)*( (pos_frac.y-Scalar(y)) < 0.5 );
	int z_inp = z + threadIdx.z - halfP + 1 - (P % 2)*( (pos_frac.z-Scalar(z)) < 0.5 );

	// Position of the current grid node (w/o periodic boundaries)
	Scalar3 pos_grid;
	pos_grid.x = gridh.x*x_inp - halfL.x;
	pos_grid.y = gridh.y*y_inp - halfL.y;
	pos_grid.z = gridh.z*z_inp - halfL.z;

	// Distance from particle to grid node and combinations with eta
	Scalar3 r = pos - pos_grid;
	Scalar3 reta = r/eta;
	Scalar r2eta = r.x*reta.x + r.y*reta.y + r.z*reta.z;

	// Grid node index accounting for periodicity
	x_inp = (x_inp<0) ? x_inp+Nx : ( (x_inp>Nx-1) ? x_inp-Nx : x_inp);
	y_inp = (y_inp<0) ? y_inp+Ny : ( (y_inp>Ny-1) ? y_inp-Ny : y_inp);
	z_inp = (z_inp<0) ? z_inp+Nz : ( (z_inp>Nz-1) ? z_inp-Nz : z_inp);
	int grid_idx = x_inp*Ny*Nz + y_inp*Nz + z_inp;

	// Contribution to the current particle from the current grid node
	force[thread_offset] = 2.0*prefac*xiterm*qi*d_qgrid[grid_idx].x*expf( -xiterm*r2eta )*reta;

	// Reduction to add all of the P^3 values
	int offs = blocksize;
	int oldoffs;
	while ( offs > 1){
		oldoffs = offs; // store the previous value of offs
		offs = (offs+1)/2; // the current value of offs is half of the previous (result is a rounded up integer)
		__syncthreads();
		if ( thread_offset + offs < oldoffs ){
			force[thread_offset] += force[thread_offset + offs];
		}
	}

	if (thread_offset == 0){
		d_force[idx] = make_scalar4(force[0].x, force[0].y, force[0].z, 0.0);
	}
}

// Add real space contribution to particle potentials
__global__ void real_space( Scalar4 *d_pos, // particle positions and types
			    Scalar *d_charge, // particle charges
			    Scalar *d_isolated_potential, // isolated body potentials
			    unsigned int *d_group_membership, // particle membership and index in group
			    unsigned int *d_group_members, // pointer to array of particles belonging to the group
			    unsigned int *d_isolated_membership, // isolated body tag each particle belongs to
			    BoxDim box, // simulation box
			    int group_size, // number of particles in active group
			    Scalar *d_phiq, // pointer to result of M_phiq * q
			    Scalar rc, // real space cutoff radius
			    int Ntable, // number of entries in the real space table
			    Scalar drtable, // spacing between table entries
			    Scalar2 *d_phiq_table, // real space potential/charge table
			    const unsigned int *d_nlist, // pointer to the neighbor list  
			    const unsigned int *d_head_list, // index used to access elements of the neighbor list
			    const unsigned int *d_n_neigh, // pointer to the number of neighbors of each particle
			    Scalar selfcoeff) // coefficient of the self term
{
  	// Index for current particle
  	int group_idx = blockDim.x * blockIdx.x + threadIdx.x;

	// Ensure that current particle is within the group
  	if (group_idx < group_size) {

		// Global ID of current particle
		unsigned int idx = d_group_members[group_idx];
		
		// Get the tag of the isolated body to which the current particle belongs
		unsigned int body_tag = d_isolated_membership[idx];

		// Get the wave space contribution to M_phiq * q
  		Scalar phiq = d_phiq[group_idx];

		// Charge of the current particle
		Scalar qi = d_charge[idx];

		// Add the real space self term
		phiq += selfcoeff*qi;

		// Number of neighbors and location of neighbors in neighbor list for current particle
		unsigned int n_neigh = d_n_neigh[idx];
		unsigned int head_i = d_head_list[idx];

		// Current particle position and type
		Scalar4 postypei = texFetchScalar4(d_pos, pos_tex, idx);
		Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);

		// Minimum and maximum distances squared for pair calculation
		Scalar rc2 = rc*rc;
		Scalar rmin2 = drtable*drtable;

		// Loop over neighbors
    		for (int j=0; j < n_neigh; j++) {
			
			// Get neighbor global and group index
			unsigned int neigh_idx = d_nlist[head_i + j];
			unsigned int neigh_group_idx = d_group_membership[neigh_idx];

			// Check if neighbor is a member of the group of interest
			if ( neigh_group_idx != NOT_MEMBER ) {

				// Position and type of neighbor particle
				Scalar4 postypej = texFetchScalar4(d_pos, pos_tex, neigh_idx);
				Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

				// Distance vector between current particle and neighbor
        			Scalar3 r = posi - posj;
        			r = box.minImage(r); // nearest image distance vector
        			Scalar dist2 = dot(r,r); // distance squared

				// Add neighbor contribution if it is within the real space cutoff radius
       				if ( ( dist2 < rc2 ) && ( dist2 >= rmin2 ) ) {

					Scalar dist = sqrtf(dist2); // distance
					r = r/dist; // convert r to a unit vector

					// Charge of neighbor particle
					Scalar qj = d_charge[neigh_idx];
					
					// Read the table values closest to the current distance
					int tableind = __scalar2int_rd( Ntable * (dist-drtable)/(rc-drtable) );	
					Scalar2 entry = texFetchScalar2(d_phiq_table, phiq_table_tex, tableind);

					// Linearly interpolate between the table values
					Scalar lininterp = dist/drtable - tableind - Scalar(1.0);
					Scalar A = entry.x + ( entry.y - entry.x )*lininterp; 

					// Real-space contributions to M_phiq * q.
					phiq += A*qj;
	
      				} // end neighbor contribution
			} // end membership check
		}// end neighbor loop

		// Subtract the result from the rigid body potential and write to the current bead's position in the output array
		d_phiq[group_idx] = d_isolated_potential[body_tag] - phiq;

	}
}

// Add real space contribution to particle forces
__global__ void real_space_force( 	Scalar4 *d_pos, // particle positions and types
					Scalar *d_charge, // particle charges
					Scalar4 *d_force, // pointer to particle forces
					unsigned int *d_group_membership, // particle membership and index in group
					unsigned int *d_group_members, // pointer to array of particles belonging to the group
					BoxDim box, // simulation box
					Scalar3 E0, // external field
					int group_size, // number of particles in active group
					Scalar rc, // real space cutoff radius
					int Ntable, // number of entries in the real space table
					Scalar drtable, // spacing between table entries
					Scalar2 *d_gradphiq_table, // potential/charge gradient real space table 
					const unsigned int *d_nlist, // pointer to the neighbor list 
					const unsigned int *d_head_list, // index used to access elements of the neighbor list
					const unsigned int *d_n_neigh) // pointer to the number of neighbors of each particle
{
  	// Index for current particle
  	int group_idx = blockDim.x * blockIdx.x + threadIdx.x;

	// Ensure that current particle is within the group
  	if (group_idx < group_size) {

		// Global ID of current particle
		unsigned int idx = d_group_members[group_idx];

		// Get the reciprocal contribution to the force
  		Scalar4 F4 = d_force[idx];
		Scalar3 F = make_scalar3(F4.x, F4.y, F4.z);

		// Charge of current particle
		Scalar qi = d_charge[idx];

		// Add the constant charge/external field force
		F += qi*E0;

		// Number of neighbors and location of neighbors in neighbor list for current particle
		unsigned int n_neigh = d_n_neigh[idx];
		unsigned int head_i = d_head_list[idx];

		// Current particle position and type
		Scalar4 postypei = texFetchScalar4(d_pos, pos_tex, idx);
		Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);

		// Minimum and maximum distances squared for pair calculation
		Scalar rc2 = rc*rc;
		Scalar rmin2 = drtable*drtable;

		// Loop over neighbors
    		for (int j=0; j < n_neigh; j++) {

			// Get neighbor global and group index
			unsigned int neigh_idx = d_nlist[head_i + j];
			unsigned int neigh_group_idx = d_group_membership[neigh_idx];

			// Check if neighbor is a member of the group of interest
			if ( neigh_group_idx != NOT_MEMBER ) {

				// Position and type of neighbor particle
				Scalar4 postypej = texFetchScalar4(d_pos, pos_tex, neigh_idx);
				Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

				// Distance vector between current particle and neighbor
        			Scalar3 r = posi - posj;
        			r = box.minImage(r); // nearest image distance vector
        			Scalar dist2 = dot(r,r); // distance squared

				// Add neighbor contribution if it is within the real space cutoff radius
       				if ( ( dist2 < rc2 ) && ( dist2 >= rmin2 ) ) {

					Scalar dist = sqrtf(dist2); // distance
					r = r/dist; // convert r to a unit vector

					// Charge of neighbor particle
					Scalar qj = d_charge[neigh_idx];
	
					// Find the entries in the real space tables between which to interpolate
					int tableind = __scalar2int_rd( Ntable * (dist-drtable)/(rc-drtable) );
					Scalar2 gradphiq_entry = texFetchScalar2(d_gradphiq_table, gradphiq_table_tex, tableind);	

					// Interpolate between the values in the tables
					Scalar lininterp = dist/drtable - tableind - Scalar(1.0);
					Scalar A = gradphiq_entry.x + ( gradphiq_entry.y - gradphiq_entry.x )*lininterp;

					// Real-space contributions to the force
					F += -A*qi*qj*r; // charge/charge

      				} // end neighbor contribution
			} // end membership check
		}// end neighbor loop

		// Write the result to the current particle's force
		F4 = make_scalar4(F.x, F.y, F.z, 0.0);
		d_force[idx] = F4;
	}
}

// Sum the bead charges on each rigid body
__global__ void sum_charges( Scalar *d_bead_charge, // bead charges
			     Scalar *d_isolated_charge, // pointer to output array for total isolated body charges
			     unsigned int *d_isolated_membership, // isolated body tag each bead belongs to
			     unsigned int *d_rigid_members, // indices of bead belonging to each rigid body
			     unsigned int *d_rigid_size, // number of beads in each rigid body
			     int max_rigid_size) // maximum rigid body size
{

	// Setup shared memory for bead charges
	extern __shared__ float shared[];

	// Get indices
	int body_idx = blockIdx.x; // current rigid body index
	int bead_offset = threadIdx.x; // current bead offset (index in rigid body)
	int thread_offset = threadIdx.x; // thread offset (index in block)
	int block_size = blockDim.x; // number of threads in block
	
	// Get the rigid body size
	int rigid_size = d_rigid_size[body_idx];

	// Ensure current bead is within the current body
	unsigned int isolated_tag;
	if (bead_offset < rigid_size) {
		
		// Get the current bead global index
		unsigned int bead_idx = d_rigid_members[body_idx*max_rigid_size + bead_offset];
		
		// Get the current isolated body tag
		isolated_tag = d_isolated_membership[bead_idx];

		// Write the current bead charge to shared memory.
		shared[thread_offset] = d_bead_charge[bead_idx];

		// Increment the bead offset
		bead_offset += block_size;

	}

	// Continue adding bead charges if the block size is less than the body size
	while (bead_offset < rigid_size) {

		unsigned int bead_idx = d_rigid_members[body_idx*max_rigid_size + bead_offset];
		shared[thread_offset] += d_bead_charge[bead_idx];
		bead_offset += block_size;
	}

	// Reduction to add all of the bead charges
	int offs = ( rigid_size < block_size ) ? rigid_size : block_size;
	int oldoffs;
	while ( offs > 1){
		oldoffs = offs; // store the previous value of offs
		offs = (offs+1)/2; // the current value of offs is half of the previous (result is a rounded up integer)
		__syncthreads();
		if ( thread_offset + offs < oldoffs ){
			shared[thread_offset] += shared[thread_offset + offs];
		}
	}

	// Have a single thread write the rigid body charge
	if (thread_offset == 0){
		d_isolated_charge[isolated_tag] = shared[0];
	}

}

// Set the charge of isolated beads in the output array of the matrix/vector multiply.  This is done in a separate kernel from sum_charges because it avoids having to allocate shared memory for each of the isolated beads.
__global__ void set_charge( Scalar *d_bead_charge, // bead charges
			    Scalar *d_isolated_charge, // pointer to output array of isolated body charges
			    unsigned int *d_group_members,  // members of the active group
			    unsigned int *d_isolated_membership, // isolated body tag each particle belongs to (from plugin)
			    unsigned int *d_rigid_membership, // rigid body index each particle belongs to (from HOOMD); isolated beads get -1
			    int group_size) // size of the active group
{
	// Active group index of current bead
	int group_idx = threadIdx.x + blockIdx.x*blockDim.x;

	// Exclude extra threads
	if (group_idx < group_size) {

		// Global index of current bead
		int idx = d_group_members[group_idx];

		// Only consider isolated beads. Rigid body charges are set in the sum_charges kernel.
		if (d_rigid_membership[idx] == NOT_MEMBER) {
		
			// Tag of the isolated body 
			unsigned int isolated_tag = d_isolated_membership[idx];
		
			// Set the bead charge to the correct location in the output array
			d_isolated_charge[isolated_tag] = d_bead_charge[idx];

		} // end isolated bead if statement

	} // end extra thread if statement
}

// Compute the objective output to the matrix/vector multiply: [-r*E_0, Q]^T
__global__ void compute_objective( Scalar4 *d_bead_pos,
				   Scalar4 *d_rigid_pos,
				   Scalar *d_isolated_charge,
				   unsigned int *d_group_membership,
				   unsigned int *d_rigid_members,
				   BoxDim box,
				   Scalar3 E0,
				   int group_size,
				   int N_isolated,
				   int max_rigid_size,
				   int len_rigid_array,
				   Scalar *d_objective)
{
	// Get thread index
	int thread_idx = threadIdx.x + blockIdx.x*blockDim.x;

	// The first len_rigid_array threads set -r*E_0 for each particle
	if (thread_idx < len_rigid_array) {

		// Get particle's global index
		unsigned int idx = d_rigid_members[thread_idx];

		// Check that entry actually corresponds to a particle in a rigid body
		if (idx != NOT_MEMBER) {

			// Get particle's position
			Scalar4 bead_postype = texFetchScalar4(d_bead_pos, pos_tex, idx);
			Scalar3 bead_pos = make_scalar3(bead_postype.x, bead_postype.y, bead_postype.z);

			// Get rigid body's center of mass position
			Scalar4 rigid_postype = d_rigid_pos[thread_idx/max_rigid_size];
			Scalar3 rigid_pos = make_scalar3(rigid_postype.x, rigid_postype.y, rigid_postype.z);

			// Particle position relative to the rigid body center of mass
        		Scalar3 r = bead_pos - rigid_pos;
        		r = box.minImage(r); // nearest image distance vector

			// Get particle's active group index
			unsigned int group_idx = d_group_membership[idx];

			// Compute the objective output and store it in the current particle's index
			d_objective[group_idx] = -(r.x*E0.x + r.y*E0.y + r.z*E0.z);
		}

	  // The next N_isolated threads set the known isolated body charges to the correct place in the output array
	} else if (thread_idx < len_rigid_array + N_isolated){

		d_objective[thread_idx - len_rigid_array + group_size] = d_isolated_charge[thread_idx - len_rigid_array];
	}
}

// Apply the preconditioner to a vector.
__global__ void precondition(Scalar *d_input,  // input to preconditioner (bead charges and isolated potentials)
			     Scalar *d_output,  // output of preconditioner
			     unsigned int *d_isolated_membership, // isolated body tag each particle belongs to
			     unsigned int *d_rigid_members, // indices of beads belonging to each rigid body
			     unsigned int *d_rigid_size, // number of beads in each rigid body
			     int group_size, // number of particles in active group
			     int N_rigid, // number of rigid bodies
			     int N_isolated, // number of isolated bodies
			     int max_rigid_size) // maximum rigid body size
{
	// Get the thread index
	int thread_idx = blockIdx.x*blockDim.x + threadIdx.x;

	// The first group_size threads scale the bead charges
	if (thread_idx < group_size) {

		d_output[thread_idx] = -4.0*PI*d_input[thread_idx];

	// The next N_isolated threads scale the isolated body potentials
	} else if (thread_idx < group_size + N_isolated) {

		d_output[thread_idx] = d_input[thread_idx]/(4.0*PI);

		// Rigid bodies need an extra scaling
		if (thread_idx < group_size + N_rigid) {
			
			// Current rigid body index
			int rigid_idx = thread_idx - group_size;

			// Get the rigid body size
			int rigid_size = d_rigid_size[rigid_idx];

			// Get the global index of one of the beads in the body
			int bead_idx = d_rigid_members[rigid_idx*max_rigid_size];

			// Get the isolated body tag of the rigid body
			int isolated_tag = d_isolated_membership[bead_idx];

			// Scale the correct isolated body potential
			d_output[group_size + isolated_tag] = d_input[group_size + isolated_tag]/rigid_size;

		} // end extra rigid body scaling
	} // end isolated body scaling
} // end kernel

// Perform the top row of the matrix/vector multiply: Sigma^T * (Phi - Phi_0) - M_phiq * q
cudaError_t PotentialChargeMultiply(    Scalar4 *d_bead_pos, // particle posisitons
					Scalar *d_bead_charge, // particle charge
					Scalar *d_isolated_potential, // isolated body potentials

					unsigned int *d_group_membership, // particle index in active group
					unsigned int *d_group_members, // particles in active group
					unsigned int *d_isolated_membership, // isolated body tag each particle belongs to
					
					const BoxDim& box, // simulation box
					int group_size, // number of particles in active group

					int block_size, // number of threads to use per block
					Scalar *d_phiq, // pointer to result of M_phiq * q

					Scalar xi, // Ewald splitting parameter
					Scalar3 eta, // spectral splitting parameter
					Scalar rc, // real space cutoff radius
					int Nx, // number of grid nodes in the x dimension
					int Ny, // number of grid nodes in the y dimension
					int Nz, // number of grid nodes in the z dimension
					Scalar3 gridh, // grid spacing
					int P, // number of grid nodes over which to spread and contract

					CUFFTCOMPLEX *d_qgrid, // wave space charge grid
					Scalar *d_scale_phiq, // potential/dipole scaling on grid
					cufftHandle plan, // plan for the FFTs

					int Ntable, // number of entries in the real space coefficient table
					Scalar drtable, // real space coefficient table spacing
					Scalar2 *d_phiq_table, // pointer to potential/charge real space table

					const unsigned int *d_nlist, // neighbor list
					const unsigned int *d_head_list, // used to access entries in the neighbor list
					const unsigned int *d_n_neigh) // number of neighbors of each particle
{
	// total number of grid nodes
 	int Ngrid = Nx*Ny*Nz;

	// for initialization and scaling, use one thread per grid node
    	int Nthreads1 = ( Ngrid > block_size ) ? block_size : Ngrid;
    	int Nblocks1 = ( Ngrid - 1 )/Nthreads1 + 1;

	// For spreading and contracting, use one P-by-P-by-P block per particle.
	dim3 Nblocks2(group_size, 1, 1); // grid is a 1-D array of N blocks, where N is number of particles
	dim3 Nthreads2(P, P, P); // block is 3-D array of P^3 threads

    	// for the real space calculation, use one thread per particle
    	dim3 Nblocks3( (group_size/block_size) + 1, 1, 1);
    	dim3 Nthreads3(block_size, 1, 1);

	// Factors needed for the kernels
	Scalar quadW = gridh.x*gridh.y*gridh.z; // trapezoidal rule weights
	Scalar xi2 = xi*xi;
	Scalar xiterm = 2.0*xi2;
	Scalar prefac = xiterm*xi/PI*sqrtf(2.0/(PI*eta.x*eta.y*eta.z));  // prefactor for the spreading and contracting exponentials
	Scalar selfterm = (1.0 - exp(-4.0*xi2))/(8.0*PI*sqrt(PI)*xi) + erfc(2.0*xi)/(4.0*PI); // real space self term

    	// Reset the grid values to zero
	initialize_grid<<<Nblocks1, Nthreads1>>>(d_qgrid, Ngrid);

	// Spread dipoles from the particles to the grid
	spread<<<Nblocks2, Nthreads2>>>(d_bead_pos, d_bead_charge, d_group_members, box, group_size, eta, Nx, Ny, Nz, gridh, P, d_qgrid,
					xiterm, prefac);

	//  Compute the Fourier transform of the gridded data
    	cufftExecC2C(plan, d_qgrid, d_qgrid, CUFFT_FORWARD);

	// Scale the grid values
    	scale<<<Nblocks1, Nthreads1>>>(d_qgrid, d_scale_phiq, Ngrid);

	// Inverse Fourier transform the gridded data
    	cufftExecC2C(plan, d_qgrid, d_qgrid, CUFFT_INVERSE);

	// Contract the gridded values to the particles
	contract<<<Nblocks2, Nthreads2, (P*P*P+3)*sizeof(float)>>>(d_bead_pos, d_group_members, box, group_size, xi, eta, Nx, Ny, Nz, gridh,
								   P, d_qgrid, d_phiq, xiterm, quadW*prefac);   

	// Compute the real space contribution
    	real_space<<<Nblocks3, Nthreads3>>>(d_bead_pos, d_bead_charge, d_isolated_potential, d_group_membership, d_group_members,
					    d_isolated_membership, box, group_size, d_phiq, rc, Ntable, drtable, d_phiq_table, d_nlist,
					    d_head_list, d_n_neigh, selfterm); 

    	gpuErrchk(cudaPeekAtLastError());
    	return cudaSuccess;
}

// Perform the matrix/vector multiply.
//
//  [ -M_phiq   Sigma^T ]   [      q      ]
//  [  Sigma       0    ] * [ Phi - Phi_0 ]
//
cudaError_t MatrixVectorMultiply(   	Scalar4 *d_bead_pos, // particle posisitons
					Scalar *d_input, // particle charges and isolated body potentials

					unsigned int *d_group_membership, // particle index in active group
					unsigned int *d_group_members, // particles in active group
					unsigned int *d_isolated_membership, // isolated body tag each particle belongs to
					unsigned int *d_rigid_membership, //rigid index each particle belongs to; isolated beads get NOT_MEMBER
					unsigned int *d_rigid_members, // indices of beads belonging to each rigid body
					unsigned int *d_rigid_size, // number of beads in each rigid body

					const BoxDim& box, // simulation box
					int group_size, // number of particles in active group
					int N_rigid, // number of rigid bodies
					int max_rigid_size, // maximum rigid body body

					int block_size, // number of threads to use per block
					int max_block_size, // maximum block size for the charge sum calculation
					Scalar *d_output, // pointer to output array

					Scalar xi, // Ewald splitting parameter
					Scalar3 eta, // spectral splitting parameter
					Scalar rc, // real space cutoff radius
					int Nx, // number of grid nodes in the x dimension
					int Ny, // number of grid nodes in the y dimension
					int Nz, // number of grid nodes in the z dimension
					Scalar3 gridh, // grid spacing
					int P, // number of grid nodes over which to spread and contract

					CUFFTCOMPLEX *d_qgrid, // wave space charge grid
					Scalar *d_scale_phiq, // potential/dipole scaling on grid
					cufftHandle plan, // plan for the FFTs

					int N_table, // number of entries in the real space coefficient table
					Scalar drtable, // real space coefficient table spacing
					Scalar2 *d_phiq_table, // pointer to potential/charge real space table

					const unsigned int *d_nlist, // neighbor list
					const unsigned int *d_head_list, // used to access entries in the neighbor list
					const unsigned int *d_n_neigh) // number of neighbors of each particle
{
	// Extract quantities from the input
	Scalar *d_bead_charge = d_input; // bead charges (including isolated beads)
	Scalar *d_isolated_potential = &d_input[group_size]; // isolated body potentials

	// Extract quantities from the output
	Scalar *d_phiq = d_output; // result of top row of the matrix/vector multiply
	Scalar *d_isolated_charge = &d_output[group_size]; // isolated body charges

	// Compute the top row of the matrix/vector multiply
	PotentialChargeMultiply(d_bead_pos, d_bead_charge, d_isolated_potential, d_group_membership, d_group_members, d_isolated_membership,
				box, group_size, block_size, d_phiq, xi, eta, rc, Nx, Ny, Nz, gridh, P, d_qgrid, d_scale_phiq, plan, N_table,
				drtable, d_phiq_table, d_nlist, d_head_list, d_n_neigh);
	gpuErrchk(cudaPeekAtLastError());
	// Sum the charges on a single rigid body.  Use one block per rigid body and the smaller of (Nmax, max_block_size) threads per block.
	int N_block = N_rigid;
	int N_thread = (max_rigid_size < max_block_size) ? max_rigid_size : max_block_size;
	sum_charges<<<N_block, N_thread, N_thread*sizeof(float)>>>(d_bead_charge, d_isolated_charge, d_isolated_membership, d_rigid_members,
								   d_rigid_size, max_rigid_size);
	gpuErrchk(cudaPeekAtLastError());
	// Set the isolated body charges. Use one thread per particle in the active group.
	N_block = (group_size/block_size) + 1;
	N_thread = block_size;
	set_charge<<<N_block, N_thread>>>(d_bead_charge, d_isolated_charge, d_group_members, d_isolated_membership, d_rigid_membership,
					  group_size);

    	gpuErrchk(cudaPeekAtLastError());
    	return cudaSuccess;
}

// Precondition the matrix/vector equation
cudaError_t Precondition(	Scalar *d_input,  // input to preconditioner (bead charges and isolated potentials)
				Scalar *d_output,  // output of preconditioner

				unsigned int *d_isolated_membership, // isolated body tag each particle belongs to (from plugin)
				unsigned int *d_rigid_members, // indices of beads belonging to each rigid body
				unsigned int *d_rigid_size, // number of beads in each rigid body

				int group_size, // number of particles in active group
				int N_rigid, // number of rigid bodies
				int N_isolated,  // number of isolated bodies
				int max_rigid_size, // maximum rigid body size

				int block_size) // number of threads to use per block
{
	// Use one thread per input/output entry
	int N_block = (group_size + N_isolated)/block_size + 1;
	precondition<<<N_block, block_size>>>(d_input, d_output, d_isolated_membership, d_rigid_members, d_rigid_size, group_size, N_rigid,
					      N_isolated, max_rigid_size);
    	
	gpuErrchk(cudaPeekAtLastError());
    	return cudaSuccess;
}

// Solve the matrix/vector equation iteratively using GMRES.
//
//  [ -M_phiq   Sigma^T ]   [      q      ]   [ -r*E_0 ]
//  [  Sigma       0    ] * [ Phi - Phi_0 ] = [    Q   ]
//
cudaError_t MatrixVectorSolve(	Scalar4 *d_bead_pos, // particle posisitons
				Scalar4 *d_rigid_pos, // rigid body centers of mass
				Scalar *d_bead_charge, // bead charges
				Scalar *d_isolated_charge, // known charge on each isolated body
				Scalar *d_isolated_potential, // isolated body potentials

				unsigned int *d_group_membership, // particle index in active group
				unsigned int *d_group_members, // particles in active group
				unsigned int *d_isolated_membership, // isolated body tag each particle belongs to
				unsigned int *d_rigid_membership, // rigid body index each particle belongs to; isolated beads get NOT_MEMBERS
				unsigned int *d_rigid_members, // indices of beads belonging to each rigid body
				unsigned int *d_rigid_size, // number of beads in each rigid body

				const BoxDim& box, // simulation box
				Scalar3 extfield, // external field
				int group_size, // number of particles in active group
				int RigidGroup_size, // number of particles in the HOOMD RigidGroup
				int N_rigid, // number of rigid bodies
				int N_isolated, // number of isolated bodies
				int max_rigid_size, // maximum rigid body size

				int block_size, // number of threads to use per block
				int max_block_size, // maximum block size for the charge sum calculation
				Scalar *d_objective, // pointer to objective output array

				Scalar xi, // Ewald splitting parameter
				Scalar errortol, // error tolerance
				Scalar3 eta, // spectral splitting parameter
				Scalar rc, // real space cutoff radius
				int Nx, // number of grid nodes in the x dimension
				int Ny, // number of grid nodes in the y dimension
				int Nz, // number of grid nodes in the z dimension
				Scalar3 gridh, // grid spacing
				int P, // number of grid nodes over which to spread and contract

				CUFFTCOMPLEX *d_qgrid, // wave space charge grid
				Scalar *d_scale_phiq, // potential/dipole scaling on grid
				cufftHandle plan, // plan for the FFTs

				int N_table, // number of entries in the real space coefficient table
				Scalar drtable, // real space coefficient table spacing
				Scalar2 *d_phiq_table, // pointer to potential/charge real space table

				const unsigned int *d_nlist, // neighbor list
				const unsigned int *d_head_list, // used to access entries in the neighbor list
				const unsigned int *d_n_neigh) // number of neighbors of each particle
{

	// Zero the objective output array
	int N_blocks = (group_size + N_isolated)/block_size + 1;
	int N_threads = block_size;
	zero_array<<<N_blocks, N_threads>>>(d_objective, group_size + N_isolated);

	// Compute objective output of the matrix/vector multiply. Use one thread per entry in the d_rigid_members (N_rigid*max_rigid_size) 		// arrays plus one thread per isolated body.
	N_blocks = (N_rigid*max_rigid_size + N_isolated)/block_size + 1;
	compute_objective<<<N_blocks, N_threads>>>(d_bead_pos, d_rigid_pos, d_isolated_charge, d_group_membership, d_rigid_members, box,
						   extfield, group_size, N_isolated, max_rigid_size, N_rigid*max_rigid_size, d_objective);

	// Create the preconditioner
	cuspPrecondition Pr(d_isolated_membership, d_rigid_members, d_rigid_size, group_size, N_rigid, N_isolated, max_rigid_size, block_size);

	// Create the matrix-free potential linear operator
	cuspPotential M(d_bead_pos, d_group_membership, d_group_members, d_isolated_membership, d_rigid_membership, d_rigid_members,
			d_rigid_size, box, group_size, N_rigid, N_isolated, max_rigid_size, block_size, max_block_size, xi, eta, rc, Nx, Ny,
			Nz, gridh, P, d_qgrid, d_scale_phiq, plan, N_table, drtable, d_phiq_table, d_nlist, d_head_list, d_n_neigh);

	// Allocate storage for the solution (q) and right side (rhs) on the GPU
	cusp::array1d<float, cusp::device_memory> q(M.num_rows, 0);
	cusp::array1d<float, cusp::device_memory> rhs(M.num_rows, 0);

	// Get pointers to the cusp arrays
	float *d_q = thrust::raw_pointer_cast(&q[0]);
	float *d_rhs = thrust::raw_pointer_cast(&rhs[0]);

	// Use the solution from the previous time step as the initial guess
	cudaMemcpy(d_q, d_bead_charge, group_size*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(&d_q[group_size], d_isolated_potential, N_isolated*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_rhs, d_objective, (group_size + N_isolated)*sizeof(float), cudaMemcpyDeviceToDevice);

	// Set the preconditioner (identity for now)
	// cusp::identity_operator<float, cusp::device_memory> Pr(M.num_rows,M.num_rows);

	// Solve the linear system using GMRES
	cusp::default_monitor<float> monitor(rhs, 100, errortol);
	int restart = 10;
	cusp::krylov::gmres(M, q, rhs, restart, monitor, Pr);

	// Store the computed charges and potentials to the correct place in device memory
	cudaMemcpy(d_bead_charge, d_q, group_size*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_isolated_potential, &d_q[group_size], N_isolated*sizeof(float), cudaMemcpyDeviceToDevice);

	// Print iteration number
	//if (monitor.converged()) {
        //    std::cout << "Solver converged after " << monitor.iteration_count() << " iterations." << std::endl;
        //} else {
        //    std::cout << "Solver reached iteration limit " << monitor.iteration_limit() << " before converging." << std::endl;
        //}

	gpuErrchk(cudaPeekAtLastError());
    	return cudaSuccess;
}

cudaError_t gpu_ComputeForce(   Scalar4 *d_bead_pos, // bead posisitons
				Scalar4 *d_rigid_pos, // rigid body centers of mass
				Scalar *d_bead_charge, // bead charges
				Scalar4 *d_bead_force, // bead forces
				Scalar *d_isolated_charge, // known charge on each isolated body
				Scalar *d_isolated_potential, // isolated body potentials

				unsigned int *d_bead_tag, // global particle tags
				unsigned int *d_group_membership, // particle index in active group
				unsigned int *d_group_members, // particles in active group
				unsigned int *d_isolated_membership_unsrt, // isolated body tag each particle belongs to (from user, unsorted)
				unsigned int *d_isolated_membership, // isolated body tag each particle belongs to (sorted)
				unsigned int *d_rigid_membership, // rigid body index each particle belongs to; isolated beads get NOT_MEMBER
				unsigned int *d_rigid_members, // indices of beads belonging to each rigid body
				unsigned int *d_rigid_size, // number of beads in each rigid body

				const BoxDim& box, // simulation box
				Scalar3 extfield, // external field
				int N_total, // total number of particles
				int group_size, // number of particles in active group
				int RigidGroup_size, // number of particles in the HOOMD RigidGroup
				int N_rigid, // number of rigid bodies
				int N_isolated, // number of isolated bodies
				int max_rigid_size, // maximum rigid body size

				int block_size, // number of threads to use per block
				int max_block_size, // maximum block size for the charge sum calculation
				Scalar *d_objective, // pointer to objective output array

				Scalar xi, // Ewald splitting parameter
				Scalar errortol, // error tolerance
				Scalar3 eta, // spectral splitting parameter
				Scalar rc, // real space cutoff radius
				int Nx, // number of grid nodes in the x dimension
				int Ny, // number of grid nodes in the y dimension
				int Nz, // number of grid nodes in the z dimension
				Scalar3 gridh, // grid spacing
				int P, // number of grid nodes over which to spread and contract

				CUFFTCOMPLEX *d_qgrid, // wave space charge grid
				Scalar *d_scale_phiq, // potential/dipole scaling on grid
				cufftHandle plan, // plan for the FFTs

				int N_table, // number of entries in the real space coefficient table
				Scalar drtable, // real space coefficient table spacing
				Scalar2 *d_phiq_table, // potential/charge real space table
				Scalar2 *d_gradphiq_table, // potential/charge gradient real space table

				const unsigned int *d_nlist, // neighbor list
				const unsigned int *d_head_list, // used to access entries in the neighbor list
				const unsigned int *d_n_neigh) // number of neighbors of each particle
{

	// total number of grid nodes
 	int N_grid = Nx*Ny*Nz;

	// for grid initialization and scaling, use one thread per grid node
    	int Nthreads1 = ( N_grid > block_size ) ? block_size : N_grid;
    	int Nblocks1 = ( N_grid - 1 )/Nthreads1 + 1;

	// For spreading and contracting, use one P-by-P-by-P block per group particle.
	dim3 Nblocks2(group_size, 1, 1); // grid is a 1-D array
	dim3 Nthreads2(P, P, P); // block is a 3-D array

    	// for updating group membership and the real space calculation, use one thread per group particle
    	dim3 Nblocks3( (group_size/block_size) + 1, 1, 1);
    	dim3 Nthreads3(block_size, 1, 1);

	// for initializing group membership, use one thread per total particle
	dim3 Nblocks4( (N_total/block_size) + 1, 1, 1 );
	dim3 Nthreads4(block_size, 1, 1);

	// Factors needed for the kernels
	Scalar quadW = gridh.x*gridh.y*gridh.z; // trapezoidal rule weights
	Scalar xiterm = 2.0*xi*xi;
	Scalar prefac = xiterm*xi/PI*sqrtf(2.0/(PI*eta.x*eta.y*eta.z));  // prefactor for the spreading and contracting exponentials

	// Handle the real space tables and positions as textured memory
    	phiq_table_tex.normalized = false; // Not normalized
    	phiq_table_tex.filterMode = cudaFilterModePoint; // Filter mode: floor of the index
    	// One dimension, Read mode: ElementType(Get what we write)
    	cudaBindTexture(0, phiq_table_tex, d_phiq_table, sizeof(Scalar2) * (N_table+1));

    	gradphiq_table_tex.normalized = false;
    	gradphiq_table_tex.filterMode = cudaFilterModePoint;
    	cudaBindTexture(0, gradphiq_table_tex, d_gradphiq_table, sizeof(Scalar2) * (N_table+1));

    	pos_tex.normalized = false;
    	pos_tex.filterMode = cudaFilterModePoint;
    	cudaBindTexture(0, pos_tex, d_bead_pos, sizeof(Scalar4) * N_total);

	// Update the group membership lists
	initialize_membership<<<Nblocks4, Nthreads4>>>(d_group_membership, d_isolated_membership, N_total); // thread per total particle
	update_membership<<<Nblocks3, Nthreads3>>>(d_group_membership, d_group_members, group_size, d_bead_tag, d_isolated_membership_unsrt,
						 d_isolated_membership); // thread per active particle

	// Compute the particle charged and isolated body potentials.
	MatrixVectorSolve(d_bead_pos, d_rigid_pos, d_bead_charge, d_isolated_charge, d_isolated_potential, d_group_membership,
			  d_group_members, d_isolated_membership, d_rigid_membership, d_rigid_members, d_rigid_size, box, extfield,
			  group_size, RigidGroup_size, N_rigid, N_isolated, max_rigid_size, block_size, max_block_size, d_objective, xi,
			  errortol, eta, rc, Nx, Ny, Nz, gridh, P, d_qgrid, d_scale_phiq, plan, N_table, drtable, d_phiq_table, d_nlist,
			  d_head_list, d_n_neigh);

    	// Reset the grid values to zero
	initialize_grid<<<Nblocks1, Nthreads1>>>(d_qgrid, N_grid);

	// Spread dipoles from the particles to the grid
	spread<<<Nblocks2, Nthreads2>>>(d_bead_pos, d_bead_charge, d_group_members, box, group_size, eta, Nx, Ny, Nz, gridh, P, d_qgrid,
					xiterm, prefac);

	//  Compute the Fourier transform of the gridded data
    	cufftExecC2C(plan, d_qgrid, d_qgrid, CUFFT_FORWARD);

	// Scale the grid values
    	scale<<<Nblocks1, Nthreads1>>>(d_qgrid, d_scale_phiq, N_grid);

	// Inverse Fourier transform the gridded data
    	cufftExecC2C(plan, d_qgrid, d_qgrid, CUFFT_INVERSE);

	// Contract the gridded values to the particles
	contract_force<<<Nblocks2, Nthreads2, 3*(P*P*P+1)*sizeof(float)>>>(d_bead_pos, d_bead_charge, d_bead_force, d_group_members, box,
							group_size, eta, Nx, Ny, Nz, gridh, P, d_qgrid, xiterm, quadW*prefac);   

	// Compute the real space contribution
    	real_space_force<<<Nblocks3, Nthreads3>>>(d_bead_pos, d_bead_charge, d_bead_force, d_group_membership, d_group_members, box, extfield,
						  group_size, rc, N_table, drtable, d_gradphiq_table, d_nlist, d_head_list, d_n_neigh); 

	cudaUnbindTexture(phiq_table_tex);
	cudaUnbindTexture(gradphiq_table_tex);
    	cudaUnbindTexture(pos_tex);

	gpuErrchk(cudaPeekAtLastError());
    	return cudaSuccess;
}

