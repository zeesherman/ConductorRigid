// ConductorRigid.cuh: Declares GPU kernel code for computing the electrostatic forces.

#include "hoomd/hoomd_config.h"
#include "ParticleData.cuh"
#include "HOOMDMath.h"
#include <cufft.h>
#include "Index1D.h"

#ifndef __ConductorRigid_CUH__
#define __ConductorRigid_CUH__

#ifdef SINGLE_PRECISION
#define CUFFTCOMPLEX cufftComplex
#else
#define CUFFTCOMPLEX cufftComplex
#endif

// Kernel driver for the calculations called by ConductorRigid.cc
cudaError_t gpu_ZeroForce(int N_total, // total number of particles
			  Scalar4 *d_force, // pointer to the particle forces 
			  int block_size); // number of threads per block

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
				const unsigned int *d_n_neigh); // number of neighbors of each particle

// Kernel called by PotentialWrapper.cuh
cudaError_t MatrixVectorMultiply(   	Scalar4 *d_pos, // particle posisitons
					Scalar *d_input, // particle charges and isolated body potentials

					unsigned int *d_group_membership, // particle index in active group
					unsigned int *d_group_members, // particles in active group
					unsigned int *d_isolated_membership, // isolated body tag each particle belongs to (from plugin)
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
					const unsigned int *d_n_neigh); // number of neighbors of each particle

// Kernel called by PreconditionWraper.cuh
cudaError_t Precondition(	Scalar *d_input,  // input to preconditioner (bead charges and isolated potentials)
				Scalar *d_output,  // output of preconditioner

				unsigned int *d_isolated_membership, // isolated body tag each particle belongs to (from plugin)
				unsigned int *d_rigid_members, // indices of beads belonging to each rigid body
				unsigned int *d_rigid_size, // number of beads in each rigid body

				int group_size, // number of particles in active group
				int N_rigid, // number of rigid bodies
				int N_isolated,  // number of isolated bodies
				int max_rigid_size, // maximum rigid body size

				int block_size); // number of threads to use per block


#endif
