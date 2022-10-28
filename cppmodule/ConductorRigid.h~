#include "hoomd/hoomd_config.h"
#include "ForceCompute.h"
#include <cufft.h>
#include "NeighborList.h"

#ifndef SINGLE_PRECISION
#define CUFFTCOMPLEX cufftComplex
#else
#define CUFFTCOMPLEX cufftComplex
#endif

#ifndef __ConductorRigid_H__
#define __ConductorRigid_H__

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

// ConductorRigid.h: Declares the ConductorRigid class, which calculates forces on charged conductors of arbitrary shape in an external electric field.

class ConductorRigid : public ForceCompute {

    public:
        // Constructs the compute and associates it with the system
        ConductorRigid(boost::shared_ptr<SystemDefinition> sysdef,
                       boost::shared_ptr<ParticleGroup> group,
	               boost::shared_ptr<NeighborList> nlist,
		       boost::python::list body,
		       boost::python::list charge,
		       boost::python::list field,
	               Scalar xi,
		       Scalar errortol,
		       std::string fileprefix,
		       int period,
		       int t0);

	// Destructor
        virtual ~ConductorRigid();

	// Set parameters needed for the force calculation
	void SetParams();

	// Update field during a simulation without recomputing parameters on the CPU
	void UpdateField(boost::python::list field);

	// Update parameters during a simulation
	void UpdateParams(boost::python::list field,
			  std::string fileprefix,
			  int period,
			  int t0);

	// Write bead charges and forces to file
	void OutputData(unsigned int timestep);

	// Performs the force calculation
        virtual void computeForces(unsigned int timestep);

    protected:

	boost::shared_ptr<ParticleGroup> m_group;	// active group of particles for which to run the electrostatic calculations
	boost::shared_ptr<RigidData> m_rigid_data;	// rigid body data
	GPUArray<Scalar4> m_bead_pos;			// bead positions and types
	GPUArray<Scalar4> m_rigid_pos;			// rigid body centers of mass
	GPUArray<Scalar> m_bead_charge;			// bead charges
	GPUArray<Scalar4> m_bead_force;			// bead forces
	GPUArray<Scalar> m_isolated_charge;		// known charge on each isolated body
	GPUArray<Scalar> m_isolated_potential;		// isolated body potentials

	GPUArray<unsigned int> m_group_membership;	// each particle's index in the active group; inactive particles get NOT_MEMBER
	GPUArray<unsigned int> m_group_members;		// indices of particles in the active group
	GPUArray<unsigned int> m_isolated_membership_unsrt; // isolated body tag to which beads (by tag) belong; inactive beads get NOT_MEMBER
	GPUArray<unsigned int> m_isolated_membership;	// isolated tag to which beads (by global index) belong; inactive beads get NOT_MEMBER
	GPUArray<unsigned int> m_rigid_membership;	// rigid index to which beads (by global index) belongs; isolated beads get NOT_MEMBER
	GPUArray<unsigned int> m_rigid_members;		// indices of the beads belonging to each rigid body
	GPUArray<unsigned int> m_rigid_size;		// number of beads in each rigid body

	Scalar3 m_extfield;				// externally applied field
	int m_N_total;					// total number of particles
	int m_group_size;				// number of particles in the active group
	int m_RigidGroup_size;				// number of particles in the HOOMD RigidGroup
	int m_N_rigid;					// number of rigid bodies
	int m_N_isolated;				// number of isolated bodies (rigid bodies + isolated beads)

	GPUArray<Scalar> m_objective;			// pointer to known objective output to the matrix/vector multiply

	Scalar m_xi;					// Ewald splitting parameter
	Scalar m_errortol;				// numerical error tolerance
	Scalar3 m_eta;					// spectral splitting parameter
	Scalar m_rc;					// real space cutoff radius
	int m_Nx;					// number of grid nodes in the x dimension
	int m_Ny;					// number of grid nodes in the y dimension
	int m_Nz;					// number of grid nodes in the z dimension
	Scalar3 m_gridh;				// grid spacing
	int m_P;					// number of grid nodes over which to spread and contract

	GPUArray<Scalar3> m_gridk;			// wave vector on grid
	GPUArray<CUFFTCOMPLEX> m_qgrid;      		// potential/charge grid
	GPUArray<Scalar> m_scale_phiq;      		// potential/charge scaling on grid		
	cufftHandle m_plan;                    		// used for the fast Fourier transformations performed on the GPU

	int m_N_table;					// number of entries in the real space tables
	Scalar m_drtable;				// real space table spacing
	GPUArray<Scalar2> m_phiq_table;			// potential/charge real space table
	GPUArray<Scalar2> m_gradphiq_table;		// potential/charge gradient real space table

	boost::shared_ptr<NeighborList> m_nlist;	// neighbor list

	std::string m_fileprefix;			// output file prefix
	int m_period;					// frequency with which to write output files
	int m_t0;					// initial timestep

    };

// Exports the ConductorRigid class to python
void export_ConductorRigid();

#endif
