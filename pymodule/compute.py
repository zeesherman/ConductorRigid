# This simple python interface activates the c++ ConductorRigid from cppmodule

import math
import numpy as np

# Import the C++ module.
import _ConductorRigid

# Because we are extending a ForceCompute, we need to bring in the base class force and some other parts from hoomd_script
from hoomd_script.force import _force
from hoomd_script import util
from hoomd_script import globals
import hoomd

# Computes the forces on charged rigid body conductors in an external field.
class ConductorRigid(_force):

    # Initialize the Dielectric force
    def __init__(self, group, body, charge, field = [0.,0.,0.], xi = 0.5, errortol = 1e-3, fileprefix = "", period = 0):

        util.print_status_line();
    
        # initialize base class
        _force.__init__(self);

	# set the cutoff radius
	self.rcut = math.sqrt(-math.log(errortol))/xi;      

	# enable the force
	self.enabled = True;

        # initialize the reflected C++ class
        if not globals.exec_conf.isCUDAEnabled():
	    globals.msg.error("Sorry, we have not written CPU code for rigid conductor calculations. \n");
            raise RuntimeError('Error creating ConductorRigid');
        else:
	    # Create a new neighbor list
	    cl_ConductorRigid = hoomd.CellListGPU(globals.system_definition);
	    globals.system.addCompute(cl_ConductorRigid, "ConductorRigid_cl");
	    self.neighbor_list = hoomd.NeighborListGPUBinned(globals.system_definition, self.rcut, 0.4, cl_ConductorRigid);
	    self.neighbor_list.setEvery(1, True); 
	    globals.system.addCompute(self.neighbor_list, "ConductorRigid_nlist");
	    self.neighbor_list.clearExclusions();
	    self.neighbor_list.countExclusions();

	    # Add the dipole force to the system
            self.cpp_force = _ConductorRigid.ConductorRigid(globals.system_definition, group.cpp_group, self.neighbor_list, body, charge, field, xi, errortol, fileprefix, period, globals.system.getCurrentTimeStep());
	    globals.system.addCompute(self.cpp_force,self.force_name);

	# Set parameters for the dipole force
	self.cpp_force.SetParams();

    # Update only the external field.  Faster than the update_parameters function.
    def update_field(self, field):
	self.cpp_force.UpdateField(field);

    # Update simulation parameters.
    def update_parameters(self, field, fileprefix, period):
	self.cpp_force.UpdateParams(field, fileprefix, period, globals.system.getCurrentTimeStep());
	self.cpp_force.SetParams();

    # The integrator calls the update_coeffs function but there are no coefficients to update, so this function does nothing
    def update_coeffs(self):
	pass



