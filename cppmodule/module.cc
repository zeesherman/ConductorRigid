// Import ConductorRigid to python
#include "ConductorRigid.h"

// Include boost.python to do the exporting
#include <boost/python.hpp>
using namespace boost::python;

// Specify the python module.
BOOST_PYTHON_MODULE(_ConductorRigid)
    {
    #ifdef ENABLE_CUDA
    export_ConductorRigid();
    #endif
    }

