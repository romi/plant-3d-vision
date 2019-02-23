#ifndef cl_h_INCLUDED
#define cl_h_INCLUDED

#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.cpp>
#else
#include <CL/cl.hpp>
#endif

namespace romi {

void init_opencl(int platform_id, int device_id);
void build_kernel(std::string source, const char* name, cl::Kernel& kernel);

extern cl::Context context;
extern cl::CommandQueue queue;

}

#endif // cl_h_INCLUDED

