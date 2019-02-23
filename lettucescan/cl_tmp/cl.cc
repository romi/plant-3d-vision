#include "cl.h"

#include <iostream>

namespace romi {

cl::Context context;
cl::CommandQueue queue;

std::vector<cl::Device> devices;

void init_opencl(int platform_id, int device_id) {
    std::vector<cl::Platform> platforms;
    // std::vector<cl::Device> devices;
    // Init OpenCL
    try {
        cl::Platform::get(&platforms);
        platforms[platform_id].getDevices(CL_DEVICE_TYPE_GPU |
                                              CL_DEVICE_TYPE_CPU,
                                          &devices); // Select the platform.
        context = cl::Context(devices);
        queue = cl::CommandQueue(context, devices[device_id]);
    } catch (cl::Error err) {
        std::cout << "Error initizalizing OpenCL devices: " << err.what() << "("
                  << err.err() << ")" << std::endl;
        throw err;
    } // catch
}

void build_kernel(std::string source_str, const char* name, cl::Kernel& kernel) {
    cl::Program::Sources source(1,
                                std::make_pair(source_str.c_str(),
                                               source_str.length()));
    cl::Program program(context, source);
    try {
        program.build(devices);
        kernel = cl::Kernel(program, name);
    } catch (cl::Error err) {
        if (err.err() == CL_BUILD_PROGRAM_FAILURE) {
            for (cl::Device dev : devices) {
                // Check the build status
                cl_build_status status =
                    program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
                if (status != CL_BUILD_ERROR)
                    continue;

                // Get the build log
                std::string name = dev.getInfo<CL_DEVICE_NAME>();
                std::string buildlog =
                    program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
                std::cerr << "Build log for " << name << ":" << std::endl
                          << buildlog << std::endl;
            }
        }
        std::cout << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
        throw err;
    } // catch
}
} // namespace romi
