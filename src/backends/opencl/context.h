#ifndef OPENCL_CONTEXT_H_
#define OPENCL_CONTEXT_H_
#include <iostream>
#include <map>
#include <string>
#include <memory>
#include <fstream>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

class Context{
    public:
        Context();
        virtual ~Context(){}

        bool BuildProgram(const std::string& kProgramName, const std::string& kBuildOptions="");

        cl::Kernel BuildKernel(const std::string& kProgramName, const std::string kKernelName);

        const cl::Context& context()const {
            return *mContext;
        }
        const cl::Device& device()const {
            return *mFirstGPUDevicePtr;
        }

        const cl::CommandQueue& command_queue()const {
            return *mCommandQueuePtr;
        }

    private:
        std::map<std::string, cl::Program> mBuildProgramMap;
        std::shared_ptr<::cl::Context> mContext;
        std::shared_ptr<::cl::Device> mFirstGPUDevicePtr;
        std::shared_ptr<::cl::CommandQueue> mCommandQueuePtr;
};

#endif
