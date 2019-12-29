#include "context.h"


Context::Context(){
    // platforms
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size()==0) {
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[0];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

    // device
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device default_device=all_devices[0];
    std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";

    // context
    mContext.reset(new cl::Context({default_device}));

    // queue
    mCommandQueuePtr.reset(new cl::CommandQueue(*mContext, default_device));
}



bool Context::BuildProgram(const std::string& kProgramName, const std::string& kBuildOptions){
    std::ifstream sourceFile(kProgramName.c_str());
    if(sourceFile.fail())
        // throw cl::Error(1, "Failed to open OpenCL source file");
        return false;
    std::string sourceCode(
            std::istreambuf_iterator<char>(sourceFile),
            (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

    // Make program of the source code in the context
    cl::Program program = cl::Program(*mContext, source);

    VECTOR_CLASS<cl::Device> devices = mContext->getInfo<CL_CONTEXT_DEVICES>();

    // Build program for these specific devices
    try{
        program.build(devices, kBuildOptions.c_str());
    } catch(cl::Error error) {
        if(error.err() == CL_BUILD_PROGRAM_FAILURE) {
            std::cout << "Build log:" << std::endl << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
        }
        return false;
    }
    mBuildProgramMap.emplace(kProgramName, program);
    return true;
}

cl::Kernel Context::BuildKernel(const std::string& kProgramName, const std::string kKernelName){
    auto map_iter = mBuildProgramMap.find(kProgramName);
    if(map_iter==mBuildProgramMap.end()){
        BuildProgram(kProgramName);
    }
    map_iter = mBuildProgramMap.find(kProgramName);
    auto program = map_iter->second;
    cl::Kernel kernel = cl::Kernel(program, kKernelName.c_str());
    return kernel;
}
