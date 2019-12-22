#include <iostream>
#include <string>
#include <cmath>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <CL/cl.hpp>
#include <CL/cl.h>

using namespace cl;
using namespace std;

float * createBlurMask(float sigma, int * maskSizePointer) {
    int maskSize = (int)ceil(3.0f*sigma);
    float * mask = new float[(maskSize*2+1)*(maskSize*2+1)];
    float sum = 0.0f;
    for(int a = -maskSize; a < maskSize+1; a++) {
        for(int b = -maskSize; b < maskSize+1; b++) {
            float temp = exp(-((float)(a*a+b*b) / (2*sigma*sigma)));
            sum += temp;
            mask[a+maskSize+(b+maskSize)*(maskSize*2+1)] = temp;
        }
    }
    // Normalize the mask
    for(int i = 0; i < (maskSize*2+1)*(maskSize*2+1); i++)
        mask[i] = mask[i] / sum;

    *maskSizePointer = maskSize;

    return mask;
}

cl::Program buildProgramFromSource(cl::Context context, std::string filename, std::string buildOptions="") {
    // Read source file
    std::ifstream sourceFile(filename.c_str());
    // if(sourceFile.fail())
    // throw Error(1, "Failed to open OpenCL source file");
    std::string sourceCode(
            std::istreambuf_iterator<char>(sourceFile),
            (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

    // Make program of the source code in the context
    cl::Program program = cl::Program(context, source);

    VECTOR_CLASS<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    // Build program for these specific devices
    // try{
    program.build(devices, buildOptions.c_str());
    // } catch(cl::Error error) {
    // if(error.err() == CL_BUILD_PROGRAM_FAILURE) {
    // std::cout << "Build log:" << std::endl << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
    // }
    // throw error;
    // }
    return program;

}

int main(int argc, char ** argv) {
    // Load image
    cv::Mat image = cv::imread("images/sunset.jpg");
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    // cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image.convertTo(image, CV_32F);

    // get all platforms (drivers), e.g. NVIDIA
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size()==0) {
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[0];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

    // get default device (CPUs, GPUs) of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }

    cl::Device default_device=all_devices[0];
    std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";

    // Create OpenCL context
    cl::Context context({default_device});

    // Compile OpenCL code
    Program program = buildProgramFromSource(context, "gaussian_blur.cl");

    // Select device and create a command queue for it
    // VECTOR_CLASS<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    CommandQueue queue = CommandQueue(context, default_device);

    // Create an OpenCL Image / texture and transfer data to the device
    Image2D clImage = Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, \
            ImageFormat(CL_R, CL_FLOAT), image.cols, image.rows, 0, (void*)((float*)image.data));

    // Create a buffer for the result
    Buffer clResult = Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*image.cols*image.rows);

    // Create Gaussian mask
    int maskSize;
    float * mask = createBlurMask(10.0f, &maskSize);

    // Create buffer for mask and transfer it to the device
    Buffer clMask = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*(maskSize*2+1)*(maskSize*2+1), mask);

    // Run Gaussian kernel
    Kernel gaussianBlur = Kernel(program, "gaussian_blur");
    gaussianBlur.setArg(0, clImage);
    gaussianBlur.setArg(1, clMask);
    gaussianBlur.setArg(2, clResult);
    gaussianBlur.setArg(3, maskSize);

    queue.enqueueNDRangeKernel(
            gaussianBlur,
            NullRange,
            NDRange(image.cols, image.rows),
            NullRange
            );

    // Transfer image back to host
    float* data = new float[image.cols*image.rows];
    queue.enqueueReadBuffer(clResult, CL_TRUE, 0, sizeof(float)*image.cols*image.rows, data);
    cv::Mat saved_image(image.rows, image.cols, CV_32F, data);
    saved_image.convertTo(saved_image, CV_8U);
    cv::imwrite("output.jpg", saved_image);
}
