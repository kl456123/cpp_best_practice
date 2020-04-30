#include <stdio.h>
#include <memory>
#include <iostream>
#include <random>
#include <assert.h>
#include <cstring>
#include "opengl/core/init.h"

#include "opengl/core/program.h"
#include "opengl/core/buffer.h"
#include "opengl/core/context.h"
#include "opengl/nn/kernels/binary.h"

// only tensor is visible in nn module
using opengl::Tensor;
using opengl::TensorList;
using opengl::Context;



int main(int argc, char** argv){
    // glut_init(argc, argv);
    ::opengl::glfw_init();
    ::opengl::glew_init();

    int maxtexsize;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE,&maxtexsize);
    printf("GL_MAX_TEXTURE_SIZE, %d\n",maxtexsize);

    // prepare runtime
    auto context = std::unique_ptr<Context>(new Context(nullptr));

    //prepare input and output
    float input1[] = {1.0, 2.0, 3.0};
    float input2[] = {1.0, 2.0, 3.0};
    TensorList inputs_cpu;
    TensorList outputs_gpu;
    inputs_cpu.emplace_back(new Tensor(input1, Tensor::DT_FLOAT, {3}));
    inputs_cpu.emplace_back(new Tensor(input2, Tensor::DT_FLOAT, {3}));

    TensorList inputs_gpu;
    for(int i=0;i<inputs_cpu.size();i++){
        inputs_gpu.emplace_back(new Tensor(Tensor::DT_FLOAT, {3}));
    }

    outputs_gpu.emplace_back(new Tensor(Tensor::DT_FLOAT, {3}));

    // upload to gpu
    for(unsigned int i=0;i<inputs_cpu.size();++i){
        context->CopyCPUTensorToDevice(inputs_cpu[i], inputs_gpu[i]);
    }

    auto binary_kernel = ::opengl::BinaryKernel(context.get());
    binary_kernel.Compute(inputs_gpu, outputs_gpu);

    context->Finish();

    TensorList outputs_cpu;
    // download to cpu
    for(unsigned int i=0;i<outputs_gpu.size();++i){
        auto output_cpu_tensor = new Tensor(Tensor::DT_FLOAT, {3});
        context->CopyDeviceTensorToCPU(output_cpu_tensor, outputs_gpu[i]);
        outputs_cpu.emplace_back(output_cpu_tensor);
    }

    // auto output_cpu_tensor = new Tensor(Tensor::DT_FLOAT, {3});
    // context->CopyDeviceTensorToCPU(output_cpu_tensor, inputs_gpu[0]);
    // outputs_cpu.emplace_back(output_cpu_tensor);

    // clean up
    for(unsigned int i=0;i<outputs_cpu.size();++i){
        // clean up outputs
        for(int j=0;j<outputs_cpu[i]->num_elements();j++){
            std::cout<<outputs_cpu[i]->host<float>()[j]<<std::endl;
        }
    }
    for(unsigned int i=0;i<inputs_gpu.size();++i){
        // clean up outputs
    }

    return 0;
}
