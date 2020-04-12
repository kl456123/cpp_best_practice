#include <stdio.h>
#include <memory>
#include <iostream>
#include <random>
#include <assert.h>
#include <cstring>
#include "init.h"

#include "program.h"
#include "glut.h"
#include "buffer.h"
#include "context.h"
#include "kernels/binary.h"




int main(int argc, char** argv){
    glut_init(argc, argv);
    glew_init();

    int maxtexsize;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE,&maxtexsize);
    printf("GL_MAX_TEXTURE_SIZE, %d\n",maxtexsize);

    {
        // buffer read and write test
        const int num = 1<<3;
        const int size = num*sizeof(float);
        auto buffer_device = ShaderBuffer(size);
        float buffer_cpu1[num]={0};
        float buffer_cpu2[num]={0};
        for(int i=0;i<num;i++){
            buffer_cpu1[i] = random()%100;
        }

        //write first from cpu1
        ::memcpy(buffer_device.Map(GL_MAP_WRITE_BIT), buffer_cpu1, size);
        buffer_device.UnMap();

        // read then to cpu2
        ::memcpy(buffer_cpu2, buffer_device.Map(GL_MAP_WRITE_BIT), size);
        buffer_device.UnMap();

        // check the same between cpu1 and cpu2
        for(int i=0;i<num;++i){
            assert(buffer_cpu1[i]==buffer_cpu2[i]);
        }
    }

    // prepare program
    // const char source[] = "";
    // const std::string fname = "../examples/glsl/binary.glsl";

    // Program program;
    // program.Attach(fname)
    // .Link();

    // program.Activate();

    // prepare runtime
    auto context = std::unique_ptr<Context>(new Context(nullptr));

    //prepare input and output
    float input1[] = {1.0, 2.0, 3.0};
    float input2[] = {1.0, 2.0, 3.0};
    TensorList inputs_cpu;
    TensorList outputs_gpu;
    inputs_cpu.emplace_back(new Tensor(input1, Tensor::DT_FLOAT, 3));
    inputs_cpu.emplace_back(new Tensor(input2, Tensor::DT_FLOAT, 3));

    TensorList inputs_gpu;
    for(int i=0;i<inputs_cpu.size();i++){
        inputs_gpu.emplace_back(new Tensor(Tensor::DT_FLOAT, 3));
    }

    outputs_gpu.emplace_back(new Tensor(Tensor::DT_FLOAT, 3));

    // download to gpu
    for(unsigned int i=0;i<inputs_cpu.size();++i){
        context->CopyCPUTensorToDevice(inputs_cpu[i], inputs_gpu[i]);
    }

    auto binary_kernel = BinaryKernel(context.get());
    binary_kernel.Compute(inputs_gpu, outputs_gpu);

    context->Finish();

    TensorList outputs_cpu;
    // upload to cpu
    for(unsigned int i=0;i<outputs_gpu.size();++i){
        auto output_cpu_tensor = new Tensor(Tensor::DT_FLOAT, 3);
        context->CopyDeviceTensorToCPU(output_cpu_tensor, outputs_gpu[i]);
        outputs_cpu.emplace_back(output_cpu_tensor);
    }

    // clean up
    for(unsigned int i=0;i<outputs_gpu.size();++i){
        // clean up outputs
        for(int j=0;j<outputs_gpu[i]->num_elements();j++){
            std::cout<<((float*)outputs_gpu[i]->host())[j]<<std::endl;
        }
    }
    for(unsigned int i=0;i<inputs_gpu.size();++i){
        // clean up outputs
    }

    // ShaderBuffer input(1<<5);
    // ShaderBuffer output(1<<5);

    // context.Compute({1,2,3});

    return 0;
}
