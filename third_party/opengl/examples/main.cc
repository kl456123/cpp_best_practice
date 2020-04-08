#include <stdio.h>
#include <memory>

#include "program.h"
#include "glut.h"
#include "buffer.h"
#include "context.h"
#include "kernels/binary.h"


int main(int argc, char** argv){
    /////////////////////////////////////
    // init and get statisic ////////////
    /////////////////////////////////////
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE);
    glutInitWindowSize(400, 300);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Hello world!");
    int maxtexsize;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE,&maxtexsize);
    printf("GL_MAX_TEXTURE_SIZE, %d\n",maxtexsize);

    // prepare program
    const char source[] = "";
    const std::string fname = "../examples/glsl/binary.glsl";

    Program program;
    program.Attach(fname)
        .Link();

    program.Activate();

    // prepare runtime
    auto context = std::unique_ptr<Context>(new Context(nullptr));

    //prepare input and output
    float input1[] = {1.0, 2.0, 3.0};
    float input2[] = {1.0, 2.0, 3.0};
    TensorList inputs_cpu;
    TensorList outputs_gpu;
    inputs_cpu.emplace_back(new Tensor(input1, Tensor::DT_FLOAT));
    inputs_cpu.emplace_back(new Tensor(input2, Tensor::DT_FLOAT));

    TensorList inputs_gpu;
    inputs_gpu.resize(inputs_cpu.size());

    // download to gpu
    for(unsigned int i=0;i<inputs_cpu.size();++i){
        context->CopyCPUTensorToDevice(inputs_cpu[i], inputs_gpu[i]);
    }

    auto binary_kernel = BinaryKernel(context.get());
    binary_kernel.Compute(inputs_gpu, outputs_gpu);

    TensorList outputs_cpu;
    outputs_cpu.resize(outputs_gpu.size());
    // upload to cpu
    for(unsigned int i=0;i<outputs_gpu.size();++i){
        context->CopyDeviceTensorToCPU(outputs_cpu[i], outputs_gpu[i]);
    }

    // clean up
    for(unsigned int i=0;i<outputs_gpu.size();++i){
        // clean up outputs
    }
    for(unsigned int i=0;i<inputs_gpu.size();++i){
        // clean up outputs
    }

    // ShaderBuffer input(1<<5);
    // ShaderBuffer output(1<<5);

    // context.Compute({1,2,3});
    // context.Finish();

    return 0;
}
