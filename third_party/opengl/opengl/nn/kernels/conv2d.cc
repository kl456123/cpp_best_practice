#include "opengl/nn/kernels/conv2d.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"



namespace opengl{
    Conv2DKernel::Conv2DKernel(Context* context)
        :Kernel(context){
            // set work size
            for(int i=0;i<3;i++){
                work_sizes_[i] = 1;
            }
            kernel_fname_ = "../opengl/nn/glsl/conv2d.glsl";

            // param for conv2d
            // (TODO breakpoint) make it can be set by manually when load graph
            padding_ = 1;
            stride_ = 1;
            kernel_size_=3;
            group_size_=1;
            dilation_=1;
        }


    Conv2DKernel::~Conv2DKernel(){
        if(program_!=nullptr){delete program_;}
    }

    void Conv2DKernel::Compute(TensorList& inputs, TensorList& outputs){
        OPENGL_CALL(glUseProgram(program_->program_id()));
        auto texture1 = inputs[0]->device<Texture>();
        auto texture2 = inputs[1]->device<Texture>();
        SetFrameBuffer(outputs);
        SetVertexShader();


        program_->Activate();
        int tex_w = texture1->shape()[0];
        int tex_h = texture1->shape()[1];

        program_->set_vec2i("image_shape", tex_w, tex_h);
        program_->set_int("padding", padding_);
        program_->set_int("kernel_size", kernel_size_);
        program_->set_int("stride_size", stride_);
        OPENGL_CHECK_ERROR;
        // input
        {
            program_->set_image2D("input_image", texture1->id(),  0);
            OPENGL_CHECK_ERROR;
        }

        // filter
        {
            program_->set_image2D("input_filter", texture2->id(),  1);
            OPENGL_CHECK_ERROR;
        }

        OPENGL_CALL(glClear(GL_COLOR_BUFFER_BIT));
        OPENGL_CALL(glDrawArrays(GL_TRIANGLES, 0, 6));
        glFinish();
        // glDispatchCompute((GLuint)tex_w, (GLuint)tex_h, 1);

    }

    void Conv2DKernel::InferOutputShape(TensorShapeList& input_shapes,
            TensorShapeList& output_shapes){
        output_shapes.clear();
        output_shapes.resize(1);
        for(auto& input_shape:input_shapes){
            // check input is the same shape
        }
        auto& image_shape = input_shapes[0];
        auto& filter_shape = input_shapes[1];
        // check the conv2d parameters accordind to filter shapes
        CHECK_EQ(filter_shape[0], kernel_size_);
        CHECK_EQ(filter_shape[1], kernel_size_);

        const int output_height = (image_shape[0]-kernel_size_+2*padding_+1)/stride_;
        const int output_width = (image_shape[1]-kernel_size_+2*padding_+1)/stride_;
        // std::vector<int> output_shape();
        output_shapes[0] = {output_height, output_width};
        // output_shapes[0] = input_shapes[0];
    }

    REGISTER_KERNEL_WITH_NAME(Conv2DKernel, "Conv2d");
}//namespace opengl



