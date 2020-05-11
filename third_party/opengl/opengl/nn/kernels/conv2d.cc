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
        }

    void Conv2DKernel::SetupAttr(const dlxnet::Attribute& attr){
        auto& conv2d_params = attr.conv2d_attr();

        // handle pads
        CHECK_EQ(conv2d_params.pads_size(), 4);
        for(auto& pad: conv2d_params.pads()){
            CHECK_EQ(conv2d_params.pads(0),pad);
        }
        padding_ = conv2d_params.pads(0);

        // handle stride
        CHECK_EQ(conv2d_params.strides_size(), 2);
        CHECK_EQ(conv2d_params.strides(0), conv2d_params.strides(1));
        stride_ = conv2d_params.strides(0);


        // handle kernel
        CHECK_EQ(conv2d_params.kernel_shape_size(), 2);
        CHECK_EQ(conv2d_params.kernel_shape(0), conv2d_params.kernel_shape(1));
        kernel_size_=conv2d_params.kernel_shape(0);

        // set default for dilation and groups now
        group_size_=1;
        dilation_=1;

        output_tensor_dformats_.emplace_back(dlxnet::TensorProto::NHWC4);
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

    }

    void Conv2DKernel::InferOutputShape(TensorShapeList& input_shapes,
            TensorShapeList& output_shapes){
        // its order list as input, weights, bias
        CHECK_EQ(input_shapes.size(), 3);
        output_shapes.clear();
        output_shapes.resize(1);
        for(auto& input_shape:input_shapes){
            // check input is the same shape
        }
        // image_shape: (n, h, w, c)
        auto& image_shape = input_shapes[0];
        auto& filter_shape = input_shapes[1];
        // check the conv2d parameters accordind to filter shapes
        // check filter is valid
        // filter_shape: (n_out, n_in , h, w)
        CHECK_EQ(filter_shape[2], kernel_size_);
        CHECK_EQ(filter_shape[3], kernel_size_);
        // channel should be the same with input image
        CHECK_EQ(filter_shape[1], image_shape[3]);

        const int output_height = (image_shape[0]-kernel_size_+2*padding_+1)/stride_;
        const int output_width = (image_shape[1]-kernel_size_+2*padding_+1)/stride_;
        output_shapes[0] = {output_height, output_width};
    }

    REGISTER_KERNEL_WITH_NAME(Conv2DKernel, "Conv");
}//namespace opengl



