#include "opengl/nn/kernels/pool.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"


namespace opengl{
    template<PoolType pool_type>
        PoolKernel<pool_type>::PoolKernel(Context* context)
        :Kernel(context){
            kernel_fname_ = "../opengl/nn/glsl/pool.glsl";
        }

    template<PoolType pool_type>
        void PoolKernel<pool_type>::SetupAttr(const dlxnet::Attribute& attr){
            // set single output dformat for all typed pool kernels
            output_tensor_dformats_.emplace_back(dlxnet::TensorProto::NHWC4);
            if(pool_type_==GlobalAveragePool){
                return;
            }
            auto& maxpool_params = attr.maxpool_attr();

            // handle pads
            CHECK_EQ(maxpool_params.pads_size(), 4);
            for(auto& pad: maxpool_params.pads()){
                CHECK_EQ(maxpool_params.pads(0),pad);
            }
            padding_ = maxpool_params.pads(0);

            // handle stride
            CHECK_EQ(maxpool_params.strides_size(), 2);
            CHECK_EQ(maxpool_params.strides(0), maxpool_params.strides(1));
            stride_ = maxpool_params.strides(0);


            // handle kernel
            CHECK_EQ(maxpool_params.kernel_shape_size(), 2);
            CHECK_EQ(maxpool_params.kernel_shape(0), maxpool_params.kernel_shape(1));
            kernel_size_=maxpool_params.kernel_shape(0);
        }
    template<PoolType pool_type>
        void PoolKernel<pool_type>::Compute(TensorList& inputs, TensorList& outputs){
            program_->Activate();
            auto input_image = inputs[0]->device<Texture>();
            SetFrameBuffer(outputs);
            SetVertexShader();


            auto input_shape = inputs[0]->shape();
            auto output_shape = outputs[0]->shape();

            program_->set_vec3i("input_shape", inputs[0]->height(),
                    inputs[0]->width(), inputs[0]->channel());
            program_->set_vec3i("output_shape", outputs[0]->height(),
                    outputs[0]->width(), outputs[0]->channel());
            program_->set_int("padding", padding_);
            program_->set_int("kernel_size", kernel_size_);
            program_->set_int("stride_size", stride_);
            program_->set_int("pool_type", pool_type_);
            // input
            {
                program_->set_image2D("input_image", input_image->id(),  0);
                OPENGL_CHECK_ERROR;
            }

            OPENGL_CALL(glClear(GL_COLOR_BUFFER_BIT));
            OPENGL_CALL(glDrawArrays(GL_TRIANGLES, 0, 6));
            glFinish();
        }
    template<PoolType pool_type>
        void PoolKernel<pool_type>::InferOutputShape(TensorShapeList& input_shapes,
                TensorShapeList& output_shapes){
            CHECK_EQ(input_shapes.size(), 1);
            output_shapes.clear();
            output_shapes.resize(1);
            auto& image_shape = input_shapes[0];
            if(pool_type_==GlobalAveragePool){
                output_shapes[0]={image_shape[0],1, 1, image_shape[3]};
                // set pool params according to the input shape
                stride_=1;
                // spatial dims
                CHECK_EQ(image_shape[1], image_shape[2]);
                kernel_size_=image_shape[1];
                padding_=0;
            }else{
                // compute output shape like conv2d
                const int output_height = (image_shape[1]-kernel_size_+2*padding_)/stride_+1;
                const int output_width = (image_shape[2]-kernel_size_+2*padding_)/stride_+1;
                output_shapes[0] = {image_shape[0], output_height, output_width, image_shape[3]};
            }

        }

    template<PoolType pool_type>
        void PoolKernel<pool_type>::InferOutputShape(const TensorList& input_tensors,
                TensorShapeList& output_shapes){
            CHECK_EQ(input_tensors.size(), 1);
            output_shapes.clear();
            output_shapes.resize(1);
            if(pool_type_==GlobalAveragePool){
                output_shapes[0]={input_tensors[0]->num(),1, 1, input_tensors[0]->channel()};
                // set pool params according to the input shape
                stride_=1;
                // spatial dims
                CHECK_EQ(input_tensors[0]->width(), input_tensors[0]->height());
                kernel_size_=input_tensors[0]->width();
                padding_=0;
            }else{
                // compute output shape like conv2d
                const int output_height = (input_tensors[0]->height()-kernel_size_+2*padding_)/stride_+1;
                const int output_width = (input_tensors[0]->width()-kernel_size_+2*padding_)/stride_+1;
                output_shapes[0] = {input_tensors[0]->num(), output_height,
                    output_width, input_tensors[0]->channel()};
            }
        }

    template<PoolType pool_type>
        PoolKernel<pool_type>::~PoolKernel(){}

    REGISTER_KERNEL_WITH_NAME(PoolKernel<MaxPool>, "MaxPool");
    REGISTER_KERNEL_WITH_NAME(PoolKernel<AveragePool>, "AveragePool");
    REGISTER_KERNEL_WITH_NAME(PoolKernel<GlobalAveragePool>, "GlobalAveragePool");
}//namespace opengl
