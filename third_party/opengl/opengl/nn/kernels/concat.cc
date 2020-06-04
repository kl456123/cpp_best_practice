#include "opengl/nn/kernels/concat.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"


namespace opengl{
    namespace{
        IntList AmendShape(const IntList& shape){
            CHECK_LE(shape.size(), 4);
            const int remain_dims = 4-shape.size();
            IntList amended_shape = shape;
            for(int i=0;i<remain_dims;++i){
                amended_shape.insert(amended_shape.begin(), 1);
            }
            return amended_shape;
        }
    }
    ConcatKernel::ConcatKernel(Context* context)
        :Kernel(context){
            kernel_fname_ = "../opengl/nn/glsl/concat.glsl";
        }

    void ConcatKernel::SetupAttr(const dlxnet::Attribute& attr){
        auto& concat_params = attr.concat_attr();
        axis_= concat_params.axis();
    }

    void ConcatKernel::Compute(TensorList& inputs, TensorList& outputs){
        DLOG(INFO)<<"ConcatKernel Inputs: "<<inputs.size();
        program_->Activate();
        auto input_image = inputs[0]->device<Texture>();
        auto other_image = inputs[1]->device<Texture>();

        SetFrameBuffer(outputs);
        SetVertexShader();
        program_->set_int("axis", axis_+4-inputs[0]->shape().size());

        // set shape arguments
        program_->set_vec4i("input_shape", AmendShape(inputs[0]->shape()));
        program_->set_vec4i("other_shape", AmendShape(inputs[1]->shape()));
        program_->set_vec4i("output_shape", AmendShape(outputs[0]->shape()));

        // input
        {
            program_->set_image2D("input_image", input_image->id(),  0);
            OPENGL_CHECK_ERROR;
        }

        {
            program_->set_image2D("other_image", other_image->id(),  1);
            OPENGL_CHECK_ERROR;
        }

        OPENGL_CALL(glClear(GL_COLOR_BUFFER_BIT));
        OPENGL_CALL(glDrawArrays(GL_TRIANGLES, 0, 6));
        glFinish();
    }

    void ConcatKernel::InferOutputShape(TensorShapeList& input_shapes,
            TensorShapeList& output_shapes){
        // set output dformat first, then we can according
        // to dformat to infer output shape
        output_tensor_dformats_.emplace_back(dlxnet::TensorProto::ANY4);

        output_shapes.clear();
        output_shapes.resize(1);
        // CHECK_EQ(input_shapes.size(), 4);
        CHECK_GE(input_shapes.size(), 2);
        output_shapes[0] = input_shapes[0];
        output_shapes[0][axis_] = 0;
        for(auto& input_shape: input_shapes){
            CHECK_EQ(output_shapes[0].size(), input_shape.size());
            for(int i=0;i<input_shape.size();++i){
                if(i!=axis_){
                    // all dim is the same except axis_ dim
                    CHECK_EQ(input_shape[i], output_shapes[0][i]);
                }else{
                    // add in axis dim
                    output_shapes[0][i]+=input_shape[i];
                }
            }
        }
    }

    ConcatKernel::~ConcatKernel(){}

    REGISTER_KERNEL_WITH_NAME(ConcatKernel, "Concat");
}//namespace opengl

