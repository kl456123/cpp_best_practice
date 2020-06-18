#include "opengl/nn/kernels/concat.h"

#include "opengl/core/program.h"
#include "opengl/core/fbo_session.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel_registry.h"
#include "opengl/utils/util.h"


namespace opengl{
    namespace{
        int MakeAxisToNHWC(const int& src_axis){
            if(src_axis==1){
                return 3;
            }
            if(src_axis==0){
                return 0;
            }
            if(src_axis==2){
                return 1;
            }
            if(src_axis==3){
                return 2;
            }
        }
    }
    ConcatKernel::ConcatKernel(Context* context)
        :Kernel(context){
            kernel_fname_ = "../opengl/nn/glsl/concat_multi.glsl";
        }

    void ConcatKernel::SetupAttr(const dlxnet::Attribute& attr){
        auto& concat_params = attr.concat_attr();
        axis_= concat_params.axis();
    }

    void ConcatKernel::Compute(TensorList& inputs, TensorList& outputs){
        DLOG(INFO)<<"ConcatKernel Inputs: "<<inputs.size();
        program_->Activate();
        // auto other_image = inputs[1]->device<Texture>();


        int axis = axis_+4-inputs[0]->shape().size();


        int axis_offset=0;
        SetFrameBuffer(outputs);
        SetVertexShader();
        OPENGL_CALL(glClear(GL_COLOR_BUFFER_BIT));
        program_->set_int("axis", axis);
        program_->set_vec4i("output_shape", AmendShape(outputs[0]->shape()));

        for(int i=0;i<inputs.size();++i){
            program_->set_int("axis_offset", axis_offset);

            // set shape arguments
            auto input_shape = AmendShape(inputs[i]->shape());
            program_->set_vec4i("input_shape", input_shape);
            // input
            {
                auto input_image = inputs[i]->device<Texture>();
                program_->set_image2D("input_image", input_image->id(),  0);
                OPENGL_CHECK_ERROR;
            }

            // origin image
            {
                auto output_image = outputs[0]->device<Texture>();
                program_->set_image2D("origin_image", output_image->id(),  1);
                OPENGL_CHECK_ERROR;
            }

            OPENGL_CALL(glDrawArrays(GL_TRIANGLES, 0, 6));
            glFinish();
            axis_offset+=input_shape[axis];
        }
    }

    void ConcatKernel::InferOutputShape(const TensorList& input_tensors,
            TensorShapeList& output_shapes){
        // set output dformat first, then we can according
        // to dformat to infer output shape
        output_tensor_dformats_.emplace_back(dlxnet::TensorProto::ANY4);
        if(axis_==-1){
            axis_=input_tensors[0]->shape().size()-1;
        }

        output_shapes.clear();
        output_shapes.resize(1);
        CHECK_GE(input_tensors.size(), 2);
        output_shapes[0] = input_tensors[0]->shape();
        output_shapes[0][axis_] = 0;
        for(auto& input_tensor: input_tensors){
            auto input_shape = input_tensor->shape();
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

