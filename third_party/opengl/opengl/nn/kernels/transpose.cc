#include "opengl/nn/kernels/transpose.h"

#include "opengl/core/program.h"
#include "opengl/core/context.h"
#include "opengl/utils/macros.h"
#include "opengl/core/functor.h"
#include "opengl/core/kernel_registry.h"


namespace opengl{
    namespace{
        IntList AmendPerm(const IntList& perms){
            IntList amended_perms;
            for(int i=0;i<4-perms.size();++i){
                amended_perms.emplace_back(i);
            }
            for(auto perm:perms){
                amended_perms.emplace_back(perm+4-perms.size());
            }
            return amended_perms;
        }
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

    void TransposeKernel::SelectKernel(const TensorList& inputs){
        if(inputs[0]->dformat()==dlxnet::TensorProto::ANY4){
            kernel_fname_ = "../opengl/nn/glsl/transpose_any4.glsl";
        }else{
            kernel_fname_ = "../opengl/nn/glsl/transpose.glsl";
        }

        // output dformat must be any4
        output_tensor_dformats_.emplace_back(dlxnet::TensorProto::ANY4);
    }

    TransposeKernel::TransposeKernel(Context* context)
        :Kernel(context){}

    void TransposeKernel::SetupAttr(const dlxnet::Attribute& attr){
        auto& transpose_params = attr.transpose_attr();
        for(auto item: transpose_params.perm()){
            perm_.emplace_back(item);
        }
        CHECK_LE(perm_.size(), 4);
    }

    void TransposeKernel::Compute(TensorList& inputs, TensorList& outputs){
        DLOG(INFO)<<"TransposeKernel Inputs: "<<inputs.size();
        Tensor* any4_tensor = nullptr;
        auto input_tensor = inputs[0];
        // if(input_tensor->dformat() == dlxnet::TensorProto::NHWC4){
            // // use tensor cache
            // VLOG(1)<<"Convert Tensor From NHWC4 To ANY4";
            // if(cached_any4_tensor_==nullptr){
                // any4_tensor = new Tensor(Tensor::DT_FLOAT, input_tensor->shape(),
                        // Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY4);
                // cached_any4_tensor_ = any4_tensor;
            // }else{
                // // assume it is the same as before
                // // it happens when loop inference
                // any4_tensor = cached_any4_tensor_;
            // }
            // functor::ConvertTensorNHWC4ToANY4()(GetContext(), input_tensor, any4_tensor);
        // }else{
            // any4_tensor = inputs[0];
        // }
        // CHECK_EQ(any4_tensor->dformat(), dlxnet::TensorProto::ANY4);
        program_->Activate();
        auto input_image = input_tensor->device<Texture>();

        program_->SetRetVal(outputs);
        // set params
        program_->set_vec4i("perm", AmendPerm(perm_));
        program_->set_vec4i("input_shape", AmendShape(input_tensor->shape()));
        program_->set_vec4i("output_shape", AmendShape(outputs[0]->shape()));

        // set args
        {
            OPENGL_CALL(program_->set_image2D("input_image", input_image->id(),  0));
        }
        program_->Run();
    }

    void TransposeKernel::InferOutputShape(TensorShapeList& input_shapes,
            TensorShapeList& output_shapes){
        output_shapes.clear();
        output_shapes.resize(1);
        CHECK_EQ(input_shapes.size(), 1);
        const auto input_shape = input_shapes[0];

        IntList& dst_shape=output_shapes[0];
        // permute dims in perms
        for(auto perm: perm_){
            dst_shape.emplace_back(input_shape[perm]);
        }
        // append the remain dims
        for(int i=perm_.size();i<input_shape.size();++i){
            dst_shape.emplace_back(input_shape[i]);
        }
    }

    TransposeKernel::~TransposeKernel(){}

    REGISTER_KERNEL_WITH_NAME(TransposeKernel, "Transpose");
}//namespace opengl
