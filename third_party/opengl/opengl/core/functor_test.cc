#include "opengl/nn/kernels/kernel_test_utils.h"
#include "opengl/core/functor.h"


using namespace ::opengl::testing;

namespace opengl{
    namespace{
        void NHWCToANYCPU(const Tensor* src_tensor, Tensor* dst_tensor){
            const float* src_data = src_tensor->host<float>();
            float* dst_data = dst_tensor->host<float>();
            CHECK_EQ(src_tensor->num_elements(), dst_tensor->num_elements());
            const int num_elements = src_tensor->num_elements();
            for(int i=0;i<num_elements;++i){
                dst_data[i] = src_data[i];
            }
        }
    }// namespace

    TEST(FunctorTest, NHWC4ToANY4Test){
        auto ctx = GetContext();
        const IntList shape{1, 2, 3, 5};
        // cpu tensor
        auto src_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::NHWC));
        auto expect_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY));
        auto actual_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY));
        // gpu tensor
        auto src_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::NHWC4));
        auto dst_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY4));

        Tensor* actual_cpu_tensor = actual_cpu_tensor_ptr.get();
        Tensor* src_cpu_tensor = src_cpu_tensor_ptr.get();
        Tensor* expect_cpu_tensor = expect_cpu_tensor_ptr.get();
        Tensor* src_gpu_tensor = src_gpu_tensor_ptr.get();
        Tensor* dst_gpu_tensor = dst_gpu_tensor_ptr.get();

        // cpu computation
        NHWCToANYCPU(src_cpu_tensor, expect_cpu_tensor);

        // gpu computation
        ctx->CopyCPUTensorToDevice(src_cpu_tensor, src_gpu_tensor);
        functor::ConvertTensorNHWC4ToANY4()(ctx, src_gpu_tensor, dst_gpu_tensor);
        ctx->CopyDeviceTensorToCPU(dst_gpu_tensor, actual_cpu_tensor);

        CheckSameTensor(expect_cpu_tensor, actual_cpu_tensor);

    }
}// namespace opengl
