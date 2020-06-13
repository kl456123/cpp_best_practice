#include "opengl/nn/kernels/kernel_test_utils.h"
#include "opengl/core/functor.h"
#include "opengl/core/driver.h"


using namespace ::opengl::testing;
using ::opengl::CopyHostToTexture;

namespace opengl{
    namespace{
        void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Tensor* device_tensor){
            CHECK(cpu_tensor->dformat()== ::dlxnet::TensorProto::ANY
                    || cpu_tensor->dformat()== ::dlxnet::TensorProto::NHWC);
            CHECK_EQ(device_tensor->dformat(), ::dlxnet::TensorProto::ANY);

            CHECK(cpu_tensor->is_host());
            CHECK(!device_tensor->is_host());

            // check same bytes
            CHECK_EQ(cpu_tensor->AllocatedSize(), device_tensor->AllocatedSize());

            auto texture = device_tensor->device<Texture>();
            const int width = texture->width();
            const int height = texture->height();
            GLenum format = texture->format();
            GLenum type = texture->type();
            // TODO(breakpoint) why DMA is slower than non DMA
            CopyHostToTexture(cpu_tensor->host(), width, height, device_tensor->device<Texture>()->id(),
                    format, type);
        }

        void CopyDeviceTensorToCPU(const Tensor* device_tensor, Tensor* cpu_tensor){
            CHECK_EQ(cpu_tensor->dformat(), ::dlxnet::TensorProto::ANY);
            CHECK_EQ(device_tensor->dformat(), ::dlxnet::TensorProto::ANY);

            CHECK(cpu_tensor->is_host());
            CHECK(!device_tensor->is_host());

            // check same bytes
            CHECK_EQ(cpu_tensor->AllocatedSize(), device_tensor->AllocatedSize());

            auto texture = device_tensor->device<Texture>();
            const int width = texture->width();
            const int height = texture->height();
            GLenum format = texture->format();
            GLenum type = texture->type();
            // TODO(breakpoint) why DMA is slower than non DMA
            CopyTextureToHost(cpu_tensor->host(), width, height, device_tensor->device<Texture>()->id(),
                    format, type);
        }

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
        for(int bz=0;bz<3;bz++){
            for(int size=1;size<=256;size*=2){
                for(int channel=1;channel<=20;channel++){
                    const IntList shape{1, size, size, channel};
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
            }
        }
    }

    TEST(FunctorTest, ANYToANY4ToANYTest){
        auto ctx = GetContext();
        DIFFERENT_SHAPE_LOOP_START;
        auto src_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY));
        auto src_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY));
        auto dst_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY4));
        auto actual_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY));

        Tensor* actual_cpu_tensor = actual_cpu_tensor_ptr.get();
        Tensor* src_cpu_tensor = src_cpu_tensor_ptr.get();
        Tensor* src_gpu_tensor = src_gpu_tensor_ptr.get();
        Tensor* dst_gpu_tensor = dst_gpu_tensor_ptr.get();

        // any to any (host->device)
        CopyCPUTensorToDevice(src_cpu_tensor, src_gpu_tensor);
        // any to any4 (device->device)
        functor::ConvertTensorANYToANY4()(ctx, src_gpu_tensor, dst_gpu_tensor);
        // any4 to any (device->device)
        functor::ConvertTensorANY4ToANY()(ctx, dst_gpu_tensor, src_gpu_tensor);
        // any to any (device->host)
        CopyDeviceTensorToCPU(src_gpu_tensor, actual_cpu_tensor);

        CheckSameTensor(src_cpu_tensor, actual_cpu_tensor);

        DIFFERENT_SHAPE_LOOP_END;
    }

    TEST(FunctorTest, ANYToNHWC4Test){
        auto ctx = GetContext();
        DIFFERENT_SHAPE_LOOP_START;
        auto src_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::NHWC));
        auto src_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT,
                    shape, Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY));

        auto dst_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::NHWC4));
        auto actual_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::NHWC));

        Tensor* actual_cpu_tensor = actual_cpu_tensor_ptr.get();
        Tensor* src_cpu_tensor = src_cpu_tensor_ptr.get();
        Tensor* src_gpu_tensor = src_gpu_tensor_ptr.get();
        Tensor* dst_gpu_tensor = dst_gpu_tensor_ptr.get();

        CopyCPUTensorToDevice(src_cpu_tensor, src_gpu_tensor);
        functor::ConvertTensorANYToNHWC4()(ctx, src_gpu_tensor, dst_gpu_tensor);
        ctx->CopyDeviceTensorToCPU(dst_gpu_tensor, actual_cpu_tensor);
        // CopyDeviceTensorToCPU(src_gpu_tensor, actual_cpu_tensor);

        CheckSameTensor(src_cpu_tensor, actual_cpu_tensor);
        DIFFERENT_SHAPE_LOOP_END;
    }
}// namespace opengl
