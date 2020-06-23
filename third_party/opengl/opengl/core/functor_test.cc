#include "opengl/nn/kernels/kernel_test_utils.h"
#include "opengl/core/functor.h"
#include "opengl/core/cpu_functor.h"
#include "opengl/core/driver.h"


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

        void ConvertTensorNHWCToNHWC4(const Tensor* tensor, void** out){
            // otherwise fall to nhwc4 case
            const int image_height = tensor->num()*tensor->height();
            const int image_width = UP_DIV(tensor->channel(), 4) * tensor->width();
            const int orig_channel = tensor->channel();
            size_t num_elements = image_height * image_width * 4;
            // copy from data to host_
            float* data = new float[num_elements];
            memset(data, 0, sizeof(float)*num_elements);
            float* orig_data = tensor->host<float>();
            const int up_channel = UP_DIV(tensor->channel(), 4)*4;
            for(int i=0;i<num_elements;++i){
                if(i%up_channel<orig_channel){
                    data[i] = orig_data[i/up_channel*orig_channel+i%up_channel];
                }
            }
            *out = data;
        }

        void ConvertTensorHWN4C4ToNCHW(void* src, Tensor* tensor){
            float* nchw_data = tensor->host<float>();
            float* hwn4c4_data = (float*)src;
            const int num_elements = tensor->num_elements();
            const int c = tensor->shape()[1];
            const int h = tensor->shape()[2];
            const int w = tensor->shape()[3];
            const int n = tensor->shape()[0];
            const int up_channel = UP_DIV(c, 4)*4;
            const int n4 = UP_DIV(n, 4);
            const int c4 = UP_DIV(c, 4);

            for(int i=0; i<num_elements; ++i){
                int cur = i;
                const int w_i = cur%w;
                cur/=w;
                const int h_i = cur%h;
                cur/=h;
                const int c_i = cur%c;
                cur/=c;
                const int n_i = cur;
                const int offset = ((((h_i*w+w_i)*n4+n_i/4)*c4+c_i/4)*4+c_i%4)*4+n_i%4;

                nchw_data[i]=hwn4c4_data[offset];
            }
        }

        void ConvertTensorToStride4(const Tensor* src_tensor, void** out){
            auto shape = src_tensor->shape();
            const int num_dims = shape.size();
            const int src_last_dim = shape[num_dims-1];
            const int dst_last_dim = UP_ROUND(src_last_dim, 4);
            const int src_num_elements = src_tensor->num_elements();
            const int dst_num_elements = src_num_elements / src_last_dim * dst_last_dim;

            float* dst_data = new float[dst_num_elements];
            const float* src_data = src_tensor->host<float>();
            memset(dst_data, 0, sizeof(float)*dst_num_elements);
            // only use data if it is in src
            for(int i=0;i<src_num_elements;++i){
                const int dst_index = i/src_last_dim*dst_last_dim+i%src_last_dim;
                dst_data[dst_index] = src_data[i];
            }

            *out = dst_data;
        }

        void ConvertTensorNHWC4ToNHWC(void* out, Tensor* tensor){
            // tensor->set_host(out);
            float* nhwc_data = tensor->host<float>();
            float* nhwc4_data = (float*)out;
            const int num_elements = tensor->num_elements();
            const int up_channel = UP_DIV(tensor->last_stride(), 4)*4;
            const int channel = tensor->last_stride();

            // there is different in their base number in the last dim(one is channel,
            // the other is up_channel)
            for(int i=0;i<num_elements;++i){
                const int offset = i/channel*up_channel+i%channel;
                nhwc_data[i] = nhwc4_data[offset];
            }
        }

        void ConvertTensorFromStride4(void* src, Tensor* dst_tensor){
            auto shape = dst_tensor->shape();
            const int num_dims = shape.size();
            const int dst_last_dim = shape[num_dims-1];
            const int dst_num_elements = dst_tensor->num_elements();
            const int src_last_dim = UP_ROUND(dst_last_dim, 4);
            float* src_data = (float*)src;
            float* dst_data = dst_tensor->host<float>();
            for(int i=0; i < dst_num_elements; ++i){
                const int src_index = i%dst_last_dim+i/dst_last_dim*src_last_dim;
                dst_data[i] = src_data[src_index];
            }
        }

        void ConvertTensorANYToANY4(const Tensor* src_tensor, void** out){
            auto shape = src_tensor->shape();
            const int num_dims = shape.size();
            const int src_last_dim = shape[num_dims-1];
            const int dst_last_dim = UP_ROUND(src_last_dim, 4);
            const int src_num_elements = src_tensor->num_elements();
            const int dst_num_elements = src_num_elements / src_last_dim * dst_last_dim;

            float* dst_data = new float[dst_num_elements];
            const float* src_data = src_tensor->host<float>();
            memset(dst_data, 0, sizeof(float)*dst_num_elements);
            // only use data if it is in src
            for(int i=0;i<src_num_elements;++i){
                const int dst_index = i/src_last_dim*dst_last_dim+i%src_last_dim;
                dst_data[dst_index] = src_data[i];
            }

            *out = dst_data;
        }

        void ConvertTensorNCHWToNHWC4(const Tensor* cpu_tensor, void** out){

            const int n = cpu_tensor->shape()[0];
            const int c = cpu_tensor->shape()[1];
            const int h = cpu_tensor->shape()[2];
            const int w = cpu_tensor->shape()[3];

            const int num_elements = n*UP_DIV(c, 4)*4*h*w;
            float* data = new float[num_elements];
            memset(data, 0, sizeof(float)*num_elements);
            float* orig_data = cpu_tensor->host<float>();
            for(int i=0;i<cpu_tensor->num_elements();++i){
                int cur = i;
                const int w_i = cur%w;
                cur/=w;
                const int h_i = cur%h;
                cur/=h;
                const int c_i = cur%c;
                cur/=c;
                const int n_i = cur;
                const int offset = (((n_i*h+h_i)*w+w_i)*UP_DIV(c, 4)+c_i/4)*4+c_i%4;
                data[offset] = orig_data[i];
            }
            *out = data;
        }

        void ConvertTensorNCHWToHWN4C4(const Tensor* tensor, void** out){
            // handle pytorch filter
            // from (out, in, h, w) to (h*w, out_4*in_4*in4, out4)
            // where in_4 = UP_DIV(in, 4), out_4=UP_DIV(out, 4), in4=out4=4
            const int n_out = tensor->shape()[0];
            const int n_in = tensor->shape()[1];
            const int h = tensor->shape()[2];
            const int w = tensor->shape()[3];
            const int in_4 = UP_DIV(n_in, 4);
            const int out_4 = UP_DIV(n_out, 4);
            const int num_elements = h*w*in_4*out_4*4*4;
            float* data = new float[num_elements];
            memset(data, 0, sizeof(float)*num_elements);
            float* orig_data = tensor->host<float>();
            for(int i=0;i<tensor->num_elements();++i){
                // decompose i to four-element tuple
                int cur = i;
                const int w_i = cur%w;
                cur /= w;
                const int h_i = cur%h;
                cur/=h;
                const int in_i = cur%n_in;
                cur/=n_in;
                const int out_i = cur;

                // then compose them to 3-element tuple
                const int hw_i = h_i*w+w_i;
                const int io4_i = (out_i/4 *in_4+in_i/4)*4+in_i%4;
                const int offset = (hw_i*in_4*out_4*4+io4_i)*4+out_i%4;
                data[offset] = orig_data[i];
            }
            *out = data;
        }
    }// namespace

    TEST(FunctorTest, NHWC4ToANY4Test){
        auto ctx = GetContext();
        DIFFERENT_SHAPE_LOOP_START;
        // IntList shape{2, 5};
        auto src_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::NHWC4));
        auto src_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY4));
        auto dst_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY4));
        auto expected_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY4));
        auto actual_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY4));

        Tensor* actual_tensor = actual_cpu_tensor_ptr.get();
        Tensor* src_cpu_tensor = src_cpu_tensor_ptr.get();
        Tensor* src_gpu_tensor = src_gpu_tensor_ptr.get();
        Tensor* dst_gpu_tensor = dst_gpu_tensor_ptr.get();
        Tensor* expected_tensor =  expected_tensor_ptr.get();

        // any4 to any4 (host->device)
        CopyCPUTensorToDevice(src_cpu_tensor, src_gpu_tensor);
        // any4 to any (device->device)
        functor::ConvertTensorNHWC4ToANY4()(ctx, src_gpu_tensor, dst_gpu_tensor);
        CopyDeviceTensorToCPU(dst_gpu_tensor, actual_tensor);

        host_functor::ConvertTensorNHWC4ToANY4()(ctx, src_cpu_tensor, expected_tensor);

        CheckSameTensor(expected_tensor, actual_tensor);
        DIFFERENT_SHAPE_LOOP_END;
    }

    TEST(FunctorTest, ANY4ToANYTest){
        auto ctx = GetContext();
        DIFFERENT_SHAPE_LOOP_START;
        // IntList shape{2, 5};
        auto src_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY4));
        auto src_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY4));
        auto dst_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY));
        auto expected_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY));
        auto actual_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY));

        Tensor* actual_tensor = actual_cpu_tensor_ptr.get();
        Tensor* src_cpu_tensor = src_cpu_tensor_ptr.get();
        Tensor* src_gpu_tensor = src_gpu_tensor_ptr.get();
        Tensor* dst_gpu_tensor = dst_gpu_tensor_ptr.get();
        Tensor* expected_tensor =  expected_tensor_ptr.get();

        // any4 to any4 (host->device)
        CopyCPUTensorToDevice(src_cpu_tensor, src_gpu_tensor);
        // any4 to any (device->device)
        functor::ConvertTensorANY4ToANY()(ctx, src_gpu_tensor, dst_gpu_tensor);
        CopyDeviceTensorToCPU(dst_gpu_tensor, actual_tensor);

        host_functor::ConvertTensorANY4ToANY()(ctx, src_cpu_tensor, expected_tensor);

        CheckSameTensor(expected_tensor, actual_tensor);
        DIFFERENT_SHAPE_LOOP_END;
    }

    TEST(FunctorTest, ANYToANY4Test){
        auto ctx = GetContext();
        DIFFERENT_SHAPE_LOOP_START;
        // IntList shape{2, 3};
        auto src_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY));
        auto src_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY));
        auto dst_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY4));
        auto actual_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY4));
        auto expected_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY4));

        Tensor* actual_cpu_tensor = actual_cpu_tensor_ptr.get();
        Tensor* src_cpu_tensor = src_cpu_tensor_ptr.get();
        Tensor* src_gpu_tensor = src_gpu_tensor_ptr.get();
        Tensor* dst_gpu_tensor = dst_gpu_tensor_ptr.get();
        Tensor* expected_cpu_tensor = expected_cpu_tensor_ptr.get();

        // any to any (host->device)
        CopyCPUTensorToDevice(src_cpu_tensor, src_gpu_tensor);
        // any to any4 (device->device)
        functor::ConvertTensorANYToANY4()(ctx, src_gpu_tensor, dst_gpu_tensor);
        // any to any (device->host)
        CopyDeviceTensorToCPU(dst_gpu_tensor, actual_cpu_tensor);
        host_functor::ConvertTensorANYToANY4()(ctx, src_cpu_tensor, expected_cpu_tensor);
        // Sync();

        CheckSameTensor(expected_cpu_tensor, actual_cpu_tensor);
        DIFFERENT_SHAPE_LOOP_END;
    }

    TEST(FunctorTest, ANYToNHWC4Test){
        auto ctx = GetContext();
        // DIFFERENT_SHAPE_LOOP_START;
        IntList shape{1, 2, 2, 2};
        auto src_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY));
        auto src_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY));
        auto dst_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::NHWC4));
        auto actual_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::NHWC4));
        auto expected_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::NHWC4));

        Tensor* actual_cpu_tensor = actual_cpu_tensor_ptr.get();
        Tensor* src_cpu_tensor = src_cpu_tensor_ptr.get();
        Tensor* src_gpu_tensor = src_gpu_tensor_ptr.get();
        Tensor* dst_gpu_tensor = dst_gpu_tensor_ptr.get();
        Tensor* expected_cpu_tensor = expected_cpu_tensor_ptr.get();

        // any to any (host->device)
        CopyCPUTensorToDevice(src_cpu_tensor, src_gpu_tensor);
        // any to any4 (device->device)
        functor::ConvertTensorANYToNHWC4()(ctx, src_gpu_tensor, dst_gpu_tensor);
        // any to any (device->host)
        CopyDeviceTensorToCPU(dst_gpu_tensor, actual_cpu_tensor);
        host_functor::ConvertTensorANYToNHWC4()(ctx, src_cpu_tensor, expected_cpu_tensor);
        // Sync();

        CheckSameTensor(expected_cpu_tensor, actual_cpu_tensor);
        // DIFFERENT_SHAPE_LOOP_END;
    }

    TEST(FunctorTest, NCHWToHWN4C4Test){
        auto ctx = GetContext();
        // DIFFERENT_SHAPE_LOOP_START;
        IntList shape = {1,2,3,6};
        auto src_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::NCHW));
        auto src_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::NCHW));
        auto dst_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::HWN4C4));
        auto actual_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::HWN4C4));
        auto expected_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Empty(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::HWN4C4));

        Tensor* actual_cpu_tensor = actual_cpu_tensor_ptr.get();
        Tensor* src_cpu_tensor = src_cpu_tensor_ptr.get();
        Tensor* src_gpu_tensor = src_gpu_tensor_ptr.get();
        Tensor* dst_gpu_tensor = dst_gpu_tensor_ptr.get();
        Tensor* expected_cpu_tensor = expected_cpu_tensor_ptr.get();

        // any to any (host->device)
        CopyCPUTensorToDevice(src_cpu_tensor, src_gpu_tensor);
        // any to any4 (device->device)
        functor::ConvertTensorNCHWToHWN4C4()(ctx, src_gpu_tensor, dst_gpu_tensor);
        // any to any (device->host)
        CopyDeviceTensorToCPU(dst_gpu_tensor, actual_cpu_tensor);
        host_functor::ConvertTensorNCHWToHWN4C4()(ctx, src_cpu_tensor, expected_cpu_tensor);
        // Sync();

        CheckSameTensor(expected_cpu_tensor, actual_cpu_tensor);
        // DIFFERENT_SHAPE_LOOP_END;
    }

    TEST(FunctorTest, DataLayoutTest){
        auto ctx = GetContext();

        // DIFFERENT_SHAPE_LOOP_START;
        IntList shape = {1,2,3,6};
        auto src_gpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY));
        auto actual_cpu_tensor_ptr = std::unique_ptr<Tensor>(new Tensor(Tensor::DT_FLOAT, shape,
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY));
        auto dst_cpu_tensor_ptr = std::unique_ptr<Tensor>(Tensor::Random(Tensor::DT_FLOAT, shape,
                    dlxnet::TensorProto::ANY));
        Tensor* actual_tensor = actual_cpu_tensor_ptr.get();
        Tensor* src_gpu_tensor = src_gpu_tensor_ptr.get();
        Tensor* dst_cpu_tensor = dst_cpu_tensor_ptr.get();

        functor::ConvertTensorTest()(ctx, src_gpu_tensor, actual_tensor);
        CopyDeviceTensorToCPU(actual_tensor, dst_cpu_tensor);
    }
}// namespace opengl
