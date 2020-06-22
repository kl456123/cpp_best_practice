#include "opengl/core/cpu_functor.h"
#include "opengl/utils/logging.h"
#include "opengl/core/tensor.h"


namespace opengl{
    namespace host_functor{
        void ConvertTensorANYToANY4::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            CHECK(src_tensor->is_host());
            CHECK(dst_tensor->is_host());
            const float* src_data = src_tensor->host<float>();
            float* dst_data = dst_tensor->host<float>();

            const int src_num_elements = src_tensor->AllocatedElements();
            const int src_last_dim = src_tensor->last_stride();
            const int dst_last_dim = UP_ROUND(src_last_dim, 4);

            memset(dst_data, 0, dst_tensor->AllocatedSize());
            // only use data if it is in src
            for(int i=0; i<src_num_elements; ++i){
                const int dst_index = i/src_last_dim*dst_last_dim+i%src_last_dim;
                if(dst_index<src_num_elements){
                    dst_data[dst_index] = src_data[i];
                }
            }
        }

        void ConvertTensorANY4ToANY::operator()(Context* ctx,
                const Tensor* src_tensor, Tensor* dst_tensor){
            CHECK(src_tensor->is_host());
            CHECK(dst_tensor->is_host());
            const float* src_data = src_tensor->host<float>();
            float* dst_data = dst_tensor->host<float>();

            const int dst_num_elements = dst_tensor->AllocatedElements();
            const int dst_last_dim = src_tensor->last_stride();
            const int src_last_dim = UP_ROUND(dst_last_dim, 4);
            const int src_num_elements = src_tensor->AllocatedElements();

            memset(dst_data, 0, dst_tensor->AllocatedSize());
            // only use data if it is in src
            for(int i=0; i<dst_num_elements; ++i){
                const int src_index = i/dst_last_dim*src_last_dim+i%dst_last_dim;
                if(src_index<src_num_elements){
                    dst_data[i] = src_data[src_index];
                }
            }
        }
    } // namespace host_functor
} // namespace opengl
