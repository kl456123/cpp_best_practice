#ifndef OPENGL_NN_KERNELS_KERNEL_TEST_UTILS_H_
#define OPENGL_NN_KERNELS_KERNEL_TEST_UTILS_H_
#include "opengl/core/fbo_session.h"
#include "opengl/core/init.h"
#include "opengl/test/test.h"
#include "opengl/nn/apis/nn_ops.h"
#include "opengl/utils/macros.h"

namespace opengl{
    namespace testing{
        // initialize environment
        std::unique_ptr<FBOSession> InitSession();
        void InitOGLContext();

        // check utils
        void CheckSameTensor(const Tensor* cpu_tensor1, const Tensor* cpu_tensor2);
        void CheckSameValueTensor(const Tensor* cpu_tensor1, const Tensor* cpu_tensor2);
        void CleanupTensorList(::opengl::TensorList* outputs_tensor);

        inline IntList ShapeToStride(IntList shape){
            IntList stride;
            int num_elements = 1;
            for(auto item:shape){
                num_elements*=item;
            }

            int temp = num_elements;
            for(auto item:shape){
                temp/=item;
                stride.emplace_back(temp);
            }
            return stride;
        }

        inline IntList OffsetToCoord(const int index, const IntList shape){
            auto stride = ShapeToStride(shape);
            IntList coords;
            for(int i=0;i<shape.size();++i){
                coords.emplace_back((index/stride[i])%shape[i]);
            }
            return coords;
        }

        inline int CoordToOffset(const IntList coords, const IntList shape){
            auto stride = ShapeToStride(shape);
            int offset = 0;
            // return {index%stride};
            for(int i=0;i<stride.size();++i){
                offset+=stride[i]*coords[i];
            }
            return offset;
        }
    }//namespace testing
}//namespace opengl


#endif
