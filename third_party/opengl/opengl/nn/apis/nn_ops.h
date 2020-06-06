#ifndef OPENGL_NN_APIS_NN_OPS_H_
#define OPENGL_NN_APIS_NN_OPS_H_
/**************************
 * This API Files Can be generated automatically
 */
#include "opengl/core/scope.h"
#include "opengl/core/types.h"

namespace opengl{
    struct Conv2dParams{
        int kernel_size;
        int stride;
        int padding;
    };

    struct ConcatParams{
        int axis;
    };
    int AddConstNode(Scope* scope, const std::string&  name, const Tensor* cpu_tensor);
    int AddConstNode(Scope* scope, const std::string&  name,
            const std::vector<int>& shape, DataFormat dst_dformat,
            DataFormat src_dformat);

    int AddConvNode(Scope* scope, const std::string&  name, std::vector<int> input_ids,
            const Conv2dParams& conv2d_params);
    int AddInputNode(Scope* scope, std::string name);

    int AddShapeNode(Scope* scope, std::string name, std::vector<int> input_ids);

    int AddConcatNode(Scope* scope, const std::string&  name, std::vector<int> input_ids,
            const ConcatParams& concat_params);
}// namespace opengl


#endif
