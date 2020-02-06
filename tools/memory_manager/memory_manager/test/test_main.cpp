#include <iostream>
#include <vector>

#include "memory_manager/core/tensor.h"
#include "memory_manager/core/tensor_shape.h"
#include "memory_manager/utils/logging.h"
#include "types.pb.h"


int main(){
    auto tensor1 = Tensor();
    CHECK(tensor1.IsInitialized())<<"tensor1 should be initialized!";
    std::vector<int64_t> dim_sizes({1,3,224,224});
    auto tensor_shape = TensorShape(dim_sizes);
    auto tensor2 = Tensor(DataType::DT_FLOAT, tensor_shape);
    std::cout<<tensor2.shape()<<std::endl;
    Allocator* allocator = cpu_allocator();
    std::cout<<allocator->GetStats()->DebugString()<<std::endl;
    return 0;
}
