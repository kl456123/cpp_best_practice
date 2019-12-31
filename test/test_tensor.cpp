#include "test_suite.h"
#include <assert.h>
#include "test_helper.h"
#include "core/tensor.h"
#include "backend.h"
#include <memory>
#include <assert.h>
#include <cmath>

#include "core/pool.h"


class TensorTestCase : public TestCase{
    public:
        bool run(){
            auto backend_type = Backend::ForwardType::OPENCL;
            Backend* gpu_backend = ExtractBackend(backend_type);


            std::shared_ptr<Tensor> tensor;
            std::vector<int> input_shape({1,1,5,5});

            // allocate in cpu
            tensor.reset(Tensor::Ones(input_shape));
            assert(CompareTensor(tensor->host<float>(), float(1.0), tensor->size()));

            // copy to device(gpu)
            tensor->CopyToDevice(backend_type);

            assert(gpu_backend->pool()->used_size()==1);

            // allocate in gpu directly
            std::shared_ptr<Tensor> tensor_gpu;
            tensor_gpu.reset(new Tensor(input_shape, Tensor::DataType::FLOAT32, backend_type));
            assert(gpu_backend->pool()->used_size()==2);

            // release buffer
            tensor_gpu.reset();
            assert(gpu_backend->pool()->used_size()==1);
            tensor.reset();
            assert(gpu_backend->pool()->used_size()==0);
            return true;
        }
};


TestSuiteRegister(TensorTestCase, "TensorTestCase");
