#include "opengl/nn/kernels/kernel_test_utils.h"
#include "opengl/nn/kernels/const.h"


using namespace ::opengl::testing;

namespace opengl{
    namespace{
        const IntList shape{2, 6, 10};
        const ::dlxnet::ModelProto BuildGraph(const Tensor* cpu_tensor){
            auto scope = std::unique_ptr<Scope>(new Scope());
            auto scope_ptr = scope.get();

            int const_id = AddConstNode(scope_ptr, "output", cpu_tensor);

            // add meta data
            scope->AddOutputName("output");
            return scope->ExportModel();
        }
    }// namespace

    TEST(ConstTest, AnyTest){
        const Tensor* cpu_tensor = Tensor::Random(Tensor::DT_FLOAT, shape,
                dlxnet::TensorProto::ANY);
        auto session = InitSession();
        session->LoadGraph(BuildGraph(cpu_tensor));

        session->Setup({});

        // do computation for the graph
        session->Run();

        ::opengl::TensorNameList output_names({"output"});
        ::opengl::StringList dformats({"ANY"});

        // get cpu outputs from device
        ::opengl::TensorList outputs_cpu;
        session->GetOutputs(output_names, dformats, &outputs_cpu);
        CheckSameTensor(cpu_tensor, outputs_cpu[0]);
    }

    TEST(ConstTest, NHWCTest){
    }

    TEST(ConstTest, NCHWTest){
    }
}//namespace opengl
