#include <string>

#include "test_suite.h"
#include "core/op_kernel.h"
#include "core/types.h"



class RegisterKernelTestCase: public TestCase{
    public:
        virtual bool run(){
            class FactOp:public OpKernel{
                public:
                    explicit FactOp(OpKernelConstruction* context)
                        :OpKernel(context){}
                    void Compute(OpKernelContext* context){
                        Tensor* output_tensor = nullptr;
                        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(), &output_tensor));
                        auto output = output_tensor->template scalar<std::string>();
                    }
            };

            // register it
            REGISTER_KERNEL_BUILDER(Name("Fact").Device(DEVICE_CPU), FactOp);
        }
};


// register
TestSuiteRegister(RegisterKernelTestCase, "RegisterKernelTestCase");
