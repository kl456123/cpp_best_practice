#include "test_suite.h"
#include "core/op.h"



class RegisterOpTestCase: public TestCase{
    public:
        virtual bool run(){
            REGISTER_OP("conv")
                .Attr("kernel_size", 3)
                .Input("input")
                .Output("output")
                .SetShapeFn()
            return true;
        }
};


// register
TestSuiteRegister(RegisterOpTestCase, "RegisterOpTestCase");
