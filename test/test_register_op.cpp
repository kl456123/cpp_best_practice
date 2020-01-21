#include "test_suite.h"
#include "core/op.h"
#include "core/common_shape_fns.h"



class RegisterOpTestCase: public TestCase{
    public:
        virtual bool run(){
            REGISTER_OP("conv")
                .Attr("kernel_size: 3")
                .Input("input")
                .Output("output")
                .SetShapeFn(shape_inference::UnknownShape);
            return true;
        }
};


// register
TestSuiteRegister(RegisterOpTestCase, "RegisterOpTestCase");
