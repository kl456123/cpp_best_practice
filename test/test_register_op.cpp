#include "test_suite.h"
#include "core/op.h"
#include "core/common_shape_fns.h"



class RegisterOpTestCase: public TestCase{
    public:
        virtual bool run(){
            // firstly register op
            REGISTER_OP("conv")
                .Attr("kernel_size: 3")
                .Input("input")
                .Output("output")
                .SetShapeFn(shape_inference::UnknownShape);

            // then look up it from registry
            const OpRegistrationData* op_reg_data=nullptr;
            if(OpRegistry::Global()->LookUp("conv", &op_reg_data)){
                return true;
            }else{
                return false;
            }
        }
};


// register
TestSuiteRegister(RegisterOpTestCase, "RegisterOpTestCase");
