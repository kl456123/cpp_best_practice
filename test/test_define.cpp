#include "test_suite.h"
#include "core/macros.h"


class DefineTestCase: public TestCase{
    public:
        virtual bool run(){
            int a = 10;
            DLCL_ASSERT(a==10);
            return true;
        }
};


TestSuiteRegister(DefineTestCase, "DefineTestCase");
