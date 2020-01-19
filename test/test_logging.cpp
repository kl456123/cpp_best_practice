#include "test_suite.h"
#include "core/logging.h"


class LoggingTestCase: public TestCase{
    public:
        virtual bool run(){
            LOG(INFO)<<"adsga";
            LOG(ERROR)<<"error";
            int a=10;
            CHECK(a==10);
            //LOG(FATAL)<<"fatal";
            return true;
        }
};


TestSuiteRegister(LoggingTestCase, "LoggingTestCase");
