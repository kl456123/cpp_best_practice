#include "test_suite.h"
#include "core/logging.h"


class LoggingTestCase: public TestCase{
    public:
        virtual bool run(){
            LOG(INFO)<<"adsga";
            LOG(ERROR)<<"error";
            //LOG(FATAL)<<"fatal";
            return true;
        }
};


TestSuiteRegister(LoggingTestCase, "LoggingTestCase");
