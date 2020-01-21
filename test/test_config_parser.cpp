#include <string>
#include "test_suite.h"
#include "config_parser.h"
#include "config.pb.h"



class ConfigParserTestCase: public TestCase{
    public:
        virtual bool run(){
            auto config_parser = ConfigParser<Config>();
            std::string config_fn = "./assets/default.pbtxt";
            // set config

            auto config_proto = config_parser.config_proto();
            config_proto->mutable_backend_config()->set_backend_type(
                    BackendConfig::BackendType::BackendConfig_BackendType_CPU);
            config_parser.SaveToTxt(config_fn);
            config_parser.LoadFromTxt(config_fn);
            config_parser.Print();
            return true;
        }
};


TestSuiteRegister(ConfigParserTestCase, "ConfigParserTestCase");
