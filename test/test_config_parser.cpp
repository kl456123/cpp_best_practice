#include <string>
#include "test_suite.h"
#include "core/config_parser.h"
#include "core/protos/config.pb.h"



class ConfigParserTestCase: public TestCase{
    public:
        virtual bool run(){
            auto config_parser = ConfigParser<Config>();
            std::string config_fn = "./default.cfg";
            config_parser.SaveToTxt(config_fn);
            config_parser.LoadFromTxt(config_fn);
            config_parser.Print();
            return true;
        }
};


TestSuiteRegister(ConfigParserTestCase, "ConfigParserTestCase");
