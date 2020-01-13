#ifndef TEST_SUITE_H__
#define TEST_SUITE_H__
#include <iostream>
#include <string>
#include <vector>
#include <memory>

class TestCase{
    friend class TestSuite;
    public:
    virtual bool run()=0;
    void set_name(std::string name){
        name_ = std::move(name);
    }

    private:
    std::string name_;
};


class TestSuite{
    public:
        static TestSuite* get();
        // no need to instanize
        static void run(const std::string name);

        static void runAll();

        void add(std::shared_ptr<TestCase>, std::string);

        // virtual ~TestSuite(){};

    private:
        static TestSuite* gInstance;
        std::vector<std::shared_ptr<TestCase>> mTestCases;
};

// use constructor to register
template<typename Case>
class TestRegister{
    public:
        TestRegister(const std::string name){
            auto test_case = std::make_shared<Case>();
            test_case->set_name(name);
            TestSuite::get()->add(test_case, name);
        }
};

// use macro to simplify it
#define TestSuiteRegister(Case, name) static TestRegister<Case> __r##Case(name)




#endif
