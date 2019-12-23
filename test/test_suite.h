#ifndef TEST_SUITE_H__
#define TEST_SUITE_H__
#include <iostream>
#include <string>
#include <vector>

class TestCase{
    friend class TestSuite;
    virtual bool run();

    private:
    std::string name;
};


class TestSuite{
    public:
        static TestSuite* get();
        // no need to instanize
        static void run(const std::string name);

        static void runAll();

        void add(TestCase*, std::string);

    private:
        static TestSuite* gInstance;
        std::vector<TestCase> mTestCases;
};




#endif
