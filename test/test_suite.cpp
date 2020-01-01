#include "test_suite.h"

TestSuite* TestSuite::gInstance=nullptr;

TestSuite* TestSuite::get(){
    if(gInstance==nullptr){
        gInstance = new TestSuite;
    }
    return gInstance;
}

void TestSuite::add(std::shared_ptr<TestCase>test_case, std::string name){
    auto suite = TestSuite::get();
    suite->mTestCases.emplace_back(test_case);
}


void TestSuite::run(const std::string name){
    auto suite = TestSuite::get();
    auto test_cases = suite->mTestCases;
    std::vector<std::string> wrongs;
    std::cout<<"Num of Total Test Cases: "<<test_cases.size()<<std::endl;
    bool all = false;
    if(name.empty()){
        all=true;
    }
    for(uint64_t i=0;i<test_cases.size();i++){
        auto& test_case = test_cases[i];
        if(!all and test_case->name.find(name)==0){
            continue;
        }
        auto res = test_case->run();
        if(!res){
            wrongs.emplace_back(test_case->name);
        }
    }

    if(wrongs.empty()){
        std::cout<<"all pass! "<<std::endl;
    }
    for(auto& str:wrongs){
        std::cout<<"Error: " <<str<<std::endl;
    }
}


void TestSuite::runAll(){
    TestSuite::run("");
}
