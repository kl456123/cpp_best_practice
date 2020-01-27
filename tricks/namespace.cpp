#include <iostream>


// it is disabled by default
namespace logging{
    int a = 10;
}

// auto using namespace
namespace {
    int b=10;
}


int main(){
    // enable logging namespace
    using namespace logging;
    std::cout<<a<<std::endl;
    // using anonymous namespace by default
    std::cout<<b<<std::endl;
    return 0;
}
