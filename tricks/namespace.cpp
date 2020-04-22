#include <iostream>


// it is disabled by default
namespace logging{
    int a = 10;
}

// auto using namespace
namespace {
    int b=10;
}


namespace testing{
    namespace{
        bool flag_ = false;
    }//namespace
    class Controler{
        public:
            static void set_flag(){
                flag_ = true;
            }
            static bool flag(){
                return flag_;
            }
            static void print_flag(){
                std::cout<<flag_<<std::endl;
            }

    };
}//namespace testing


int main(){
    using testing::Controler;
    Controler::print_flag();
    Controler::set_flag();
    Controler::print_flag();
    return 0;
}
