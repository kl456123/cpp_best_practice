#include <iostream>
using namespace std;


class Base{
    public:
        Base(){
            a_=1;
        }
        void PrintA(){
            std::cout<<"PrintA"<<std::endl;
        }
        virtual void PrintB(){
            std::cout<<"PrintB"<<std::endl;
        }
        static void PrintC(){
            std::cout<<"PrintC"<<std::endl;
        }

        int a_;
};

class Base2{
    public:
        void PrintA(){
            std::cout<<"PrintA"<<std::endl;
        }
        static void PrintC(){
            std::cout<<"PrintC"<<std::endl;
        }
};

class Derived: public Base{
    public:
        void PrintA(){
            // automatic pass this to PrintA
            Base::PrintA();
        }
};

int main(){
    Base* cls=nullptr;
    std::cout<<sizeof(Base2)<<std::endl;
    cls->PrintA();
    cls->PrintC();
    Base::PrintC();

    // cls->PrintB();//error
    Derived d;
    std::cout<<d.a_<<std::endl;
    std::cout<<d.a_<<std::endl;
    return 0;
}
