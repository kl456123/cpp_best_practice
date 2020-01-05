// learn how to use lamba function in c++
// note that T a=b; calls copy constructor
//
//
// equaltive to the following code
// struct unique_name{
// unique_name(x):x_(x){
// }
//  int operator(){
//  return x_;
//  };
//
//
// }
//
//
//
//
#include <iostream>

using namespace std;

namespace lambda{
    void base_usage(){
        cout<<"base_usage"<<endl;
        auto _=[]{cout<<"no arguments "<<endl;};
        _();
    }

    void full_usage(){
        cout<<"full_usage "<<endl;
        auto add=[](int a, int b)->int{return a+b;};
        int sum = add(1,2);
    }

    void lazy_usage(){
        cout<<"lazy_usage "<<endl;
        auto add=[](int a, int b){return a+b;};
        int sum = add(1,2);
    }

    void closure(){
        cout<<"capture"<<endl;
        int x = 0;
        auto add = [x]{
            return x;
        };

        auto add2 = [&x]{
            return x;
        };
        x = 4;
        cout<<"x: add() "<<add()<<endl;
        cout<<"x: add2() "<<add2()<<endl;
    }
}// namespace lambda


int main(){
    lambda::base_usage();
    lambda::full_usage();
    lambda::lazy_usage();
    lambda::closure();
}


