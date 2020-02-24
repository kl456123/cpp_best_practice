#include <iostream>

using namespace std;


class Base{
    public:
        Base()=default;
        virtual ~Base()=default;
        virtual void Print(){
            cout<<a<<endl;
        }
    protected:
        int a=0;
};

class Derived : public Base{
    public:
        void Print(){
            cout<<a<<endl;
        }
    // protected:
        // int a=3;
};


int main(){
    auto derived = Derived();
    derived.Print();
    return 0;
}
