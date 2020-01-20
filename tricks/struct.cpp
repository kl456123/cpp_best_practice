#include <iostream>
#include <string>

struct Instance{
    int a=10;
    int b[5];
    int c[2][2];
};


int main(){
    Instance a;
    a.b[0] = 10;
    a.b[1] = 1;
    a.c[0][0] = 10;
    a.c[1][0] = 2;
    Instance b;
    b = a;
    std::cout<<b.b[0]<<std::endl;
    std::cout<<b.c[1][0]<<std::endl;
}
