#include <iostream>
#include <string>
#include "backend.h"

using namespace std;


int main(){
    string name("CPU");
    Backend* backend  = extract_backend(name);
    if(backend==nullptr){
        cout<<"no cpu backend found"<<endl;
    }else{
        cout<<backend<<endl;
    }
    return 0;
}
