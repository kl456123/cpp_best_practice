#include <iostream>
using namespace std;

// recurrent
// non-var arg func
template<typename T>
void print(const T& t){
    cout<<t<<endl;
}


// var arg func
template<typename T, typename ...Args>
void print(const T&t, const Args&... rest){
    cout<<t<<endl;
    print(rest...);
}


int main(){
    print(40,10, 20, 30);
    print("agdes", "dgag", "dgasg");
    print("dgasdg", "asdg", 40);
    return 0;
}
