#include <iostream>


// void print(int& i){
// std::cout<<i<<std::endl;
// }

// void print(int&& i){
// std::cout<<std::forward<int>(i)<<std::endl;
// }

class Base{
    public:
        void print(){
            std::cout<<"in base"<<std::endl;
        }
};

void print(int&& i){
    std::cout<<i<<std::endl;
}

void print_class(Base&& base_class){
    base_class.print();
}

int get(){
    return 10;
}

Base get_class(){
    return Base();
}



int main(){
    print(get());
    print(10);
    print_class(get_class());
}
