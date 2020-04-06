#include <iostream>
#include <atomic>


int main(){
    std::atomic_int_fast32_t ref_(1);
    std::cout<<ref_.fetch_sub(1)<<std::endl;;
    std::cout<<ref_.load()<<std::endl;
    std::cout<<ref_.fetch_add(1)<<std::endl;
    return 0;
}
