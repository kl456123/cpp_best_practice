#include <iostream>


enum DataType{
    DT_INT=0,
    DT_FLOAT=1,
    DT_DOUBLE=2
};

int main(){
    DataType data_type=DataType::DT_INT;
    uint32_t a = static_cast<uint32_t>(data_type);
    std::cout<<a<<std::endl;
    return 0;
}
