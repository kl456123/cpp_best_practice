#include <iostream>

using namespace std;


enum DataType{
    DT_FLOAT,
    DT_INT,
    DT_DOUBLE
};

// how to map DataType to Enum and Enum to DataType
template <typename T>
struct IsValidateType{
    static constexpr bool value=false;
};


template<typename T>
struct DataTypeToEnum{
    static_assert(IsValidateType<T>::value, "Specified Data Type is not supported!");
    // static void Print(){
        // cout<<IsValidateType<T>::value<<endl;
    // }
};

template<DataType VALUE>
struct EnumToDataType{};

#define MATCH_TYPE_AND_ENUM(TYPE, ENUM)         \
    template<>                                  \
struct DataTypeToEnum<TYPE>{                \
    static DataType v(){return ENUM;}       \
    static constexpr DataType value=ENUM;             \
};                                          \
template<>                                      \
struct IsValidateType<TYPE>{                    \
    static constexpr bool value=true;           \
};                                              \
template<>                                      \
struct EnumToDataType<ENUM>{                    \
    typedef TYPE Type;                          \
}

// template<>
// struct EnumToDataType<DT_INT>{
    // static constexpr DataType value = DT_INT;
// };


MATCH_TYPE_AND_ENUM(float, DT_FLOAT);


int main(){
    EnumToDataType<DT_FLOAT>::Type a=1.0;
    // DataTypeToEnum<int>::value;
    // cout<<IsValidateType<double>::value<<endl;
    cout<<a<<endl;

}
