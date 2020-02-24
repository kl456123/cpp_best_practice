#include <iostream>
#include <stdio.h>
#include <tuple>
#include <string>

using namespace std;


#define TRACE(x, format) printf(#x " = %" #format "\n", x)
#define TRACE_INT(x) TRACE(x d)
#define TRACE_INT_BY_INDEX(x, i) TRACE(x##i, d)
#define QUATE(x) (#x[0])
#define PRINT(...) printf(__VA_ARGS__)
#define PRINT_MSG()          \
    PRINT("print in file: %s, line: %d\n", __FILE__, __LINE__)
#define PRINT_FUNC()         \
    PRINT("print func: %s\n", __func__)
// ## ignore the comma before va_args when it is empty
#define PRINT_FORMAT(format, ...) printf(format, ##__VA_ARGS__)

// num of arguments
// decltype refers to the type of arg
#define NUMARGS(...)  std::tuple_size<decltype(std::make_tuple(__VA_ARGS__))>::value
#define SUM(...)      TRACE(NUMARGS(__VA_ARGS__), lu)


#define CHECK(stmt)                                                 \
    do{                                                             \
        if(stmt){                                                   \
            std::cout<<std::string(#stmt)<<"true"<<std::endl;       \
        }else{                                                      \
            std::cout<<std::string(#stmt)<<"false"<<std::endl;      \
        }                                                           \
    }while(false)

// Identifier concatenation helper macros.
#define SE_MACRO_CONCAT_INNER(__x)             \
    std::cout<<std::string(#__x)<<std::endl

#define SE_MACRO_CONCAT(__x)                \
        SE_MACRO_CONCAT_INNER(__x);         \
    std::cout<<std::string(#__x)<<std::endl

bool ret_bool(int a){
    if(a==3){
        return false;
    }else{
        return true;
    }
}



int main(){
    int x1=0;
    int x2=2;
    int x3=3;
    TRACE(static_cast<float>(x1), f);
    // PRINT(QUATE(x1));
    if(QUATE(d)=='d'){
        PRINT("success\n");
    }
    TRACE_INT_BY_INDEX(x, 1);
    TRACE_INT_BY_INDEX(x, 2);
    TRACE_INT_BY_INDEX(x, 3);

    PRINT_MSG();
    PRINT_FUNC();
    PRINT("dgagas\n");
    PRINT_FORMAT("dgaa\n");
    SUM("aasdga", "bsga");
    CHECK(ret_bool(3));
    SE_MACRO_CONCAT(ret_bool(3));
    return 0;
}
