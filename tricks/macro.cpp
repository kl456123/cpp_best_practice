#include <iostream>
#include <stdio.h>
#include <tuple>

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
    return 0;
}
