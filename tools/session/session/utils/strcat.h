#ifndef SESSION_UTILS_STRCAT_H_
#define SESSION_UTILS_STRCAT_H_
#include <string>

namespace string_utils{
    template<typename T>
        std::string str_cat(const T& arg){
            return arg;
        }
    template <typename T, typename ...Args>
        std::string str_cat(const T& arg1, const Args&... args){
            return arg1+str_cat(args...);
        }
}


#endif
