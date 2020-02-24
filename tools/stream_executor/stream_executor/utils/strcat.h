#ifndef SESSION_UTILS_STRCAT_H_
#define SESSION_UTILS_STRCAT_H_
#include <string>
#include <vector>

namespace string_utils{
    template<typename T>
        std::string str_cat(const T& arg){
            return arg;
        }
    template <typename T, typename ...Args>
        std::string str_cat(const T& arg1, const Args&... args){
            return arg1+str_cat(args...);
        }

    std::string str_join(std::vector<std::string> strs, std::string delimeter);
}


#endif
