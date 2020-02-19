#include "session/utils/strcat.h"

using namespace string_utils;


std::string str_join(std::vector<std::string> strs, std::string delimeter){
        std::string res("");
        for(auto& str: strs){
            res = res+str;
        }
        return res;
    }

