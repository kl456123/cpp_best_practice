#include <iostream>
#include <string>
#include <absl/strings/str_cat.h>
#include <absl/strings/string_view.h>


using std::string;
using StringPiece = absl::string_view;


int main(){
    string s = "agasdg";
    StringPiece s_v = s;

    std::cout<<s_v<<std::endl;
    string s_t(std::move(s));
    std::cout<<s_v<<std::endl;
    return 0;
}
