#include <iostream>
#include <string>
#include <absl/strings/string_view.h>
#include <absl/strings/strip.h>


typedef absl::string_view StringPiece;



class Scanner{
    public:
        enum CharClass{
            ALL,
            DIGIT,
            LETTER,
            LETTER_DIGIT,
            SPACE
        };
        explicit Scanner(StringPiece source):cur_(source){
            RestartCapture();
        }

        Scanner& One(CharClass clz){
            if (cur_.empty() || !Matches(clz, cur_[0])) {
                return Error();
            }
            cur_.remove_prefix(1);
            return *this;
        }

        Scanner& OneLiteral(StringPiece s){
            if(!absl::ConsumePrefix(&cur_, s)){
                error_ = true;
            }
            return *this;
        }

        Scanner& ZeroOrOneLiteral(StringPiece s){
            absl::ConsumePrefix(&cur_, s);
            return *this;
        }

        Scanner& Many(CharClass clz){
            return One(clz).Any(clz);
        }

        Scanner& Any(CharClass clz){
            while(!cur_.empty() && Matches(clz, cur_[0])){
                cur_.remove_prefix(1);
            }
            return *this;
        }
        // Reset the capture start point.
        //
        // Later, when GetResult is called and if it returns true, the capture
        // returned will start at the position at the time this was called.
        Scanner& RestartCapture() {
            capture_start_ = cur_.data();
            capture_end_ = nullptr;
            return *this;
        }

        // Stop capturing input.
        //
        // Later, when GetResult is called and if it returns true, the capture
        // returned will end at the position at the time this was called.
        Scanner& StopCapture() {
            capture_end_ = cur_.data();
            return *this;
        }

        bool GetResult(StringPiece* remaining=nullptr,
                StringPiece* capture=nullptr){
            if(error_){
                return false;
            }
            if(remaining!=nullptr){
                *remaining = cur_;
            }

            if(capture!=nullptr){
                const char* end =  capture_end_==nullptr? cur_.data():capture_end_  ;
                *capture = StringPiece(capture_start_ , end-capture_start_);
            }

            return true;
        }

        // Shorthand for Any(SPACE).
        Scanner& AnySpace() { return Any(SPACE); }
    private:
        Scanner& Error(){
            error_ = true;
            return *this;
        }

        static bool IsLetter(char ch) {
            return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z');
        }

        static bool IsLowerLetter(char ch) { return ch >= 'a' && ch <= 'z'; }

        static bool IsDigit(char ch) { return ch >= '0' && ch <= '9'; }

        static bool IsSpace(char ch) {
            return (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\v' || ch == '\f' ||
                    ch == '\r');
        }

        static bool Matches(CharClass clz, char ch){
            switch(clz){
                case ALL:
                    return true;
                case LETTER:
                    return IsLetter(ch);
                case DIGIT:
                    return IsDigit(ch);
                case LETTER_DIGIT:
                    return IsLetter(ch) || IsDigit(ch);
                case SPACE:
                    return IsSpace(ch);
            }
            return false;
        }
        StringPiece cur_;
        const char* capture_start_=nullptr;
        const char* capture_end_=nullptr;
        bool error_ = false;
};


int main(){
    std::string s = "dtype: type";
    StringPiece sp = s;
    StringPiece out;

    bool res = Scanner(sp)
        .One(Scanner::LETTER)
        .Any(Scanner::LETTER_DIGIT)
        .StopCapture()
        .AnySpace()
        .OneLiteral(":")
        .AnySpace()
        .GetResult(&sp, &out);
    std::cout<<res<<std::endl;
    std::cout<<"remain: "<<sp<<std::endl;
    std::cout<<"capture: "<<out<<std::endl;
    return 0;
}
