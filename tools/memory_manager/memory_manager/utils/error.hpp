#ifndef CORE_ERROR_H_
#define CORE_ERROR_H_
#include <string>


// error code
//
enum class ErrorCode{
    OK,
    RUNTIME_ERROR,
    UNSUPPORTED_EXTENSION_USE
};


// status class(enum error code and msg)
class Status{
    public:
        // default initialization
        Status():code_(ErrorCode::OK), error_description_(""){
        }

        Status(ErrorCode error_status, std::string error_description="")
            :code_(error_status),error_description_(error_description){
            }

        // override bool ()
        explicit operator bool() const {
            return code_==ErrorCode::OK;
        }

        bool ok(){
            return bool(*this);
        }

        static Status OK(){
            return Status();
        }

        // get mem
        ErrorCode code() const{
            return code_;
        }

        std::string error_description()const {
            return error_description_;
        }

        void throw_if_error(){
            if(!bool(*this)){
                internal_throw_on_error();
            }
        }

    private:
        // private func
        [[noreturn]] void internal_throw_on_error()const ;
    private:
        // enum and strings
        ErrorCode code_;
        std::string  error_description_;
};

// func used to create status
Status create_error(ErrorCode error_status, std::string error_description);


// include where the error happened(lines, func and files)
Status create_error_msg(ErrorCode error_status,
        const char* func, const char* file, int line, const char* msg);

// noreturn used for blocking program here(or terminated)
[[noreturn]] void throw_error(Status error);

// define macro used for assert and terminate program if error happened

#define THROW(x) throw(x)


// some tricks in macro
#define THROW_ERROR_ON(func, file, line, msg)                                              \
    do                                                                                  \
    {                                                                                   \
        throw_error(create_error_msg(ErrorCode::RUNTIME_ERROR, func, file, line, msg)); \
    }while(false)

#define THROW_ERROR(msg) THROW_ERROR_ON(__func__, __FILE__, __LINE__, msg)
#endif
