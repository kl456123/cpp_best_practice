#ifndef CORE_LOGGING_H_
#define CORE_LOGGING_H_
#include <sstream>
#include <limits>
#include "stream_executor/utils/macros.h"

namespace logging{
    // log levels
    const int INFO=0;
    const int WARNING=1;
    const int ERROR=2;
    const int FATAL=3;
    const int NUM_SEVERITIES=4;

    // log message used like stream
    class LogMessage: public std::basic_ostringstream<char>{
        public:
            LogMessage(const char* fname, int line, int severity);
            ~LogMessage() override;

            LogMessage& AtLocation(const char* fname, int line);

            static int64_t MinLogLevel();
        protected:
            // call it(print to stream) when it is deconstructed
            void GenerateLogMessage();

        private:
            const char* fname_;
            int line_;
            int severity_;
    };

    // log fatal is different from log others
    class LogMessageFatal: public LogMessage{
        public:
            LogMessageFatal(const char* file, int line);
            ~LogMessageFatal() override;
    };

    // container for a string, can be converted to bool type
    struct CheckOpString{
        CheckOpString(std::string* str):str_(str){}
        operator bool()const {return PREDICT_TRUE(str_!=nullptr);}
        std::string* str_;
    };

    template<typename T>
        inline void MakeCheckOpValueString(std::ostream* os, const T& v){
            (*os)<<v;
        }

    template <>
        void MakeCheckOpValueString(std::ostream* os, const char& v);

    template <>
        void MakeCheckOpValueString(std::ostream* os, const signed char& v);

    template <>
        void MakeCheckOpValueString(std::ostream* os, const unsigned char& v);

    template<typename T1, typename T2>
        std::string* MakeCheckOpString(const T1& v1, const T2& v2, const char* exprtext);

    // error message builder
    class CheckOpMessageBuilder{
        public:
            explicit CheckOpMessageBuilder(const char* exprtext);
            ~CheckOpMessageBuilder();

            std::ostream* ForVar1(){return stream_;}
            std::ostream* ForVar2();

            // insert ")"
            std::string* NewString();
        private:
            std::ostringstream* stream_;
    };

    template<typename T1, typename T2>
        std::string* MakeCheckOpString(const T1& v1, const T2& v2, const char* exprtext){
            CheckOpMessageBuilder comb(exprtext);
            MakeCheckOpValueString(comb.ForVar1(), v2);
            MakeCheckOpValueString(comb.ForVar2(), v2);
            return comb.NewString();
        }

    template<typename T>
        T&& CheckNotNull(const char* file, int line, const char* exprtext, T&& t){
            if(t==nullptr){
                LogMessageFatal(file, line)<<std::string(exprtext);
            }
            // type resevered
            return std::forward<T>(t);
        }
}// namespace logging


// some macros
//
#define _DLCL_LOG_INFO logging::LogMessage(__FILE__, __LINE__, logging::INFO)
#define _DLCL_LOG_WARNING logging::LogMessage(__FILE__, __LINE__, logging::WARNING)
#define _DLCL_LOG_ERROR logging::LogMessage(__FILE__, __LINE__, logging::ERROR)
#define _DLCL_LOG_FATAL logging::LogMessageFatal(__FILE__, __LINE__)


// logging and checker
#define LOG(severity) _DLCL_LOG_##severity

#define CHECK(cond)                                     \
    if(PREDICT_FALSE(!(cond)))                          \
    LOG(FATAL)<<"Check failed: " #cond "  "

#define DEFINE_CHECK_OP_IMPL(name, op)                                          \
    template <typename T1, typename T2>                                           \
    inline std::string* name##Impl(const T1& v1, const T2& v2,                    \
            const char* exprtext) {                             \
        if (PREDICT_TRUE(v1 op v2))                                                 \
        return NULL;                                                              \
        else                                                                        \
        return logging::MakeCheckOpString(v1, v2, exprtext);                      \
    }                                                                             \
    inline std::string* name##Impl(int v1, int v2, const char* exprtext) {        \
        return name##Impl<int, int>(v1, v2, exprtext);                              \
    }                                                                             \
    inline std::string* name##Impl(const size_t v1, const int v2,                 \
            const char* exprtext) {                             \
        if (PREDICT_FALSE(v2 < 0)) {                                                \
            return logging::MakeCheckOpString(v1, v2, exprtext);                      \
        }                                                                           \
        return name##Impl<size_t, size_t>(v1, v2, exprtext);                        \
    }                                                                             \
    inline std::string* name##Impl(const int v1, const size_t v2,                 \
            const char* exprtext) {                             \
        if (PREDICT_FALSE(v2 >= std::numeric_limits<int>::max())) {                 \
            return logging::MakeCheckOpString(v1, v2, exprtext);                      \
        }                                                                           \
        const size_t uval = (size_t)((unsigned)v2);                                 \
        return name##Impl<size_t, size_t>(v1, uval, exprtext);                      \
    }

// define some functions using macro
    DEFINE_CHECK_OP_IMPL(Check_EQ, ==)
    DEFINE_CHECK_OP_IMPL(Check_NE, !=)
    DEFINE_CHECK_OP_IMPL(Check_LE, <=)
    DEFINE_CHECK_OP_IMPL(Check_GE, >=)
    DEFINE_CHECK_OP_IMPL(Check_GT, >)
DEFINE_CHECK_OP_IMPL(Check_LT, <)

    // call function using single macro
#define CHECK_OP(name, op, val1, val2)                      \
        while(logging::CheckOpString _result = name##Impl(           \
                    val1, val2, #val1 " " #op " " #val2))     \
    logging::LogMessageFatal(__FILE__, __LINE__)<<*(_result.str_)

    // it can print var1 and var2
#define CHECK_EQ(val1, val2)    CHECK_OP(Check_EQ, ==, val1, val2)
#define CHECK_NE(val1, val2)    CHECK_OP(Check_NE, !=, val1, val2)
#define CHECK_LE(val1, val2)    CHECK_OP(Check_LE, <=, val1, val2)
#define CHECK_GE(val1, val2)    CHECK_OP(Check_GE, >=, val1, val2)
#define CHECK_LT(val1, val2)    CHECK_OP(Check_LT, <, val1, val2)
#define CHECK_GT(val1, val2)    CHECK_OP(Check_GT, >, val1, val2)
    // no null
#define CHECK_NOTNULL(val)                                  \
        CheckNotNull(__FILE__, __LINE__, "'" #val "' Must be non NULL", (val))



#endif
