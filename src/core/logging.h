#ifndef CORE_LOGGING_H_
#define CORE_LOGGING_H_
#include <sstream>
#include "core/define.h"

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
    if(PREDICT_FALSE((!cond)))                          \
    LOG(FATAL)<<"Check failed: " #cond "  "

#define CHECK_OP(name, op, val1, val2)


#endif
