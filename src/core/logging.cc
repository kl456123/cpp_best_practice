// #include "core/logging.h"
// #include "core/platform/env_time.h"
// #include <stdlib.h>

// using namespace logging;

// int64_t MinVLogLevelFromEnv(){
    // //(TODO) get log level from env
    // return INFO;
// }




// LogMessage::LogMessage(const char* fname, int line, int severity)
    // :fname_(fname), line_(line), severity_(severity){
    // }

// int64_t LogMessage::MinLogLevel(){
    // static int64_t min_vlog_level = MinVLogLevelFromEnv();
    // return min_vlog_level;
// }

// LogMessage& LogMessage::AtLocation(const char* fname, int line){
    // fname_=fname;
    // line_=line;
    // return *this;
// }


// LogMessage::~LogMessage(){
    // static int64_t min_log_level = MinLogLevel();
    // if(severity_>=min_log_level){
        // GenerateLogMessage();
    // }
// }


// // log fname , line and level with time
// void LogMessage::GenerateLogMessage(){
    // // get time
    // uint64_t now_micros = EnvTime::NowMicros();
    // time_t now_seconds = static_cast<time_t>(now_micros / 1000000);
    // int32_t micros_remainder = static_cast<int32_t>(now_micros % 1000000);
    // const size_t time_buffer_size = 30;
    // char time_buffer[time_buffer_size];
    // strftime(time_buffer, time_buffer_size, "%Y-%m-%d %H:%M:%S",
            // localtime(&now_seconds));
    // const size_t tid_buffer_size = 10;
    // char tid_buffer[tid_buffer_size] = "";
    // fprintf(stderr, "%s.%06d: %c%s %s:%d] %s\n", time_buffer, micros_remainder,
            // "IWEF"[severity_], tid_buffer, fname_, line_, str().c_str());
// }

// LogMessageFatal::LogMessageFatal(const char* file, int line):LogMessage(file, line, FATAL){
// }

// LogMessageFatal::~LogMessageFatal(){
    // GenerateLogMessage();
    // abort();
// }


// CheckOpMessageBuilder::CheckOpMessageBuilder(const char* exprtext)
    // :stream_(new std::ostringstream){
        // // insert the header of failed error message
    // *stream_<<"Check failed: "<<exprtext<<"(";
// }


// CheckOpMessageBuilder::~CheckOpMessageBuilder(){
    // delete stream_;
// }

// std::ostream* CheckOpMessageBuilder::ForVar2(){
    // *stream_<<" vs. ";
    // return stream_;
// }

// std::string* CheckOpMessageBuilder::NewString(){
    // *stream_<<")";
    // return new std::string(stream_->str());
// }
