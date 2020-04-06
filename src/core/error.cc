// #include "core/error.hpp"
// #include <array>
// #include <stdexcept>

// Status create_error(ErrorCode error_status, std::string error_description)
// {
    // return Status(error_status, error_description);
// }


// Status create_error_msg(ErrorCode error_code,
        // const char* func, const char* file, int line, const char* msg){
    // // combine all to char*
    // std::array<char, 512> out{0};
    // snprintf(out.data(), out.size(), "in %s %s: %d: %s", func, file, line, msg);
    // return Status(error_code, std::string(out.data()));
// }

// void throw_error(Status error){
    // THROW(std::runtime_error(error.error_description()));
// }

// // used for privacy
// void Status::internal_throw_on_error() const{
    // THROW(std::runtime_error(error_description_));
// }
