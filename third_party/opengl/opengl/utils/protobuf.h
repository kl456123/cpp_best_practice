#ifndef OEPNGL_UTILS_PROTOBUF_H_
#define OEPNGL_UTILS_PROTOBUF_H_
/**
 * This file contains Utility functions about protobuf.
 * like Decode and Encode protos
 *
 */
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

namespace opengl{
    bool ReadProtoFromBinary(const char* file_path, google::protobuf::Message* message);
}//namespace opengl


#endif

