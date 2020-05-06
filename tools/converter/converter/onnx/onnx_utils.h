#ifndef CONVERTER_ONNX_ONNX_UTILS_H_
#define CONVERTER_ONNX_ONNX_UTILS_H_
/*******************************
 * The Utils is to help to parse onnx proto model format
 */
#include "onnx.pb.h"
#include <vector>
#include <string>

void ParseAttrValueToString(const onnx::AttributeProto&,
        std::string* pieces);


#endif
