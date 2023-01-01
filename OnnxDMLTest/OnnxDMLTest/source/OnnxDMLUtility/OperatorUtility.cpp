#include "helper/pch.h"
// force using fp16 and int/uint 32
inline DML_TENSOR_DATA_TYPE TensorType2DmlTensorType(const ONNX_PARSER::TensorType type) {
    switch(type){
    case ONNX_PARSER::TensorType::UINT64:
        return DML_TENSOR_DATA_TYPE::DML_TENSOR_DATA_TYPE_UINT32;
    case ONNX_PARSER::TensorType::INT64:
        return DML_TENSOR_DATA_TYPE::DML_TENSOR_DATA_TYPE_INT32;
    case ONNX_PARSER::TensorType::FLOAT:
    case ONNX_PARSER::TensorType::DOUBLE:
        return DML_TENSOR_DATA_TYPE::DML_TENSOR_DATA_TYPE_FLOAT16;
    default:
        return static_cast<DML_TENSOR_DATA_TYPE>(type);
    }
}