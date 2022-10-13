
// force using fp16 and int/uint 32
inline DML_TENSOR_DATA_TYPE OnnxTensorType2DmlTensorType(const TensorType type) {
    switch(type){
    case UINT64:
        return UINT32;
    case INT64:
        return INT32;
    case FLOAT:
    case DOUBLE:
        return FLOAT16;
    default:
        return type;
    }
}