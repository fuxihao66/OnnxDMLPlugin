// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//#include "precomp.h"
#include "../OnnxDMLCore/OperatorRegistration.h"

namespace Dml
{

class DmlOperatorCast// : public DmlOperator
{
public:
    //using Self = DmlOperatorCast;

    DmlOperatorCast(
        std::map<std::string, dml::Expression>& expressionMap, 
        const ONNX_PARSER::Op& node, 
        dml::Graph& graph)
    {
        if (node.inputNames.size() != 1)
            throw std::exception("Cast parameter number must be 1!");

        auto& inputName = node.inputNames[0];
        m_input = expressionMap[inputName];
        targetDataType = TensorType2DmlTensorType(node.outputInfo.tensorType);
    }

    dml::Expression Create(){
        return dml::Cast(m_input, targetDataType);
    }
private:
    dml::Expression m_input;
    DML_TENSOR_DATA_TYPE targetDataType;
};

DML_OP_DEFINE_CREATION_FUNCTION(Cast, DmlOperatorCast);
DML_OP_DEFINE_CREATION_FUNCTION(CastLike15, DmlOperatorCast);

} // namespace Dml
