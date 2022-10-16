// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Dml
{

union ActivationOperatorDescUnion
{
    DML_ACTIVATION_IDENTITY_OPERATOR_DESC identity;
    DML_ACTIVATION_ELU_OPERATOR_DESC elu;
    DML_ACTIVATION_CELU_OPERATOR_DESC celu;
    DML_ACTIVATION_HARDMAX_OPERATOR_DESC hardmax;
    DML_ACTIVATION_HARDMAX1_OPERATOR_DESC hardmax1;
    DML_ACTIVATION_HARD_SIGMOID_OPERATOR_DESC hardSigmoid;
    DML_ACTIVATION_LEAKY_RELU_OPERATOR_DESC leakyRelu;
    DML_ACTIVATION_LINEAR_OPERATOR_DESC linear;
    DML_ACTIVATION_LOG_SOFTMAX_OPERATOR_DESC logSoftmax;
    DML_ACTIVATION_LOG_SOFTMAX1_OPERATOR_DESC logSoftmax1;
    DML_ACTIVATION_PARAMETERIZED_RELU_OPERATOR_DESC parameterizedRelu;
    DML_ACTIVATION_PARAMETRIC_SOFTPLUS_OPERATOR_DESC parametricSoftplus;
    DML_ACTIVATION_RELU_OPERATOR_DESC relu;
    DML_ACTIVATION_SCALED_TANH_OPERATOR_DESC scaledTanh;
    DML_ACTIVATION_SCALED_ELU_OPERATOR_DESC scaledElu;
    DML_ACTIVATION_SIGMOID_OPERATOR_DESC sigmoid;
    DML_ACTIVATION_SOFTMAX_OPERATOR_DESC softmax;
    DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC softmax1;
    DML_ACTIVATION_SOFTPLUS_OPERATOR_DESC softplus;
    DML_ACTIVATION_SOFTSIGN_OPERATOR_DESC softsign;
    DML_ACTIVATION_TANH_OPERATOR_DESC tanh;
    DML_ACTIVATION_THRESHOLDED_RELU_OPERATOR_DESC thresholdedRelu;
    DML_ACTIVATION_SHRINK_OPERATOR_DESC shrink;
    DML_ACTIVATION_GELU_OPERATOR_DESC gelu;
};

template<DML_OPERATOR_TYPE operatorType>
class DmlOperatorActivation
{
public:
    DmlOperatorActivation(
        const std::map<std::string, dml::Expression>& expressionMap, const Op& node, dml::Graph& graph
        )
    {
        std::vector<char> attribVal;
        
        switch (operatorType)
        {
        case DML_OPERATOR_ACTIVATION_ELU:
        case DML_OPERATOR_ACTIVATION_CELU:
        {
            bool hasAlpha = node.GetAttribute("alpha", ONNX_PARSER::AttributeType::FLOAT, attribVal);
            if (hasAlpha)
                memcpy(&operatorDesc.elu.Alpha, attribVal.data(), attribVal.size());
            else
                operatorDesc.elu.Alpha = 1.0f;
            break;
        }
        // case DML_OPERATOR_ACTIVATION_SOFTMAX:
        // case DML_OPERATOR_ACTIVATION_LOG_SOFTMAX:
        // case DML_OPERATOR_ACTIVATION_HARDMAX:
        // case DML_OPERATOR_ACTIVATION_SOFTMAX1:
        // case DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1:
        // case DML_OPERATOR_ACTIVATION_HARDMAX1:
        case DML_OPERATOR_ACTIVATION_HARD_SIGMOID:
        {
            bool hasAlpha = node.GetAttribute("alpha", ONNX_PARSER::AttributeType::FLOAT, attribVal);
            if (hasAlpha)
                memcpy(&operatorDesc.hardSigmoid.Alpha, attribVal.data(), attribVal.size());
            else
                operatorDesc.hardSigmoid.Alpha = 0.2f;
            bool hasBeta = node.GetAttribute("beta", ONNX_PARSER::AttributeType::FLOAT, attribVal);
            if (hasBeta)
                memcpy(&operatorDesc.hardSigmoid.Beta, attribVal.data(), attribVal.size());
            else   
                operatorDesc.hardSigmoid.Beta = 0.5f;
            break;
        }
        case DML_OPERATOR_ACTIVATION_LEAKY_RELU:
            bool hasAlpha = node.GetAttribute("alpha", ONNX_PARSER::AttributeType::FLOAT, attribVal);
            if (hasAlpha)
                memcpy(&operatorDesc.leakyRelu.Alpha, attribVal.data(), attribVal.size());
            else
                operatorDesc.leakyRelu.Alpha = 0.01f;
            break;

        // case DML_OPERATOR_ACTIVATION_LINEAR: // TODO: NOT USED
        //     bool hasAlpha = node.GetAttribute("alpha", ONNX_PARSER::AttributeType::FLOAT, attribVal);
        //     if (hasAlpha)
        //         memcpy(&operatorDesc.linear.Alpha, attribVal.data(), attribVal.size());
        //     else
        //         operatorDesc.linear.Alpha = 0.2f;
        //     bool hasBeta = node.GetAttribute("beta", ONNX_PARSER::AttributeType::FLOAT, attribVal);
        //     if (hasBeta)
        //         memcpy(&operatorDesc.linear.Beta, attribVal.data(), attribVal.size());
        //     else   
        //         operatorDesc.linear.Beta = 0.5f;
        //     break;

        // case DML_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS:  // TODO: NOT USED
        //     operatorDesc.parametricSoftplus.Alpha = ;
        //     operatorDesc.parametricSoftplus.Beta  = ;
        //     break;

        case DML_OPERATOR_ACTIVATION_SCALED_ELU:
        {
            bool hasAlpha = node.GetAttribute("alpha", ONNX_PARSER::AttributeType::FLOAT, attribVal);
            if (hasAlpha)
                memcpy(&operatorDesc.scaledElu.Alpha, attribVal.data(), attribVal.size());
            else
                operatorDesc.scaledElu.Alpha = 1.67326f;
            bool hasGamma = node.GetAttribute("gamma", ONNX_PARSER::AttributeType::FLOAT, attribVal);
            if (hasGamma)
                memcpy(&operatorDesc.scaledElu.Gamma, attribVal.data(), attribVal.size());
            else   
                operatorDesc.scaledElu.Gamma = 1.0507f;
            break;
        }

        // case DML_OPERATOR_ACTIVATION_SCALED_TANH: // TODO: NOT USED
        //     operatorDesc.scaledTanh.Alpha = ;
        //     operatorDesc.scaledTanh.Beta  = ;
        //     break;

        case DML_OPERATOR_ACTIVATION_SOFTPLUS:
        
            operatorDesc.softplus.Steepness = 1.0f;
            break;

        case DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU:
            bool hasAlpha = node.GetAttribute("alpha", ONNX_PARSER::AttributeType::FLOAT, attribVal);
            if (hasAlpha)
                memcpy(&operatorDesc.thresholdedRelu.Alpha, attribVal.data(), attribVal.size());
            else
                operatorDesc.thresholdedRelu.Alpha = 1.f;
            break;

        case DML_OPERATOR_ACTIVATION_SHRINK:
            bool hasBias = node.GetAttribute("bias", ONNX_PARSER::AttributeType::FLOAT, attribVal);
            if (hasBias)
                memcpy(&operatorDesc.shrink.Bias, attribVal.data(), attribVal.size());
            else
                operatorDesc.shrink.Bias = 0.f;
            bool hasThreshold = node.GetAttribute("lambd", ONNX_PARSER::AttributeType::FLOAT, attribVal);
            if (hasThreshold)
                memcpy(&operatorDesc.shrink.Threshold, attribVal.data(), attribVal.size());
            else   
                operatorDesc.shrink.Threshold = 0.5f;
            break;

        case DML_OPERATOR_ACTIVATION_IDENTITY:
        case DML_OPERATOR_ACTIVATION_PARAMETERIZED_RELU:
        case DML_OPERATOR_ACTIVATION_RELU:
        case DML_OPERATOR_ACTIVATION_SIGMOID:
        case DML_OPERATOR_ACTIVATION_TANH:
        case DML_OPERATOR_ACTIVATION_SOFTSIGN:
        // case DML_OPERATOR_ACTIVATION_GELU:
            break;

        default:
            assert(false);
            break;
        }

        gsl::span<const uint32_t> outputSizes = m_outputTensorDescs[0].GetSizes();
        std::vector<DML_TENSOR_DESC> inputDescs;
        std::vector<DML_TENSOR_DESC> outputDescs;

        

        m_input = expressionMap[node.inputNames[0]];
        detail::GraphBuilder* builder = m_input.Impl()->GetGraphBuilder();

        TensorDesc inputTensor = m_input.Impl()->GetOutputDesc();
        TensorDesc outputTensor(inputTensor.dataType, inputTensor.sizes, builder->GetTensorPolicy());

        if (operatorType == DML_OPERATOR_ACTIVATION_PARAMETERIZED_RELU)
        {
            *m_slope = expressionMap[node.inputNames[1]];
            TensorDesc slopeTensor = m_slope->Impl()->GetOutputDesc();

            operatorDesc.parameterizedRelu.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
            operatorDesc.parameterizedRelu.SlopeTensor = slopeTensor.AsPtr<DML_TENSOR_DESC>();
            operatorDesc.parameterizedRelu.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();
            
        }
        else // All other activation descrptions are equivalent to Elu in layout.
        {
            operatorDesc.elu.InputTensor = inputTensor.AsPtr<DML_TENSOR_DESC>();
            operatorDesc.elu.OutputTensor = outputTensor.AsPtr<DML_TENSOR_DESC>();

            m_slope = std::nullopt;
        }

    }
    dml::Expression Create(){
        detail::GraphBuilder* builder = m_input.Impl()->GetGraphBuilder(); 
        
        detail::NodeID node;
        if (m_slope){
            detail::NodeOutput* const inputs[] = { m_input.Impl(), m_slope->Impl() }; 
            node = builder->CreateOperatorNode(operatorType, &operatorDesc, inputs);
        }
        else{
            detail::NodeOutput* const inputs[] = { m_input.Impl() }; 
            node = builder->CreateOperatorNode(operatorType, &operatorDesc, inputs);
        }
         
        detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, *operatorDesc.OutputTensor); 
       
        return output;
    }
private:
    dml::Expression m_input;
    std::optional<dml::Expression> m_slope;
    ActivationOperatorDescUnion operatorDesc;
    // DML_OPERATOR_TYPE remappedOperatorType(const DML_OPERATOR_TYPE operatorType) const {
    //     switch (operatorType)
    //     {
    //         case DML_OPERATOR_ACTIVATION_HARDMAX:
    //             return DML_OPERATOR_ACTIVATION_HARDMAX1;
    //         case DML_OPERATOR_ACTIVATION_SOFTMAX:
    //             return DML_OPERATOR_ACTIVATION_SOFTMAX1;
    //         case DML_OPERATOR_ACTIVATION_LOG_SOFTMAX:
    //             return DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1;
    //         default:
    //             return operatorType;
    //     }
    // }
};


DML_OP_DEFINE_CREATION_FUNCTION(Sigmoid,             DmlOperatorActivation<DML_OPERATOR_ACTIVATION_SIGMOID>);
DML_OP_DEFINE_CREATION_FUNCTION(HardSigmoid,         DmlOperatorActivation<DML_OPERATOR_ACTIVATION_HARD_SIGMOID>);
DML_OP_DEFINE_CREATION_FUNCTION(Tanh,                DmlOperatorActivation<DML_OPERATOR_ACTIVATION_TANH>);
// DML_OP_DEFINE_CREATION_FUNCTION(ScaledTanh,          DmlOperatorActivation<DML_OPERATOR_ACTIVATION_SCALED_TANH>);
DML_OP_DEFINE_CREATION_FUNCTION(Relu,                DmlOperatorActivation<DML_OPERATOR_ACTIVATION_RELU>);
DML_OP_DEFINE_CREATION_FUNCTION(Celu,                DmlOperatorActivation<DML_OPERATOR_ACTIVATION_CELU>);
DML_OP_DEFINE_CREATION_FUNCTION(LeakyRelu,           DmlOperatorActivation<DML_OPERATOR_ACTIVATION_LEAKY_RELU>);
DML_OP_DEFINE_CREATION_FUNCTION(PRelu,               DmlOperatorActivation<DML_OPERATOR_ACTIVATION_PARAMETERIZED_RELU>);
DML_OP_DEFINE_CREATION_FUNCTION(ThresholdedRelu,     DmlOperatorActivation<DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU>);
DML_OP_DEFINE_CREATION_FUNCTION(Elu,                 DmlOperatorActivation<DML_OPERATOR_ACTIVATION_ELU>);
DML_OP_DEFINE_CREATION_FUNCTION(Selu,                DmlOperatorActivation<DML_OPERATOR_ACTIVATION_SCALED_ELU>);
DML_OP_DEFINE_CREATION_FUNCTION(Softsign,            DmlOperatorActivation<DML_OPERATOR_ACTIVATION_SOFTSIGN>);
DML_OP_DEFINE_CREATION_FUNCTION(Softplus,            DmlOperatorActivation<DML_OPERATOR_ACTIVATION_SOFTPLUS>);
// DML_OP_DEFINE_CREATION_FUNCTION(ParametricSoftplus,  DmlOperatorActivation<DML_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS>);
DML_OP_DEFINE_CREATION_FUNCTION(Dropout,             DmlOperatorActivation<DML_OPERATOR_ACTIVATION_IDENTITY>);
// DML_OP_DEFINE_CREATION_FUNCTION(Softmax,             DmlOperatorActivation<DML_OPERATOR_ACTIVATION_SOFTMAX>);
// // DML_OP_DEFINE_CREATION_FUNCTION(Softmax13,           DmlOperatorActivation<DML_OPERATOR_ACTIVATION_SOFTMAX1>);
// DML_OP_DEFINE_CREATION_FUNCTION(LogSoftmax,          DmlOperatorActivation<DML_OPERATOR_ACTIVATION_LOG_SOFTMAX>);
// DML_OP_DEFINE_CREATION_FUNCTION(LogSoftmax13,        DmlOperatorActivation<DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1>);
// DML_OP_DEFINE_CREATION_FUNCTION(Hardmax,             DmlOperatorActivation<DML_OPERATOR_ACTIVATION_HARDMAX>);
// DML_OP_DEFINE_CREATION_FUNCTION(Hardmax13,           DmlOperatorActivation<DML_OPERATOR_ACTIVATION_HARDMAX1>);
DML_OP_DEFINE_CREATION_FUNCTION(Shrink,              DmlOperatorActivation<DML_OPERATOR_ACTIVATION_SHRINK>);
// DML_OP_DEFINE_CREATION_FUNCTION(Gelu,                DmlOperatorActivation<DML_OPERATOR_ACTIVATION_GELU>);

} // namespace Dml
