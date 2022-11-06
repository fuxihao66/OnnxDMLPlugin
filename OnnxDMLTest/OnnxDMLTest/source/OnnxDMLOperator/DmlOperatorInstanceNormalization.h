// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorInstanceNormalization
{
public:
    DmlOperatorInstanceNormalization(std::map<std::string, dml::Expression>& expressionMap, 
                                     ONNX_PARSER::Op& node, dml::Graph& graph, unsigned int opsetVersion)
    {
        m_input = expressionMap[node.inputNames[0]];
        m_weight = expressionMap[node.inputNames[1]];
        m_bias = expressionMap[node.inputNames[2]];
        
        dml::TensorDimensions inputShape = m_input.GetOutputDesc().sizes;
        dml::TensorDimensions weightShape = m_weight.GetOutputDesc().sizes;
        
        {
            std::vector<char> temp;
            bool hasEpsilon = node.GetAttribute("epsilon", ONNX_PARSER::AttributeType::FLOAT, temp);
            if (hasEpsilon){
                memcpy(&epsilon, temp.data(), temp.size());
            }
            else{
                epsilon = 1e-5f;
            }
        }
        
        axes.resize(inputShape.size() - 2);
        std::iota(axes.begin(), axes.end(), 2u); // if input is nxcxhxw, then axes is [2,3]
        
        // operatorDesc.FusedActivation = fusedActivation ? &fusedActivationDmlDesc : nullptr;

    }
    dml::Expression Create(){

        return dml::MeanVarianceNormalization(m_input, 
                                              dml::Reinterpret(m_weight, TensorDimentions{1, weightShape[0], 1, 1}, nullopt), // reshape tensor from 1d to 4d
                                              dml::Reinterpret(m_bias, TensorDimentions{1, weightShape[0], 1, 1}, nullopt),
                                              axes, true, epsilon);
    }
private:
    dml::Expression m_input;
    dml::Expression m_weight;
    dml::Expression m_bias;
    std::vector<uint32_t> axes;
    // bool normalizeVariance;
    float epsilon;

};

DML_OP_DEFINE_CREATION_FUNCTION(InstanceNormalization, DmlOperatorInstanceNormalization);
// DML_OP_DEFINE_CREATION_FUNCTION(FusedInstanceNormalization, DmlOperatorInstanceNormalization);

} // namespace Dml
