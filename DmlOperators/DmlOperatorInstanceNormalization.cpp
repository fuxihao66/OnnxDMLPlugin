// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorInstanceNormalization
{
public:
    DmlOperatorInstanceNormalization(const std::map<std::string, dml::Expression>& expressionMap, 
                                     const Op& node, dml::Graph& graph, unsigned int opsetVersion)
    {
        m_input = expressionMap[node.inputNames[0]];
        m_weight = expressionMap[node.inputNames[1]];
        m_bias = expressionMap[node.inputNames[2]];
        
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
        
        
        // operatorDesc.FusedActivation = fusedActivation ? &fusedActivationDmlDesc : nullptr;

    }
    dml::Expression Create(){

        return dml::MeanVarianceNormalization(m_input, 
                                              dml::Reinterpret(m_weight, ), // reshape tensor from 1d to 4d
                                              dml::Reinterpret(m_bias, ),
                                              axes, );
    }
private:
    dml::Expression m_input;
    dml::Expression m_weight;
    dml::Expression m_bias;
    std::vector<uint32_t> axes;
    bool normalizeVariance;
    float epsilon;

};

DML_OP_DEFINE_CREATION_FUNCTION(InstanceNormalization, DmlOperatorInstanceNormalization);
// DML_OP_DEFINE_CREATION_FUNCTION(FusedInstanceNormalization, DmlOperatorInstanceNormalization);

} // namespace Dml
