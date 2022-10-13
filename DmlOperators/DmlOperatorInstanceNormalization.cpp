// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorInstanceNormalization
{
public:
    DmlOperatorInstanceNormalization(const std::map<std::string, dml::Expression>& expressionMap, 
                                     const Op& node, dml::Graph& graph)
    {
        m_input = expressionMap[node.inputNames[0]];
        m_weight = expressionMap[node.inputNames[1]];
        m_bias = expressionMap[node.inputNames[2]];
        // "Instance" normalization is really spatial normalization,
        // where the spatial channels are reduced and normalized, while
        // batch and channel remain independent. So pass a list of axes
        // just beyond the leading batch and channel dimensions (starting
        // at axis 2 up to the last spatial dimension).
        const uint32_t inputDimensionCount = m_inputTensorDescs.front().GetDimensionCount();
        std::vector<uint32_t> axes(inputDimensionCount - 2);
        std::iota(axes.begin(), axes.end(), 2u);

        const float epsilon = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Epsilon, DefaultEpsilon);
        const std::optional<ActivationOperatorDesc> fusedActivation = FusionHelpers::TryGetFusedActivationDesc(kernelCreationContext);
        DML_OPERATOR_DESC fusedActivationDmlDesc = fusedActivation ? fusedActivation->GetDmlDesc() : DML_OPERATOR_DESC();

        // Shift IN_SCALE and IN_BIAS input tensor descs {1, C, 1, 1} out of 1D tensors.
        Shift1DInputsTensorDesc(kernelCreationContext, IN_SCALE, 2, inputDimensionCount);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = &inputDescs[0];
        operatorDesc.ScaleTensor = &inputDescs[1];
        operatorDesc.BiasTensor = &inputDescs[2];
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.Axes = axes.data();
        operatorDesc.AxisCount = static_cast<uint32_t>(axes.size());
        operatorDesc.NormalizeVariance = true;
        operatorDesc.Epsilon = epsilon;
        // operatorDesc.FusedActivation = fusedActivation ? &fusedActivationDmlDesc : nullptr;

    }
    dml::Expression Create(){

        return dml::MeanVarianceNormalization(m_input, 
                                              dml::Reinterpret(m_weight, ), // reshape tensor from 1d to 4d
                                              dml::Reinterpret(m_bias, ));
    }
private:
    dml::Expression m_input;
    dml::Expression m_weight;
    dml::Expression m_bias;
};

DML_OP_DEFINE_CREATION_FUNCTION(InstanceNormalization, DmlOperatorInstanceNormalization);
// DML_OP_DEFINE_CREATION_FUNCTION(FusedInstanceNormalization, DmlOperatorInstanceNormalization);

} // namespace Dml
