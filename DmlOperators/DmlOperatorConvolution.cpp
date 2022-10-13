// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// modified by fuxihao, 10/7/2022


namespace Dml
{

class DmlOperatorConvolution //: public DmlOperator, public ConvolutionHelperBase
{
private:
    bool hasBias = true;
    bool autoPad = false;

    dml::Expression m_input;
    dml::Expression m_weight;
    std::optional<dml::Expression> m_bias;

public:
    using Self = DmlOperatorConvolution;

    DmlOperatorConvolution(
        const std::map<std::string, dml::Expression>& expressionMap, const Op& node, dml::Graph& graph
        )
    // :   DmlOperator(kernelInfo),
        // ConvolutionHelperBase(kernelInfo, kernelInfo.GetTensorShapeDescription(), direction == DML_CONVOLUTION_DIRECTION_BACKWARD, hasDynamicPads, 0, 1)
    {
        if (node.inputNames.size() < 2)
            throw std::exception("Convolution parameter number is less than 2!");

        auto& inputName = node.inputNames[0];
        auto& weightName = node.inputNames[1];
        m_input = expressionMap[inputName];
        m_weight = expressionMap[weightName];

        if (node.inputNames.size() == 2){
            hasBias = false;
            bias = std::nullopt;
        }
        else{
            auto& biasName = node.inputNames[2];
            bias = expressionMap[biasName];
        }
        // attribute





        // uint32_t biasIndex = hasDynamicPads ? 3 : 2;
        // bool hasBiasInput = kernelInfo.GetInputCount() > biasIndex;

        // std::vector<std::optional<uint32_t>> kernelInputIndices = { 0, 1, biasIndex };

        // DmlOperator::Initialize(kernelInfo, kernelInputIndices);

        // KernelArgs kernelArgs(m_kernel, NchwSpatialDimensionCount);

        // // Zero the output padding before sending to DirectML. Although it was needed to compute
        // // the output size, we don't want DML to see the values, which should just be ignored.
        // memset(kernelArgs.outputPadding, 0, sizeof(kernelArgs.outputPadding));

        // DML_CONVOLUTION_OPERATOR_DESC convDesc = {};
        // convDesc.InputTensor = &inputDescs[0];
        // convDesc.FilterTensor = &inputDescs[1];
        // convDesc.BiasTensor = hasBiasInput ? &inputDescs[biasIndex] : nullptr;
        // convDesc.OutputTensor = &outputDescs[0];
        // convDesc.Mode = mode;
        // convDesc.Direction = direction;
        // convDesc.DimensionCount = kernelArgs.spatialDimensionCount;
        // convDesc.Strides = kernelArgs.strides;
        // convDesc.Dilations = kernelArgs.dilations;
        // convDesc.StartPadding = kernelArgs.startPadding;
        // convDesc.EndPadding = kernelArgs.endPadding;
        // convDesc.OutputPadding = kernelArgs.outputPadding;
        // convDesc.GroupCount = m_groupCount;
        // convDesc.FusedActivation = fusedActivation ? &fusedActivationDmlDesc : nullptr;

    }

    dml::Expression Create(){
        return dml::ConvolutionBuilder(m_input, m_weight, m_bias)
                    .Mode()
                    .Direction()
                    .Strides()
                    .Dilations()
                    .StartPadding(std::array<uint32_t, 2>{ 1u, 1u })
                    .EndPadding(std::array<uint32_t, 2>{ 1u, 1u })
                    .OutputPadding()
                    .GroupCount()
                    //.FusedActivation(dml::FusedActivation::Relu())
                    .Build();
    }

};

// // A specific type of operation for registration.
// template <DML_CONVOLUTION_MODE Mode, DML_CONVOLUTION_DIRECTION Direction, bool hasDynamicPads = false>
// class DmlOperatorConvolutionTemplate : public DmlOperatorConvolution
// {
// public:
//     DmlOperatorConvolutionTemplate(const MLOperatorKernelCreationContext& kernelInfo)
//     :   DmlOperatorConvolution(kernelInfo, Mode, Direction, hasDynamicPads)
//     {
//     }
// };

DML_OP_DEFINE_CREATION_FUNCTION(Conv,                           DmlOperatorConvolutionTemplate<DML_CONVOLUTION_MODE_CROSS_CORRELATION, DML_CONVOLUTION_DIRECTION_FORWARD>);
// DML_OP_DEFINE_CREATION_FUNCTION(ConvTranspose,                  DmlOperatorConvolutionTemplate<DML_CONVOLUTION_MODE_CROSS_CORRELATION, DML_CONVOLUTION_DIRECTION_BACKWARD>);
// DML_OP_DEFINE_CREATION_FUNCTION(FusedConv,                      DmlOperatorConvolutionTemplate<DML_CONVOLUTION_MODE_CROSS_CORRELATION, DML_CONVOLUTION_DIRECTION_FORWARD>);
// DML_OP_DEFINE_CREATION_FUNCTION(FusedConvTranspose,             DmlOperatorConvolutionTemplate<DML_CONVOLUTION_MODE_CROSS_CORRELATION, DML_CONVOLUTION_DIRECTION_BACKWARD>);
// DML_OP_DEFINE_CREATION_FUNCTION(ConvTransposeWithDynamicPads,   DmlOperatorConvolutionTemplate<DML_CONVOLUTION_MODE_CROSS_CORRELATION, DML_CONVOLUTION_DIRECTION_BACKWARD, true>);

} // namespace Dml
