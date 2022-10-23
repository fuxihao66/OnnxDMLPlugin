// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// modified by fuxihao, 10/7/2022


namespace Dml
{

class DmlOperatorConvolution //: public DmlOperator, public ConvolutionHelperBase
{

public:
    using Self = DmlOperatorConvolution;

    DmlOperatorConvolution(
        const std::map<std::string, dml::Expression>& expressionMap, const Op& node, dml::Graph& graph, unsigned int opsetVersion )
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
            // hasBias = false;
            bias = std::nullopt;
        }
        else{
            auto& biasName = node.inputNames[2];
            bias = expressionMap[biasName];
        }
        // attribute
        std::vector<char> tempAttri;
        
        { 
        //auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.
            bool hasAutoPad = node.GetAttribute("auto_pad", ONNX_PARSER::AttributeType::STRING, tempAttri);
            if (hasAutoPad){
                autoPad.resize(tempAttri.size());
                memcpy(autoPad.data(), tempAttri.data(), tempAttri.size());
            }
            else{
                autoPad = "NOTSET";
            }
        }
        
        {
            bool hasGroup = node.GetAttribute("group", ONNX_PARSER::AttributeType::INT, tempAttri);
            if (hasGroup){
                memcpy(&group, tempAttri.data(), tempAttri.size());
            }
            else{
                group = 1;
            }
        }

        auto getIntsAttriAndCopy = [&](const std::string& attriName, std::vector<int>& attriVec){
            bool hasAttri = node.GetAttribute(attriName, ONNX_PARSER::AttributeType::INTS, tempAttri);
            if (hasAttri){
                attriVec.resize(tempAttri.size() / 4);
                memcpy(attriVec.data(), tempAttri.data(), tempAttri.size());
            }
            else{
                assert(false);
            }
        }

        getIntsAttriAndCopy("dilations", dialations);
        // getIntsAttriAndCopy("kernel_shape", kernelShape);// no necessity
        getIntsAttriAndCopy("pads", paddings);
        getIntsAttriAndCopy("strides", strides);
        
        outputPaddings.resize(paddings.size() / 2);
        startPaddings.resize(paddings.size() / 2);
        endPaddings.resize(paddings.size() / 2);

        std::fill(outputPaddings.begin(), outputPaddings.end(), 0);

        int index = 0;
        for (int i = 0; i < paddings.size() / 2; i++){
            startPaddings[i] = paddings[index++];
            endPaddings[i] = paddings[index++];
        }


    }

    dml::Expression Create(){
        return dml::ConvolutionBuilder(m_input, m_weight, m_bias)
                    .Mode()
                    .Direction(DML_CONVOLUTION_DIRECTION_FORWARD) // TODO: Add reverse direction to support transposedconv
                    .Strides(strides)
                    .Dilations(dialations)
                    .StartPadding(startPaddings)
                    .EndPadding(endPaddings)
                    .OutputPadding(outputPaddings)
                    .GroupCount(group)
                    //.FusedActivation(dml::FusedActivation::Relu())
                    .Build();
    }
private:
    // bool hasBias = true;
    std::string autoPad;
    std::vector<int> dialations;
    // std::vector<int> kernelShape;
    std::vector<int> startPaddings;
    std::vector<int> endPaddings;
    std::vector<int> outputPaddings;
    std::vector<int> strides;
    int group;
    dml::Expression m_input;
    dml::Expression m_weight;
    std::optional<dml::Expression> m_bias;

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
