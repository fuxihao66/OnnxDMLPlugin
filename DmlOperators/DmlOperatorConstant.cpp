// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{
// TODO: will be implemented later
// class DmlOperatorConstantOfShape : public DmlOperator, public ConstantOfShapeHelper
// {
// public:
//     using Self = DmlOperatorConstantOfShape;

//     DmlOperatorConstantOfShape(const MLOperatorKernelCreationContext& kernelCreationContext)
//     :   DmlOperator(kernelCreationContext),
//         ConstantOfShapeHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription())
//     {
//         ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 1); // ignored shape tensor
//         ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1); // output tensor

//         std::vector<std::optional<uint32_t>> inputIndices = {}; // The shape tensor is not GPU bound.
//         std::vector<std::optional<uint32_t>> outputIndices = { 0 };
//         Initialize(kernelCreationContext, inputIndices, outputIndices);

//         std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

//         DML_FILL_VALUE_CONSTANT_OPERATOR_DESC operatorDesc = {};
//         operatorDesc.OutputTensor = outputDescs.data();
//         operatorDesc.ValueDataType = this->m_outputTensorDescs.front().GetDmlDataType();
//         // operatorDesc.Value already zeroed.

//         // Read the tensor attribute for the output fill pattern.
//         if (kernelCreationContext.HasAttribute(AttrName::Value, MLOperatorAttributeTypeTensor))
//         {
//             ComPtr<IMLOperatorKernelCreationContext> kernelCreationContextInterface = kernelCreationContext.GetInterface();
//             ComPtr<IMLOperatorAttributes1> attributes;
//             ComPtr<IMLOperatorTensor> valueTensor;

//             // Get the extended attributes to be able to access the constant tensor.
//             ORT_THROW_IF_FAILED(kernelCreationContextInterface.As(&attributes));
//             ORT_THROW_IF_FAILED(attributes->GetTensorAttribute(AttrName::Value, &valueTensor));
//             MLOperatorTensor wrappedValueTensor(valueTensor.Get());

//             // Read the raw bytes from the tensor, agnostic to data type, which becomes the GPU fill pattern.
//             ML_CHECK_VALID_ARGUMENT(wrappedValueTensor.IsCpuData());
//             const uint32_t elementCount = wrappedValueTensor.GetTotalElementCount();
//             ML_CHECK_VALID_ARGUMENT(elementCount == 1); // Expect exactly one element.
//             const size_t rawDataByteSize = GetByteSizeFromMlDataType(wrappedValueTensor.GetTensorDataType());
//             const std::byte* rawData = static_cast<const std::byte*>(valueTensor->GetData());

//             memcpy(operatorDesc.Value.Bytes, rawData, std::min(rawDataByteSize, sizeof(operatorDesc.Value.Bytes)));
//         }
//         // Else valueBytes is empty, and the default fill pattern is 0.


//         DML_OPERATOR_DESC opDesc = { DML_OPERATOR_FILL_VALUE_CONSTANT, &operatorDesc };
//         SetDmlOperatorDesc(opDesc, kernelCreationContext);
//     }

// private:
//     std::vector<std::byte> valueBytes;
// };

class DmlOperatorConstant// : public DmlOperator, public ConstantOfShapeHelper
{
public:
    DmlOperatorConstant() = default;
    DmlOperatorConstant(const std::map<std::string, dml::Expression>& expressionMap, const Op& node, dml::Graph& graph)
    {
        // get raw data from attribute
        outputSizes = ;
        valueDataType = node.outputTensorType;
        // copy
        memcpy(constOpRawData.Bytes, rawData, std::min(rawDataByteSize, sizeof(operatorDesc.Value.Bytes)));
    }

    dml::Expression Create(){
        retrun dml::FillValueConstant(
                            graph,
                            outputSizes,
                            valueDataType,
                            constOpRawData);
    }

private:
    DML_SCALAR_UNION constOpRawData;
    DML_TENSOR_DATA_TYPE valueDataType;
    TensorDimensions outputSizes,

    // std::vector<std::byte> valueBytes;
};

class DmlOperatorShape : public DmlOperatorConstant
{
public:
    DmlOperatorShape() = default;
    DmlOperatorShape(const std::map<std::string, dml::Expression>& expressionMap, const Op& node, dml::Graph& graph) 
        : DmlOperatorConstant()
    {
        if (node.inputNames.size() != 1)
            throw std::exception("Shape parameter number must be 1!");

        valueDataType = TensorType::INT32; // implicitly change UINT64 to UINT32, INT64 to INT32

        std::memcpy(outputSizes.data(), node.outputInfo.shapes, node.outputInfo.GetSize() * sizeof(uint32_t));
        // copy
        memcpy(constOpRawData.Bytes, node.inputInfo[0].shapes, node.inputInfo[0].GetSize() * sizeof(uint32_t));
    }
};

// DML_OP_DEFINE_CREATION_FUNCTION(ConstantOfShape, DmlOperatorConstantOfShape);
DML_OP_DEFINE_CREATION_FUNCTION(Constant, DmlOperatorConstant);
DML_OP_DEFINE_CREATION_FUNCTION(Shape, DmlOperatorShape);

} // namespace Dml
