// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorSlice //: public DmlOperator, public SliceHelper
{
public:
    DmlOperatorSlice(const std::map<std::string, dml::Expression>& expressionMap, const Op& node, dml::Graph& graph, unsigned int opsetVersion)
    // :   DmlOperator(kernelInfo),
    //     SliceHelper(kernelInfo, kernelInfo.GetTensorShapeDescription(), opsetVersion)
    {
        const uint32_t inputCount = node.inputInfo.size();
        assert((opsetVersion <  10 && inputCount == 1)
                             || (opsetVersion >= 10 && inputCount >= 3 && inputCount <= 5));

        if (opsetVersion < 10){
            std::vector<int> attri;
            {
                std::vector<char> temp;
                bool hasAxis = node.GetAttribute(axes, ONNX_PARSER::AttributeType::INTS, temp);
                if (!hasAxis){

                }
                else{
                    
                }
            }
        }
        else{

        }


        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_SLICE1_OPERATOR_DESC sliceDesc = {};
        sliceDesc.InputTensor = inputDescs.data();
        sliceDesc.OutputTensor = outputDescs.data();
        sliceDesc.DimensionCount = gsl::narrow_cast<uint32_t>(m_offsets.size());
        sliceDesc.InputWindowOffsets = m_offsets.data();
        sliceDesc.InputWindowSizes = m_sizes.data();
        sliceDesc.InputWindowStrides = m_strides.data();

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_SLICE1, &sliceDesc };
        SetDmlOperatorDesc(opDesc, kernelInfo);
    }

    dml::Expression Create(){

        return dml::Slice(
                            m_input,
                            inputWindowOffsets,
                            inputWindowSizes,
                            inputWindowStrides);
    }
private:
    dml::Expression m_input;
    std::vector<uint32_t> inputWindowOffsets;
    std::vector<uint32_t> inputWindowSizes;
    std::vector<int32_t> inputWindowStrides;
};

DML_OP_DEFINE_CREATION_FUNCTION(Slice,  DmlOperatorSlice );
// DML_OP_DEFINE_CREATION_FUNCTION(Slice7,  VersionedKernel<DmlOperatorSlice, 7> );
// DML_OP_DEFINE_CREATION_FUNCTION(Slice10, VersionedKernel<DmlOperatorSlice, 10>);
// DML_OP_DEFINE_CREATION_FUNCTION(Slice11, VersionedKernel<DmlOperatorSlice, 11>);
// DML_OP_DEFINE_CREATION_FUNCTION(Slice13, VersionedKernel<DmlOperatorSlice, 13>);
} // namespace Dml
