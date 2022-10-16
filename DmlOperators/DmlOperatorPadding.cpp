// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorPadding// : public DmlOperator, public PaddingHelper
{
public:
    DmlOperatorPadding(const std::map<std::string, dml::Expression>& expressionMap, const Op& node, dml::Graph& graph, unsigned int opsetVersion)
    {
        const uint32_t inputCount = node.inputNames.size();
        assert((opsetVersion >= 2 && opsetVersion < 11 && inputCount == 1)
                             || (opsetVersion >= 11 && inputCount >= 2 && inputCount <= 3));

        
        std::vector<char> tempAttri;
        { 
            std::string mode;
            bool hasMode = node.GetAttribute("mode", ONNX_PARSER::AttributeType::STRING, tempAttri);
            if (hasMode){
                mode.resize(tempAttri.size());
                memcpy(mode.data(), tempAttri.data(), tempAttri.size());
            }
            else{
                mode = "constant";
            }

            if (mode == "constant"){
                paddingMode = DML_PADDING_MODE_CONSTANT;
    
            }
            else if (mode == "reflect"){
                paddingMode = DML_PADDING_MODE_REFLECTION;
            }
            else if (mode == "edge"){
                paddingMode = DML_PADDING_MODE_EDGE;
            }
            else{
                assert(false);
            }
        }
        if (opsetVersion >= 16){
            // TODO: not supported yet
            assert(false);
        }
        if (opsetVersion >= 11){
            // get pads and constvalue from initializer

        }
        else if (opsetVersion >= 2){
            bool hasValue = node.GetAttribute("value", ONNX_PARSER::AttributeType::FLOAT, tempAttri);
            if (hasValue){
                memcpy(&paddingValue, tempAttri.data(), tempAttri.size());
            }
            else{
                paddingValue = 0.f;
            }
            std::vector<int> pads;
            bool hasPads = node.GetAttribute("pads", ONNX_PARSER::AttributeType::INTS, tempAttri);
            if (hasPads){
                pads.resize(tempAttri.size() / 4);
                memcpy(pads.data(), tempAttri.data(), tempAttri.size());
            }
            else{
                assert(false);
            }
        }
        else{
            assert(false);
        }

        // // Pad the parameters to respect DML's requirements
        // m_startPadding.insert(
        //     m_startPadding.begin(),
        //     m_inputTensorDescs[0].GetDimensionCount() - gsl::narrow_cast<uint32_t>(m_startPadding.size()),
        //     0);

        // m_endPadding.insert(
        //     m_endPadding.begin(),
        //     m_inputTensorDescs[0].GetDimensionCount() - gsl::narrow_cast<uint32_t>(m_endPadding.size()),
        //     0);

    }

    dml::Expression Create(){
        return dml::Padding(
                            m_input,
                            paddingMode,
                            paddingValue,
                            startPadding,
                            endPadding);
    }
private:
    dml::Expression m_input;
    DML_PADDING_MODE paddingMode;
    float paddingValue;
    std::vector<uint32_t> startPadding;
    std::vector<uint32_t> endPadding;
};


DML_OP_DEFINE_CREATION_FUNCTION(Pad, DmlOperatorPadding);
// DML_OP_DEFINE_CREATION_FUNCTION(Pad7, VersionedKernel<DmlOperatorPadding, 7>);
// DML_OP_DEFINE_CREATION_FUNCTION(Pad11, VersionedKernel<DmlOperatorPadding, 11>);
// DML_OP_DEFINE_CREATION_FUNCTION(Pad13, VersionedKernel<DmlOperatorPadding, 13>);

} // namespace Dml
