// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorPadding// : public DmlOperator, public PaddingHelper
{
public:
    DmlOperatorPadding(std::map<std::string, dml::Expression>& expressionMap, ONNX_PARSER::Op& node, dml::Graph& graph, unsigned int opsetVersion)
    {
        const uint32_t inputCount = node.inputNames.size();
        assert((opsetVersion >= 2 && opsetVersion < 11 && inputCount == 1)
                             || (opsetVersion >= 11 && inputCount >= 2 && inputCount <= 3));
        m_input = expressionMap[node.inputNames[0]];
        Dimensions inputShape = m_input.GetOutputDesc().sizes;
        
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

        std::vector<int> paddings;
        if (opsetVersion >= 11){
            bool hasPads = node.GetAttribute("pads", ONNX_PARSER::AttributeType::TENSOR, tempAttri);
            if (hasPads){
                paddings.resize(tempAttri.size() / 4);
                memcpy(paddings.data(), tempAttri.data(), tempAttri.size());
            }

            // TODO: CHECK INITIALIZER TYPE (scalar ??)
            bool hasConstants = node.GetAttribute("constant_value", ONNX_PARSER::AttributeType::TENSOR, tempAttri);
            if (hasConstants){
                memcpy(&paddingValue, tempAttri.data(), tempAttri.size());
            }
        }
        else if (opsetVersion >= 2){
            bool hasValue = node.GetAttribute("value", ONNX_PARSER::AttributeType::FLOAT, tempAttri);
            if (hasValue){
                memcpy(&paddingValue, tempAttri.data(), tempAttri.size());
            }
            else{
                paddingValue = 0.f;
            }
            bool hasPads = node.GetAttribute("pads", ONNX_PARSER::AttributeType::INTS, tempAttri);
            if (hasPads){
                paddings.resize(tempAttri.size() / 4);
                memcpy(paddings.data(), tempAttri.data(), tempAttri.size());
            }
            else{
                assert(false);
            }
        }
        else{
            assert(false);
        }

        assert(paddings.size() == inputShape.size() * 2);
        // // Pad the parameters to respect DML's requirements
        startPadding.resize(inputShape.size());
        endPadding.resize(inputShape.size());
        int index = 0;
        for (int i = 0; i < inputShape.size(); i++){
            startPadding[i] = paddings[index++];
            endPadding[i] = paddings[index++];
        }
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
