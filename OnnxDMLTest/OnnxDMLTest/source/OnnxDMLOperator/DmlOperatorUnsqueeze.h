#include "../OnnxDMLCore/OperatorRegistration.h"

namespace Dml
{
class DmlOperatorUnsqueeze
{
public:
    DmlOperatorUnsqueeze(std::map<std::string, dml::Expression>& expressionMap, ONNX_PARSER::Op& node, dml::Graph& graph, unsigned int opsetVersion)
    {
        if (node.inputNames.size() != 1)
            throw std::exception("Unsqueeze parameter number must be 1!");
        auto & inputName = node.inputNames[0];
        if (expressionMap.count(inputName) == 0){
            throw std::exception("Dependency does not meet! Please check topological sorting!");
        }
        m_input = expressionMap[inputName];

        dml::TensorDimensions inputShape = m_input.GetOutputDesc().sizes;
        // get unsqueeze axis from attribute 
        std::vector<int> axis;
        {
            std::vector<char> temp;
            bool hasAxis = node.GetAttribute("axes", ONNX_PARSER::AttributeType::INTS, temp);
            if (!hasAxis)
                assert(false);
            axis.resize(temp.size() / 4);
            memcpy(axis.data(), temp.data(), temp.size());
        }
        int origAxisSize = axis.size();

        axis.push_back(origAxisSize + inputShape.size());

        int index = 0;
        for (int i = 0; i < axis[0]; i++){
            outputSizes.push_back(inputShape[index++]);
        }
        for (int i = 0; i < origAxisSize; i++){
            outputSizes.push_back(1);
            for (int i = axis[i]; i < axis[i+1]; i++){
                outputSizes.push_back(inputShape[index++]);
            }
        }
    }

    dml::Expression Create(){
        // TODO: use identity to copy first?
        return dml::Reinterpret(m_input,
                                outputSizes,
                                std::nullopt);
    }
private:
    dml::TensorDimensions outputSizes;
    // DML_TENSOR_DATA_TYPE valueDataType;
    dml::Expression m_input;
    
};

DML_OP_DEFINE_CREATION_FUNCTION(Unsqueeze, DmlOperatorUnsqueeze);

} // namespace Dml