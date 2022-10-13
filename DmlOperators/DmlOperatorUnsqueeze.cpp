namespace Dml
{
class DmlOperatorUnsqueeze
{
public:
    DmlOperatorUnsqueeze(const std::map<std::string, dml::Expression>& expressionMap, const Op& node, dml::Graph& graph)
    {
        if (node.inputNames.size() != 1)
            throw std::exception("Unsqueeze parameter number must be 1!");
        auto & inputName = node.inputNames[0];
        if (expressionMap.count(inputName) == 0){
            throw std::exception("Dependency does not meet! Please check topological sorting!");
        }
        m_input = expressionMap[inputName];

        valueDataType = m_input.dataType;

        auto & size = inputTensor.sizes;

        // get unsqueeze axis from attribute 
        uint32_t axis = ;
        outputSizes = ;
    }

    dml::Expression Create(){
        // TODO: use identity to copy first?
        return dml::Reinterpret(m_input,
                                valueDataType,
                                outputSizes,
                                nullopt)
    }
private:
    TensorDimensions outputSizes,
    DML_TENSOR_DATA_TYPE valueDataType;
    dml::Expression m_input;
    
};

DML_OP_DEFINE_CREATION_FUNCTION(Unsqueeze, DmlOperatorUnsqueeze);

} // namespace Dml