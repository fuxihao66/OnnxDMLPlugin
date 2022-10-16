
class DmlOperatorShape 
{
public:
    DmlOperatorShape() = default;
    DmlOperatorShape(const std::map<std::string, dml::Expression>& expressionMap, const Op& node, dml::Graph& graph, unsigned int opsetVersion) 
    {
        if (node.inputNames.size() != 1)
            throw std::exception("Shape parameter number must be 1!");

        valueDataType = TensorType::INT32; // implicitly change UINT64 to UINT32, INT64 to INT32

        std::memcpy(outputSizes.data(), node.outputInfo.shapes, node.outputInfo.GetSize() * sizeof(uint32_t));
        // copy
        memcpy(constOpRawData.Bytes, node.inputInfo[0].shapes, node.inputInfo[0].GetSize() * sizeof(uint32_t));
    }
};