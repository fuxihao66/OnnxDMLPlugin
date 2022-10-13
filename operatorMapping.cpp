
// https://github.com/onnx/onnx/blob/main/docs/Operators.md
// onnxruntime\core\providers\dml\DmlExecutionProvider\src\Operators\OperatorRegistration.cpp

// operator that has different parameter with different ir version
#define REG_INFO_VER(operatorName, version) \
    #operatorName, Create##operatorName##version,
#define REG_INFO(operatorName) \
    #operatorName, Create##operatorName,
// operaters that can be mapped to DmlOperatorCopy
#define REG_INFO_COPY(operatorName) \
    #operatorName, CreateCopy,

#define DML_OP_EXTERN_CREATION_FUNCTION(operatorName) CALLBACK Create##operatorName(const std::map<std::string, dml::Expression> &expressionMap, const Op &node, dml::Graph& graph)

DML_OP_EXTERN_CREATION_FUNCTION(Copy); // for Unsqueeze
DML_OP_EXTERN_CREATION_FUNCTION(Conv);
DML_OP_EXTERN_CREATION_FUNCTION(InstanceNormalization);
DML_OP_EXTERN_CREATION_FUNCTION(Relu);
DML_OP_EXTERN_CREATION_FUNCTION(Gather);
DML_OP_EXTERN_CREATION_FUNCTION(Concat);
DML_OP_EXTERN_CREATION_FUNCTION(Slice7);
DML_OP_EXTERN_CREATION_FUNCTION(Slice10);
DML_OP_EXTERN_CREATION_FUNCTION(Slice11);
DML_OP_EXTERN_CREATION_FUNCTION(Slice13);
DML_OP_EXTERN_CREATION_FUNCTION(Pad7);
DML_OP_EXTERN_CREATION_FUNCTION(Pad11);
DML_OP_EXTERN_CREATION_FUNCTION(Pad13);
DML_OP_EXTERN_CREATION_FUNCTION(Add);
DML_OP_EXTERN_CREATION_FUNCTION(Mul);
DML_OP_EXTERN_CREATION_FUNCTION(Div);
DML_OP_EXTERN_CREATION_FUNCTION(Upsample7);
DML_OP_EXTERN_CREATION_FUNCTION(Upsample9);
DML_OP_EXTERN_CREATION_FUNCTION(Upsample10);
DML_OP_EXTERN_CREATION_FUNCTION(Cast);
DML_OP_EXTERN_CREATION_FUNCTION(Floor);
DML_OP_EXTERN_CREATION_FUNCTION(Constant); // not implemented in ORT
DML_OP_EXTERN_CREATION_FUNCTION(Shape);    // not implemented in ORT, use constant to implement shape
// or combine operator?

DML_OP_DEFINE_CREATION_FUNCTION(Concat, DmlOperatorConcat);

#define DML_OP_DEFINE_CREATION_FUNCTION(operatorName, ...)                                                                               \
    \
extern dml::Expression CALLBACK Create##operatorName(const std::map<std::string, dml::Expression> &expressionMap, const Op &node, dml::Graph& graph) \
    \
{                                                                                                                                   \
        using T = __VA_ARGS__;                                                                                                           \
        OperatorGenerator<T>::CreateDmlExpression(expressionMap, node);                                                                  \
    \
}

using CreateFn = dml::Expression(CALLBACK *)(const std::map<std::string, dml::Expression> &expressionMap, const Op &node, dml::Graph& graph);

constexpr static std::unordered_map<std::string, CreateFn> g_operatorRegistrationMap =
    {
        {REG_INFO(Conv, )},
        {REG_INFO(InstanceNormalization, )},
        // Data Reorganization Layers
        {REG_INFO(Concat, )}, // Adds negative axis.
        {REG_INFO_VER(Slice, 7, )},
        {REG_INFO_VER(Slice, 10, )}, // Adds negative axes.
        {REG_INFO_VER(Slice, 11, )}, // Adds negative axes.
        {REG_INFO_VER(Slice, 13, )}, // Adds negative axes.
        {REG_INFO_VER(Pad, 7, )},
        {REG_INFO_VER(Pad, 11, )}, // https://microsoft.visualstudio.com/OS/_workitems/edit/26007728
        {REG_INFO_VER(Pad, 13, )}, // https://microsoft.visualstudio.com/OS/_workitems/edit/26007728
        {REG_INFO(Constant, )},
        {REG_INFO(Shape, )},
        {REG_INFO(Gather, )},

        // Data reorganization that merely changes the dimensions while keeping the data identical.
        // {REG_INFO_COPY(Unsqueeze, )}, // used by ORT
        {REG_INFO_VER(Unsqueeze, 1)},
        {REG_INFO_VER(Unsqueeze, 11)},
        {REG_INFO_VER(Unsqueeze, 13)},
        // Elementwise
        {REG_INFO(Floor, )},
        {REG_INFO(Add, )},
        {REG_INFO(Mul, )},
        {REG_INFO(Div, )},

        // Imaging Operators
        {REG_INFO_VER(Upsample, 7, )},
        {REG_INFO_VER(Upsample, 9, /*scales*/)},
        {REG_INFO_VER(Upsample, 10, /*scales*/)},
        // Activation Functions
        {REG_INFO(Relu, )},

        // Uncategorized
        {REG_INFO(Cast, )},
};

dml::Expression CreateExpression(const std::map<std::string, dml::Expression> &expressionMap, const Op &node, dml::Graph& graph)
{
    return g_operatorRegistrationMap[node.opType](expressionMap, node, opsetVersion);
}
