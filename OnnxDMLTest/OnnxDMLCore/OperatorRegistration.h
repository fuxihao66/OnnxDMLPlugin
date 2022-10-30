#pragma once

#include "OnnxDMLOperatorGenerator.h"

// operator that has different parameter with different ir version
#define REG_INFO_VER(operatorName, version) \
    #operatorName, Create##operatorName##version,
#define REG_INFO(operatorName) \
    #operatorName, Create##operatorName,
// operaters that can be mapped to DmlOperatorCopy
#define REG_INFO_COPY(operatorName) \
    #operatorName, CreateCopy,

#define DML_OP_EXTERN_CREATION_FUNCTION(operatorName) CALLBACK Create##operatorName(const std::map<std::string, dml::Expression> &expressionMap, const Op &node, dml::Graph& graph)


#define DML_OP_DEFINE_CREATION_FUNCTION(operatorName, ...)                                                                               \
    \
extern dml::Expression CALLBACK Create##operatorName(const std::map<std::string, dml::Expression> &expressionMap, const Op &node, dml::Graph& graph) \
    \
{                                                                                                                                   \
        using T = __VA_ARGS__;                                                                                                           \
        OperatorGenerator<T>::CreateDmlExpression(expressionMap, node);                                                                  \
    \
}