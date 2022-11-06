#pragma once
#include "../helper/pch.h"

dml::Expression CreateExpression(std::map<std::string, dml::Expression> &expressionMap, ONNX_PARSER::Op &node, dml::Graph& graph, unsigned int opsetVersion);