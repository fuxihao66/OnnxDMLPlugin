#pragma once
#include "../helper/pch.h"

dml::Expression CreateExpression(const std::map<std::string, dml::Expression> &expressionMap, const ONNX_PARSER::Op &node, dml::Graph& graph, unsigned int opsetVersion);