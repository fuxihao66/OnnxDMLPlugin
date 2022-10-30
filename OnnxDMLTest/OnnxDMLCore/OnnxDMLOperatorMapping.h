#pragma once


dml::Expression CreateExpression(const std::map<std::string, dml::Expression> &expressionMap, const Op &node, dml::Graph& graph);