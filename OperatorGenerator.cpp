

template <typename T>
class OperatorGenerator{
public:
    dml::Expression CreateDmlExpression(const std::map<std::string, dml::Expression>& expressionMap, const Op& node){
        T op(expressionMap, node);

        return op.Create();
    }
};