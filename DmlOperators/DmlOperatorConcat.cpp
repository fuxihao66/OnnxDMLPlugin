// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{


class DmlOperatorConcat 
{
public:
    DmlOperatorConcat(const std::map<std::string, dml::Expression>& expressionMap, const Op& node, dml::Graph& graph)
    {
        if (node.inputNames.size() <= 1)
            assert(false);  // 

        
        for (auto & inputName : node.inputNames ){
            m_inputs.push_back(expressionMap[inputName]);
        }

        // get from attribute
        int tempaxis = ;
        
        if (tempaxis < 0)
            axis = node.inputInfo[0].dims + tempaxis
        else    
            axis = tempaxis;

    }

    dml::Expression Create(){
        dml::Join(m_inputs, axis);
    }
private:
    std::vector<dml:Expression> m_inputs;
    uint32_t axis;
};

DML_OP_DEFINE_CREATION_FUNCTION(Concat, DmlOperatorConcat);

} // namespace Dml
