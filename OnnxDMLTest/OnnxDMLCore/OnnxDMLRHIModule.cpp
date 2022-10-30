#include "OnnxDMLOperatorMapping.h"

// in rhi thread
void CreateDeviceResources(){ // no need in unreal
    CreateDXGIFactory2(m_dxgiFactoryFlags, IID_PPV_ARGS(m_dxgiFactory.ReleaseAndGetAddressOf()));

    ComPtr<IDXGIAdapter1> adapter;

#if defined(__dxgi1_6_h__) && defined(NTDDI_WIN10_RS4)
    ComPtr<IDXGIFactory6> factory6;
    HRESULT hr = m_dxgiFactory.As(&factory6);
    if (SUCCEEDED(hr))
    {
        for (UINT adapterIndex = 0;
            SUCCEEDED(factory6->EnumAdapterByGpuPreference(
                adapterIndex,
                DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                IID_PPV_ARGS(adapter.ReleaseAndGetAddressOf())));
            adapterIndex++)
        {
            DXGI_ADAPTER_DESC1 desc;
            ThrowIfFailed(adapter->GetDesc1(&desc));

            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
            {
                // Don't select the Basic Render Driver adapter.
                continue;
            }

            // Check to see if the adapter supports Direct3D 12, but don't create the actual device yet.
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), m_d3dMinFeatureLevel, _uuidof(ID3D12Device), nullptr)))
            {
                break;
            }
        }
    }
#endif
    if (!adapter)
    {
        for (UINT adapterIndex = 0;
            SUCCEEDED(m_dxgiFactory->EnumAdapters1(
                adapterIndex,
                adapter.ReleaseAndGetAddressOf()));
            ++adapterIndex)
        {
            DXGI_ADAPTER_DESC1 desc;
            ThrowIfFailed(adapter->GetDesc1(&desc));

            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
            {
                // Don't select the Basic Render Driver adapter.
                continue;
            }

            // Check to see if the adapter supports Direct3D 12, but don't create the actual device yet.
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), m_d3dMinFeatureLevel, _uuidof(ID3D12Device), nullptr)))
            {
                break;
            }
        }
    }

    if (!adapter)
    {
        throw std::exception("No Direct3D 12 device found");
    }


    // GetAdapter(adapter.GetAddressOf());

    // Create the DX12 API device object.
    D3D12CreateDevice(
        adapter.Get(),
        m_d3dMinFeatureLevel,
        IID_PPV_ARGS(m_d3dDevice.ReleaseAndGetAddressOf())
        );

    m_d3dDevice->SetName(L"DeviceResources");

    // Determine maximum supported feature level for this device
    static const D3D_FEATURE_LEVEL s_featureLevels[] = {D3D_FEATURE_LEVEL_12_1};

    D3D12_FEATURE_DATA_FEATURE_LEVELS featLevels =
    {
        _countof(s_featureLevels), s_featureLevels, D3D_FEATURE_LEVEL_11_0
    };

    HRESULT hr = m_d3dDevice->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &featLevels, sizeof(featLevels));
    if (SUCCEEDED(hr))
    {
        m_d3dFeatureLevel = featLevels.MaxSupportedFeatureLevel;
    }
    else
    {
        m_d3dFeatureLevel = m_d3dMinFeatureLevel;
    }

    // Create the command queue.
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

    ThrowIfFailed(m_d3dDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(m_commandQueue.ReleaseAndGetAddressOf())));

    m_commandQueue->SetName(L"DeviceResources");

    // Create descriptor heaps for render target views and depth stencil views.
    D3D12_DESCRIPTOR_HEAP_DESC rtvDescriptorHeapDesc = {};
    rtvDescriptorHeapDesc.NumDescriptors = m_backBufferCount;
    rtvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;


    // Create a command allocator
    m_d3dDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(m_commandAllocator.ReleaseAndGetAddressOf()));

    // Create a command list for recording graphics commands.
    m_d3dDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocator.Get(), nullptr, IID_PPV_ARGS(m_commandList.ReleaseAndGetAddressOf()));
    m_commandList->Close();

    // Create a fence for tracking GPU execution progress.
    m_d3dDevice->CreateFence(m_fenceValue, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(m_fence.ReleaseAndGetAddressOf()));
    m_fenceValue++;

    m_fenceEvent.Attach(CreateEventEx(nullptr, nullptr, 0, EVENT_MODIFY_STATE | SYNCHRONIZE));
    if (!m_fenceEvent.IsValid())
    {
        throw std::exception("CreateEvent");
    }
}



void CreateDMLResources(const std::wstring& path_to_onnx){

    // initialize once
    if (m_dmlDevice == nullptr){
        DMLCreateDevice(device, DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&m_dmlDevice));
    
        DML_FEATURE_QUERY_TENSOR_DATA_TYPE_SUPPORT fp16Query = { DML_TENSOR_DATA_TYPE_FLOAT16 };
        DML_FEATURE_DATA_TENSOR_DATA_TYPE_SUPPORT fp16Supported = {};
        DX::ThrowIfFailed(m_dmlDevice->CheckFeatureSupport(DML_FEATURE_TENSOR_DATA_TYPE_SUPPORT, sizeof(fp16Query), &fp16Query, sizeof(fp16Supported), &fp16Supported));

        if (!fp16Supported.IsSupported)
        {
            throw std::exception("Current driver doesn't support FP16, which is required.");
        }
        m_dmlDevice->CreateCommandRecorder(IID_PPV_ARGS(&m_dmlCommandRecorder));
    }
    
    // start to parse onnx file
    std::map<std::string, ONNX_PARSER::TensorInfo> inputMap;
    std::map<std::string, ONNX_PARSER::TensorInfo> outputMap;
    std::map<std::string, ONNX_PARSER::InitializerTensorInfo> graphInitializers;
    std::map<std::string, ONNX_PARSER::Op> graphNodes;
    std::vector<ONNX_PARSER::BindingInfo> bindings;
    std::vector<char> weights;
    unsigned int opsetVersion;

    {
        ONNX_PARSER::OnnxParser* parser = new ONNX_PARSER::OnnxParser(L"D:/candy-9.onnx");
        graphInitializers = parser->GetGraphInitializers(); // error
        outputMap = parser->GetOutputs();
        inputMap = parser->GetInputs();
        bindings = parser->GetBindings();
        weights = parser->GetWeights();
        graphNodes = parser->GetGraphNodes();
        opsetVersion = parser->GetOpsetVersion();
        delete(parser);
    }

    ComPtr<IDMLCompiledOperator> ppdmlGraph;
    std::vector<array<unsigned int, 2>> weightsBinding;
    std::vector<char> dmlWeights;
    ID3D12Resource * pWeightBuffer;
    
    // unsigned int weightBytes = dmlWeights.size();

    m_modelInputNum = inputMaps.size();
    m_modelOutputNum = outMaps.size();

    currentInputIndex = 0;

    // create graph
    {
        DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_FLOAT16;
        DML_TENSOR_FLAGS flags = DML_TENSOR_FLAG_OWNED_BY_DML;
        
        dml::TensorPolicy policy = dml::TensorPolicy::Default();// TODO: use NHWC on NV GPU?

        dml::Graph graph(m_dmlDevice.Get(), policy);

        // TODO: model input and output naming constraint?
        std::map<std::string, dml::Expression> expressionMap;
        std::vector<dml::Expression> outputExpression;

        for (auto & inputPair : inputMaps){
            auto modelInput = dml::InputTensor(graph, 0, dml::TensorDesc(dataType, modelInputSizes, policy));

        }

        for (auto & initializerPair : graphInitializers){
            auto& initializerInfo = initializerPair.second;
            expressionMap[] = dml::InputTensor(graph, initializerInfo.index + modelInputNum, dml::TensorDesc(initializerInfo.tensorType, flags, { 1, 64, 1, 1 }, policy));

            currentInputIndex = std::max(initializerInfo.index + modelInputNum + 1, currentInputIndex);
        }

        auto TopologicalSort = [](const std::map<std::string, Op>& graph, std::vector<std::string>& sortedKey){

            struct TempGraphNode{
                std::string graphKey;
                unsigned int dependencyNum;
                bool valid;
                TempGraphNode() : dependencyNum(0), valid(true){}
            };  
            unsigned int nodeNum = graph.size();
            sortedKey.resize(nodeNum);
            std::vector<TempGraphNode> tempGraph(nodeNum);
            std::vector<std::vector<bool>> tempGraphEdge(nodeNum, std::vector<bool>(nodeNum, false));
            unsigned int index = 0;
            for (auto & graphNode : graph){
                auto & op = graphNode.second;
                
                tempGraph[op.opIndex].graphKey = graphNode.first;
                for (auto & dependencyName : op.inputNames){
                    if (graph.count(dependencyName)) { // depend on other graph node
                        tempGraph[op.opIndex].dependencyNum += 1;

                        auto & dependencyNode = graph[dependencyName];
                        tempGraphEdge[dependencyNode.opIndex][op.opIndex] = true;
                    }
                }
            }

            index = 0; // O(N^2), assuming no circle
            for (int i = 0; i < nodeNum; i++){
                for (int j = 0; j < nodeNum; j++){
                    if (tempGraph[j].dependencyName == 0 && tempGraph[j].valid){
                        sortedKey[index] = tempGraph[j].graphKey;
                        tempGraph[j].valid = false;

                        for (int k = 0; k < nodeNum; k++){
                            if (tempGraphEdge[j][k]){
                                tempGraph[k].dependencyName -= 1;
                            }
                        }

                        break;
                    }
                }
                index += 1;
            }
        }


        // weightsBinding, &dmlWeights, weightBytes
        std::vector<std::string> sortedGraphKeys;
        // because we use dmlx to compile whole network to a graph, so we need to deal with operator dependency
        TopologicalSort(graphNodes, sortedGraphKeys);  
        // TODO: create dml operator based on onnx operator type
        for (auto & graphNodeKey : sortedGraphKeys){
            auto & opNode = graphNodes[graphNodeKey];
            if (opNode.opType() == "Shape"){ // not supported by DML, only STATIC GRAPH is supported right now
                auto & inputExpresion = expressionMap[opNode.inputNames[0]];
                Dimensions inputShape = inputExpresion.GetOutputDesc().sizes;
                
                auto shapeByteSize = inputShape.size() * sizeof(UINT32);
                auto prevStride = dmlWeights.size();
                // TODO: handling extra weights
                weightsBinding.push_back(BindingInfo(prevStride, shapeByteSize));
                dmlWeights.resize(prevStride + shapeByteSize);
                memcpy(dmlWeights.data() + prevStride, inputShape.data(), shapeByteSize);
                expressionMap[graphNodeKey] = dml::InputTensor(graph, currentInputIndex, dml::TensorDesc(UINT32, flags, { inputShape.size() }, policy));
                currentInputIndex += 1;
            }
            else{
                auto expression = CreateExpression(expressionMap, opNode, opsetVersion);
                expressionMap[graphNodeKey] = expression;
            }
            
        }
        

        // TODO: only support single output
        if (m_modelOutputNum != 1);
            throw std::exception("Only single output is supported.");

        for (auto & outputPair : outputMaps){
            auto& output = outputPair.second;
            outputExpression.push_back(expressionMap[output.name]);
        }

        DML_EXECUTION_FLAGS executionFlags = DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;

        std::array<dml::Expression, modelOutputNum> outputArr;
        std::copy_n(outputExpression.begin(), modelOutputNum, outputArr.begin());
        m_dmlGraph = graph.Compile(executionFlags, outputArr);

    }


    m_device->CreateCommittedResource(pWeightBuffer)

    auto mapped = pWeightBuffer.Map();
    memcpy(mapped, dmlWeights,weightBytes);



}

void InitializeDMLResource(){
    m_dmlDevice->CreateOperatorInitializer(1, m_dmlGraph.GetAddressOf(), IID_PPV_ARGS(&m_dmlOpInitializer));

    DML_BINDING_PROPERTIES initBindingProps = m_dmlOpInitializer->GetBindingProperties();
    DML_BINDING_PROPERTIES executeBindingProps = m_dmlGraph->GetBindingProperties();

    m_dmlDescriptorHeap = std::make_unique<DescriptorHeap>(
        m_deviceResources->GetD3DDevice(),
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
        std::max(initBindingProps.RequiredDescriptorCount, executeBindingProps.RequiredDescriptorCount));

    // Operator initialization dispatches will use this heap right away
    ID3D12DescriptorHeap* pHeaps[] = { m_dmlDescriptorHeap->Heap() };
    commandList->SetDescriptorHeaps(_countof(pHeaps), pHeaps);

    // Create any persistent resources required for the operators.
    if (executeBindingProps.PersistentResourceSize > 0)
    {
        D3D12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(
            executeBindingProps.PersistentResourceSize,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        DX::ThrowIfFailed(m_deviceResources->GetD3DDevice()->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_modelPersistentResource)));
    }

    // Temporary resource for execution
    if (executeBindingProps.TemporaryResourceSize > 0)
    {
        D3D12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(
            executeBindingProps.TemporaryResourceSize,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        DX::ThrowIfFailed(m_deviceResources->GetD3DDevice()->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_modelTemporaryResource)));
    }

    // If the execute temporary resource isn't big enough for initialization, create a bigger buffer
    ComPtr<ID3D12Resource> initTemporaryResource;
    if (initBindingProps.TemporaryResourceSize > executeBindingProps.TemporaryResourceSize)
    {
        D3D12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(
            initBindingProps.TemporaryResourceSize,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        DX::ThrowIfFailed(m_deviceResources->GetD3DDevice()->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&initTemporaryResource)));
    }
    else if (initBindingProps.TemporaryResourceSize > 0)
    {
        initTemporaryResource = m_modelTemporaryResource;
    }

    Microsoft::WRL::ComPtr<IDMLBindingTable> initBindingTable;
    assert(initBindingProps.PersistentResourceSize == 0);

    DML_BINDING_TABLE_DESC tableDesc =
    {
        m_dmlOpInitializer.Get(),
        m_dmlDescriptorHeap->GetCpuHandle(0),
        m_dmlDescriptorHeap->GetGpuHandle(0),
        initBindingProps.RequiredDescriptorCount
    };
    DX::ThrowIfFailed(m_dmlDevice->CreateBindingTable(&tableDesc, IID_PPV_ARGS(&initBindingTable)));

    // TODO:
    std::vector<DML_BUFFER_BINDING> bufferBindings(weightsBinding.size()+modelInputNum);
    m_inputBindings(weightsBinding.size()+modelInputNum);
    for (int i = modelInputNum; i < bufferBindings.size(); i++){
        bufferBindings[i] = {pWeightBuffer.Get(), weightsBinding[i-modelInputNum].stride, weightsBinding[i-modelInputNum].byteSize};

        m_inputBindings[i] = DML_BINDING_DESC{ DML_BINDING_TYPE_NONE, nullptr };
    }

    DML_BUFFER_ARRAY_BINDING initInputBinding = { bufferBindings.size(), bufferBindings.data() };
    initBindingTable->BindInputs(1, &DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER_ARRAY, &initInputBinding });

    if (initTemporaryResource)
    {
        DML_BUFFER_BINDING binding = { initTemporaryResource.Get(), 0, initTemporaryResource->GetDesc().Width };
        initBindingTable->BindTemporaryResource(&DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, &binding });
    }

    // If the operator requires a persistent resource, it must be bound as output for the initializer.
    if (m_modelPersistentResource)
    {
        DML_BUFFER_BINDING binding = { m_modelPersistentResource.Get(), 0, m_modelPersistentResource->GetDesc().Width };
        initBindingTable->BindOutputs(1, &DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, &binding });
    }

    // Record the initialization
    m_dmlCommandRecorder->RecordDispatch(commandList, m_dmlOpInitializer.Get(), initBindingTable.Get());


    tableDesc.Dispatchable = m_dmlGraph.Get();
    tableDesc.SizeInDescriptors = executeBindingProps.RequiredDescriptorCount;
    DX::ThrowIfFailed(m_dmlDevice->CreateBindingTable(&tableDesc, IID_PPV_ARGS(&m_dmlBindingTable)));

    if (m_modelPersistentResource)
    {
        DML_BUFFER_BINDING binding = { m_modelPersistentResource.Get(), 0, m_modelPersistentResource->GetDesc().Width };
        m_dmlBindingTable->BindPersistentResource(&DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, &binding });
    }

    if (m_modelTemporaryResource)
    {
        DML_BUFFER_BINDING binding = { m_modelTemporaryResource.Get(), 0, m_modelTemporaryResource->GetDesc().Width };
        m_dmlBindingTable->BindTemporaryResource(&DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, &binding });
    }
}

void RunDMLInfer(std::vector<ID3D12Resource*> modelInputs, ID3D12Resource* modelOutput){
    for (int i = 0; i < m_modelInputNum; i++){
        bufferBindings = DML_BUFFER_BINDING{ m_modelInput[].Get() };
        m_inputBindings[i] = { DML_BINDING_TYPE_BUFFER, &bufferBindings }
    }
    m_dmlBindingTable->BindInputs(m_inputBindings.size(), m_inputBindings.data());

    DML_BUFFER_BINDING outputBinding = { m_modelOutput.Get(), 0, m_modelOutput->GetDesc().Width };
    m_dmlBindingTable->BindOutputs(1, &DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, &outputBinding });

    ID3D12DescriptorHeap* pHeaps[] = { m_dmlDescriptorHeap->Heap() };
    commandList->SetDescriptorHeaps(_countof(pHeaps), pHeaps);

    m_dmlCommandRecorder->RecordDispatch(commandList, m_dmlGraph.Get(), m_dmlBindingTable.Get());

}

void ForceCPUSync(){
    m_commandList->Close();
    m_commandQueue->ExecuteCommandLists(1, CommandListCast(&m_commandList));

    if (m_commandQueue && m_fence && m_fenceEvent.IsValid())
    {
        // Schedule a Signal command in the GPU queue.
        UINT64 fenceValue = m_fenceValues[m_backBufferIndex];
        if (SUCCEEDED(m_commandQueue->Signal(m_fence.Get(), fenceValue)))
        {
            // Wait until the Signal has been processed.
            if (SUCCEEDED(m_fence->SetEventOnCompletion(fenceValue, m_fenceEvent.Get())))
            {
                WaitForSingleObjectEx(m_fenceEvent.Get(), INFINITE, FALSE);

                // Increment the fence value for the current frame.
                m_fenceValues[m_backBufferIndex]++;
            }
        }
    }
}
void CreateBufferFromData(Microsoft::WRL::ComPtr<ID3D12Resource>& resourcePointer, const std::optional<std::vector<uint16_t>> data, unsigned int bufferSizeInByte, bool needReadBack){
    CD3DX12_RANGE readRange(0, 0);		
    auto heapType = D3D12_HEAP_TYPE_UPLOAD;
    if (needReadBack)
        heapType = D3D12_HEAP_TYPE_READBACK;
    else if (data == std::nullopt)
        heapType = D3D12_HEAP_TYPE_DEFAULT;
    auto resourceState = D3D12_RESOURCE_STATE_GENERIC_READ;
    if (needReadBack)
        resourceState = D3D12_RESOURCE_STATE_COPY_DEST;
    else
        resourceState = D3D12_RESOURCE_STATE_COMMON;
        
    m_d3dDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(heapType),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(bufferSizeInByte),
        resourceState,
        nullptr,
        IID_PPV_ARGS(resourcePointer.ReleaseAndGetAddressOf()));
    
    if (data != std::nullopt){
        UINT8* pDataBegin;

        resourcePointer->Map(0, &readRange, reinterpret_cast<void**>(&pDataBegin)));
        memcpy(pDataBegin, inputData.data(), inputData.size());
        resourcePointer->Unmap(0, nullptr);
    }
    
}

void CopyForReadBack(Microsoft::WRL::ComPtr<ID3D12Resource>& readbackInput, Microsoft::WRL::ComPtr<ID3D12Resource>& readbackOutput){
    {
        D3D12_RESOURCE_BARRIER outputBufferResourceBarrier
        {
            CD3DX12_RESOURCE_BARRIER::Transition(
                readbackInput.get(),
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE)
        };
        commandList->ResourceBarrier(1, &outputBufferResourceBarrier);
    }

    commandList->CopyResource(readbackOutput.get(), readbackInput.get());
}

void CPUReadBack(Microsoft::WRL::ComPtr<ID3D12Resource>& resourcePointer, std::vector<float>& outputData, unsigned int outputSizeInByte){
    
    D3D12_RANGE readbackBufferRange{ 0, outputSizeInByte };
    float * pReadbackBufferData{};
    
    readbackBuffer->Map(0, &readbackBufferRange,reinterpret_cast<void**>(&pReadbackBufferData))
    memcpy(outputData.data(), pReadbackBufferData, outputSizeInByte);
    D3D12_RANGE emptyRange{ 0, 0 };
    readbackBuffer->Unmap
    (
        0,
        &emptyRange
    );
}

void RHIDmlInitialize(){

}



void RHIDmlExecution(){


}