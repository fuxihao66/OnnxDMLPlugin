#include "OnnxDMLOperatorMapping.h"
#include "OnnxDMLRHIModule.h"

#include "Common/OnnxParser.h"

namespace ODI {
    D3D12RHIContext::D3D12RHIContext() {
        CreateDeviceResources();
        CreateDMLResources();
    }

    // in rhi thread
    void D3D12RHIContext::CreateDeviceResources() { // no need in unreal
        CreateDXGIFactory2(m_dxgiFactoryFlags, IID_PPV_ARGS(m_dxgiFactory.ReleaseAndGetAddressOf()));

        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;

#if defined(__dxgi1_6_h__) && defined(NTDDI_WIN10_RS4)
        Microsoft::WRL::ComPtr<IDXGIFactory6> factory6;
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
                adapter->GetDesc1(&desc);

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
                adapter->GetDesc1(&desc);

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
        static const D3D_FEATURE_LEVEL s_featureLevels[] = { D3D_FEATURE_LEVEL_12_1 };

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

        m_d3dDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(m_commandQueue.ReleaseAndGetAddressOf()));

        m_commandQueue->SetName(L"DeviceResources");

        //// Create descriptor heaps for render target views and depth stencil views.
        //D3D12_DESCRIPTOR_HEAP_DESC rtvDescriptorHeapDesc = {};
        //rtvDescriptorHeapDesc.NumDescriptors = m_backBufferCount;
        //rtvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;


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
    void D3D12RHIContext::CreateDMLResources() {



        // initialize once
        if (m_dmlDevice == nullptr) {
            DMLCreateDevice(m_d3dDevice.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&m_dmlDevice));

            DML_FEATURE_QUERY_TENSOR_DATA_TYPE_SUPPORT fp16Query = { DML_TENSOR_DATA_TYPE_FLOAT16 };
            DML_FEATURE_DATA_TENSOR_DATA_TYPE_SUPPORT fp16Supported = {};
            m_dmlDevice->CheckFeatureSupport(DML_FEATURE_TENSOR_DATA_TYPE_SUPPORT, sizeof(fp16Query), &fp16Query, sizeof(fp16Supported), &fp16Supported);

            if (!fp16Supported.IsSupported)
            {
                throw std::exception("Current driver doesn't support FP16, which is required.");
            }
            m_dmlDevice->CreateCommandRecorder(IID_PPV_ARGS(&m_dmlCommandRecorder));
        }



        m_dmlDescriptorHeap = std::make_unique<DirectX::DescriptorHeap>(
            m_d3dDevice.Get(),
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            MAX_DESCRIPTOR_COUNT);

    }
    void D3D12RHIContext::InitializeNewModel(const std::wstring& path_to_onnx, const std::string& modelName) {
        m_modelNameToResourceInfo[modelName] = ModelInfo();

        auto& currOnnxInfo = m_modelNameToResourceInfo[modelName];

        // start to parse onnx file
        std::map<std::string, ONNX_PARSER::TensorInfo> inputMap;
        std::map<std::string, ONNX_PARSER::TensorInfo> outputMap;
        std::map<std::string, ONNX_PARSER::InitializerTensorInfo> graphInitializers;
        std::map<std::string, ONNX_PARSER::Op> graphNodes;
        std::vector<ONNX_PARSER::BindingInfo> weightsBinding;
        std::vector<char> dmlWeights;
        unsigned int opsetVersion;

        {
            ONNX_PARSER::OnnxParser* parser = new ONNX_PARSER::OnnxParser(L"D:/candy-9.onnx");
            graphInitializers = parser->GetGraphInitializers(); // error
            outputMap = parser->GetOutputs();
            inputMap = parser->GetInputs();
            weightsBinding = parser->GetBindings();
            dmlWeights = parser->GetWeights();
            graphNodes = parser->GetGraphNodes();
            opsetVersion = parser->GetOpsetVersion();
            delete(parser);
        }

        //Microsoft::WRL::ComPtr<IDMLCompiledOperator> ppdmlGraph;
        //ID3D12Resource* pWeightBuffer;

        // unsigned int weightBytes = dmlWeights.size();

        unsigned int modelInputNum = inputMap.size();
        unsigned int modelOutputNum = outputMap.size();

        currOnnxInfo.modelInputNum = modelInputNum;
        currOnnxInfo.modelOutputNum = modelOutputNum;

        unsigned int currentInputIndex = 0;

        // create graph
        {
            DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_FLOAT16;
            DML_TENSOR_FLAGS flags = DML_TENSOR_FLAG_OWNED_BY_DML;

            dml::TensorPolicy policy = dml::TensorPolicy::Default();// TODO: use NHWC on NV GPU?

            dml::Graph graph(m_dmlDevice.Get(), policy);

            // TODO: model input and output naming constraint?
            std::map<std::string, dml::Expression> expressionMap;
            std::vector<dml::Expression> outputExpression;

            int inputIndex = 0;
            for (auto& inputPair : inputMap) {
                auto& inputInfo = inputPair.second;
                dml::TensorDimensions modelInputShape;
                
                for (int i = 0; i < inputInfo.dims; i++) {
                    modelInputShape.push_back(inputInfo.shapes[i]);
                }
                auto modelInput = dml::InputTensor(graph, inputIndex, dml::TensorDesc(static_cast<DML_TENSOR_DATA_TYPE>(inputInfo.tensorType), modelInputShape, policy));
                expressionMap[inputPair.first] = modelInput;
                inputIndex += 1;
            }

            for (auto& initializerPair : graphInitializers) {
                auto& initializerInfo = initializerPair.second;

                dml::TensorDimensions modelWeightShape;

                for (int i = 0; i < initializerInfo.dims; i++) {
                    modelWeightShape.push_back(initializerInfo.shapes[i]);
                }

                expressionMap[initializerPair.first] = dml::InputTensor(graph, initializerInfo.index + modelInputNum, 
                    dml::TensorDesc(static_cast<DML_TENSOR_DATA_TYPE>(initializerInfo.tensorType), flags, modelWeightShape, policy));

                currentInputIndex = std::max(initializerInfo.index + modelInputNum + 1, currentInputIndex);
            }

            auto TopologicalSort = [](std::map<std::string, ONNX_PARSER::Op>& graph, std::vector<std::string>& sortedKey) {

                struct TempGraphNode {
                    std::string graphKey;
                    unsigned int dependencyNum;
                    bool valid;
                    TempGraphNode() : dependencyNum(0), valid(true) {}
                };
                unsigned int nodeNum = graph.size();
                sortedKey.resize(nodeNum);
                std::vector<TempGraphNode> tempGraph(nodeNum);
                std::vector<std::vector<bool>> tempGraphEdge(nodeNum, std::vector<bool>(nodeNum, false));
                unsigned int index = 0;
                for (auto& graphNode : graph) {
                    auto& op = graphNode.second;

                    tempGraph[op.opIndex].graphKey = graphNode.first;
                    for (const std::string& dependencyName : op.inputNames) {
                        if (graph.count(dependencyName)) { // depend on other graph node
                            tempGraph[op.opIndex].dependencyNum += 1;

                            auto& dependencyNode = graph[dependencyName];
                            tempGraphEdge[dependencyNode.opIndex][op.opIndex] = true;
                        }
                    }
                }

                index = 0; // O(N^2), assuming no circle
                for (int i = 0; i < nodeNum; i++) {
                    for (int j = 0; j < nodeNum; j++) {
                        if (tempGraph[j].dependencyNum == 0 && tempGraph[j].valid) {
                            sortedKey[index] = tempGraph[j].graphKey;
                            tempGraph[j].valid = false;

                            for (int k = 0; k < nodeNum; k++) {
                                if (tempGraphEdge[j][k]) {
                                    tempGraph[k].dependencyNum -= 1;
                                }
                            }

                            break;
                        }
                    }
                    index += 1;
                }
            };


            // weightsBinding, &dmlWeights, weightBytes
            std::vector<std::string> sortedGraphKeys;
            // because we use dmlx to compile whole network to a graph, so we need to deal with operator dependency
            TopologicalSort(graphNodes, sortedGraphKeys);
            // TODO: create dml operator based on onnx operator type
            for (auto& graphNodeKey : sortedGraphKeys) {
                auto& opNode = graphNodes[graphNodeKey];
                if (opNode.opType == "Shape") { // not supported by DML, only STATIC GRAPH is supported right now
                    auto& inputExpresion = expressionMap[opNode.inputNames[0]];
                    dml::TensorDimensions inputShape = inputExpresion.GetOutputDesc().sizes;

                    auto shapeByteSize = inputShape.size() * sizeof(UINT32);
                    auto prevStride = dmlWeights.size();
                    // TODO: handling extra weights
                    weightsBinding.push_back(ONNX_PARSER::BindingInfo(prevStride, shapeByteSize));
                    dmlWeights.resize(prevStride + shapeByteSize);
                    memcpy(dmlWeights.data() + prevStride, inputShape.data(), shapeByteSize);
                    expressionMap[graphNodeKey] = dml::InputTensor(graph, currentInputIndex, 
                        dml::TensorDesc(static_cast<DML_TENSOR_DATA_TYPE>(ONNX_PARSER::TensorType::UINT32), flags, dml::TensorDimensions{ static_cast<uint32_t>(inputShape.size()) }, policy));
                    currentInputIndex += 1;
                }
                else {
                    auto expression = CreateExpression(expressionMap, opNode, graph, opsetVersion);
                    expressionMap[graphNodeKey] = expression;
                }

            }


            // TODO: only support single output
            if (modelOutputNum != 1);
            throw std::exception("Only single output is supported.");

            for (auto& outputPair : outputMap) {
                auto& output = outputPair.second;
                outputExpression.push_back(expressionMap[output.name]);
            }

            DML_EXECUTION_FLAGS executionFlags = DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;

            std::array<dml::Expression, 1> outputArr;
            std::copy_n(outputExpression.begin(), modelOutputNum, outputArr.begin());
            currOnnxInfo.dmlGraph = graph.Compile(executionFlags, outputArr).Get();


        }

        { // create and upload resource
            CD3DX12_RANGE readRange(0, 0);
            m_d3dDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(dmlWeights.size()),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(&currOnnxInfo.modelOperatorWeights));
            UINT8* pDataBegin;

            currOnnxInfo.modelOperatorWeights->Map(0, &readRange, reinterpret_cast<void**>(&pDataBegin));
            memcpy(pDataBegin, dmlWeights.data(), dmlWeights.size());
            currOnnxInfo.modelOperatorWeights->Unmap(0, nullptr);
        }
        

        


        m_dmlDevice->CreateOperatorInitializer(1, currOnnxInfo.dmlGraph.GetAddressOf(), IID_PPV_ARGS(&currOnnxInfo.dmlOpInitializer));

        DML_BINDING_PROPERTIES initBindingProps = currOnnxInfo.dmlOpInitializer->GetBindingProperties();
        DML_BINDING_PROPERTIES executeBindingProps = currOnnxInfo.dmlGraph->GetBindingProperties();



        // Operator initialization dispatches will use this heap right away
        ID3D12DescriptorHeap* pHeaps[] = { m_dmlDescriptorHeap->Heap() };
        m_commandList->SetDescriptorHeaps(_countof(pHeaps), pHeaps);

        // Create any persistent resources required for the operators.
        if (executeBindingProps.PersistentResourceSize > 0)
        {
            D3D12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(
                executeBindingProps.PersistentResourceSize,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

            m_d3dDevice->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &resourceDesc,
                D3D12_RESOURCE_STATE_COMMON,
                nullptr,
                IID_PPV_ARGS(&currOnnxInfo.modelPersistentResource));
        }

        // Temporary resource for execution
        if (executeBindingProps.TemporaryResourceSize > 0)
        {
            D3D12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(
                executeBindingProps.TemporaryResourceSize,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

            m_d3dDevice->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &resourceDesc,
                D3D12_RESOURCE_STATE_COMMON,
                nullptr,
                IID_PPV_ARGS(&currOnnxInfo.modelTemporaryResource));
        }

        // If the execute temporary resource isn't big enough for initialization, create a bigger buffer
        Microsoft::WRL::ComPtr<ID3D12Resource> initTemporaryResource;
        if (initBindingProps.TemporaryResourceSize > executeBindingProps.TemporaryResourceSize)
        {
            D3D12_RESOURCE_DESC resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(
                initBindingProps.TemporaryResourceSize,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

            m_d3dDevice->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &resourceDesc,
                D3D12_RESOURCE_STATE_COMMON,
                nullptr,
                IID_PPV_ARGS(&initTemporaryResource));
        }
        else if (initBindingProps.TemporaryResourceSize > 0)
        {
            initTemporaryResource = currOnnxInfo.modelTemporaryResource;
        }

        Microsoft::WRL::ComPtr<IDMLBindingTable> initBindingTable;
        assert(initBindingProps.PersistentResourceSize == 0);

        DML_BINDING_TABLE_DESC tableDesc =
        {
            currOnnxInfo.dmlOpInitializer.Get(),
            m_dmlDescriptorHeap->GetCpuHandle(currOnnxInfo.descriptorCPUOffset), // TODO: 
            m_dmlDescriptorHeap->GetGpuHandle(currOnnxInfo.descriptorGPUOffset),
            initBindingProps.RequiredDescriptorCount
        };
        m_dmlDevice->CreateBindingTable(&tableDesc, IID_PPV_ARGS(&initBindingTable));



        // TODO:
        currOnnxInfo.inputBindings.resize(weightsBinding.size() + modelInputNum);
        
        std::vector<DML_BUFFER_BINDING> bufferBindings(weightsBinding.size() + modelInputNum);
        for (int i = modelInputNum; i < bufferBindings.size(); i++) {
            bufferBindings[i] = { currOnnxInfo.modelOperatorWeights.Get(), weightsBinding[i - modelInputNum].stride, weightsBinding[i - modelInputNum].byteSize };

            currOnnxInfo.inputBindings[i] = DML_BINDING_DESC{ DML_BINDING_TYPE_NONE, nullptr };
        }

        DML_BUFFER_ARRAY_BINDING initInputBinding = { bufferBindings.size(), bufferBindings.data() };
        initBindingTable->BindInputs(1, &DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER_ARRAY, &initInputBinding });

        if (initTemporaryResource)
        {
            DML_BUFFER_BINDING binding = { initTemporaryResource.Get(), 0, initTemporaryResource->GetDesc().Width };
            initBindingTable->BindTemporaryResource(&DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, &binding });
        }

        // If the operator requires a persistent resource, it must be bound as output for the initializer.
        if (currOnnxInfo.modelPersistentResource)
        {
            DML_BUFFER_BINDING binding = { currOnnxInfo.modelPersistentResource.Get(), 0, currOnnxInfo.modelPersistentResource->GetDesc().Width };
            initBindingTable->BindOutputs(1, &DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, &binding });
        }

        // Record the initialization
        m_dmlCommandRecorder->RecordDispatch(m_commandList.Get(), currOnnxInfo.dmlOpInitializer.Get(), initBindingTable.Get());


        tableDesc.Dispatchable = currOnnxInfo.dmlGraph.Get();
        tableDesc.SizeInDescriptors = executeBindingProps.RequiredDescriptorCount;
        m_dmlDevice->CreateBindingTable(&tableDesc, IID_PPV_ARGS(&currOnnxInfo.dmlBindingTable));

        if (currOnnxInfo.modelPersistentResource)
        {
            DML_BUFFER_BINDING binding = { currOnnxInfo.modelPersistentResource.Get(), 0, currOnnxInfo.modelPersistentResource->GetDesc().Width };
            currOnnxInfo.dmlBindingTable->BindPersistentResource(&DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, &binding });
        }

        if (currOnnxInfo.modelTemporaryResource)
        {
            DML_BUFFER_BINDING binding = { currOnnxInfo.modelTemporaryResource.Get(), 0, currOnnxInfo.modelTemporaryResource->GetDesc().Width };
            currOnnxInfo.dmlBindingTable->BindTemporaryResource(&DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, &binding });
        }
    }

    

    /*void D3D12RHIContext::InitializeDMLResource() {
        
    }*/

    void D3D12RHIContext::RunDMLInfer(const std::map<std::string, ID3D12Resource*> modelInputs, ID3D12Resource* modelOutput, const std::string& modelName) {
        auto& currOnnxInfo = m_modelNameToResourceInfo[modelName];

        auto modelInputNum = currOnnxInfo.modelInputNum;
        auto modelOutputNum = currOnnxInfo.modelOutputNum;
        auto dmlGraph = currOnnxInfo.dmlGraph;
        auto dmlBindingTable = currOnnxInfo.dmlBindingTable;


        int inputIndex = 0;
        for (auto& input : modelInputs) {
            auto resourcePointer = input.second;
            auto bufferBindings = DML_BUFFER_BINDING{ resourcePointer };
            currOnnxInfo.inputBindings[inputIndex] = { DML_BINDING_TYPE_BUFFER, &bufferBindings };

            inputIndex += 1;
        }
        
        dmlBindingTable->BindInputs(currOnnxInfo.inputBindings.size(), currOnnxInfo.inputBindings.data());

        DML_BUFFER_BINDING outputBinding = { modelOutput, 0, modelOutput->GetDesc().Width };
        dmlBindingTable->BindOutputs(1, &DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, &outputBinding });

        ID3D12DescriptorHeap* pHeaps[] = { m_dmlDescriptorHeap->Heap() };
        m_commandList->SetDescriptorHeaps(_countof(pHeaps), pHeaps);

        m_dmlCommandRecorder->RecordDispatch(m_commandList.Get(), dmlGraph.Get(), dmlBindingTable.Get());

    }

    void D3D12RHIContext::ForceCPUSync() {
        m_commandList->Close();
        ID3D12CommandList* commandLists[] = { m_commandList.Get() };

        m_commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);

        if (m_commandQueue && m_fence && m_fenceEvent.IsValid())
        {
            // Schedule a Signal command in the GPU queue.
            UINT64 fenceValue = m_fenceValue;
            if (SUCCEEDED(m_commandQueue->Signal(m_fence.Get(), fenceValue)))
            {
                // Wait until the Signal has been processed.
                if (SUCCEEDED(m_fence->SetEventOnCompletion(fenceValue, m_fenceEvent.Get())))
                {
                    WaitForSingleObjectEx(m_fenceEvent.Get(), INFINITE, FALSE);

                    // Increment the fence value for the current frame.
                    m_fenceValue++;
                }
            }
        }
    }
    void D3D12RHIContext::CreateBufferFromData(Microsoft::WRL::ComPtr<ID3D12Resource> resourcePointer, const std::optional<std::vector<uint16_t>> data, unsigned int bufferSizeInByte, bool needReadBack) {
        CD3DX12_RANGE readRange(0, 0);
        auto heapType = D3D12_HEAP_TYPE_UPLOAD;
        if (needReadBack)
            heapType = D3D12_HEAP_TYPE_READBACK;
        else if (data == std::nullopt)
            heapType = D3D12_HEAP_TYPE_DEFAULT;
        auto resourceState = D3D12_RESOURCE_STATE_GENERIC_READ;
        if (needReadBack)
            resourceState = D3D12_RESOURCE_STATE_COPY_DEST;
        else if (data == std::nullopt)
            resourceState = D3D12_RESOURCE_STATE_COMMON;

        m_d3dDevice->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(heapType),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(bufferSizeInByte),
            resourceState,
            nullptr,
            IID_PPV_ARGS(resourcePointer.ReleaseAndGetAddressOf()));

        if (data != std::nullopt) {
            UINT8* pDataBegin;

            resourcePointer->Map(0, &readRange, reinterpret_cast<void**>(&pDataBegin));
            memcpy(pDataBegin, data->data(), data->size());
            resourcePointer->Unmap(0, nullptr);
        }

    }

    void D3D12RHIContext::CopyForReadBack(Microsoft::WRL::ComPtr<ID3D12Resource>& readbackInput, Microsoft::WRL::ComPtr<ID3D12Resource>& readbackOutput) {
        {
            D3D12_RESOURCE_BARRIER outputBufferResourceBarrier
            {
                CD3DX12_RESOURCE_BARRIER::Transition(
                    readbackInput.Get(),
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_SOURCE)
            };
            m_commandList->ResourceBarrier(1, &outputBufferResourceBarrier);
        }

        m_commandList->CopyResource(readbackOutput.Get(), readbackInput.Get());
    }

    void D3D12RHIContext::CPUReadBack(Microsoft::WRL::ComPtr<ID3D12Resource> resourcePointer, std::vector<float>& outputData, unsigned int outputSizeInByte) {

        D3D12_RANGE readbackBufferRange{ 0, outputSizeInByte };
        float* pReadbackBufferData{};

        resourcePointer->Map(0, &readbackBufferRange, reinterpret_cast<void**>(&pReadbackBufferData));
        memcpy(outputData.data(), pReadbackBufferData, outputSizeInByte);
        D3D12_RANGE emptyRange{ 0, 0 };
        resourcePointer->Unmap
        (
            0,
            &emptyRange
        );
    }
}

void RHIDmlInitialize(){

}



void RHIDmlExecution(){


}