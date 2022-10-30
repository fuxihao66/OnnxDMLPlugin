#pragma once
#include "../helper/pch.h"
// global resources for all dml model
namespace ODI{

    class D3D12RHIContext{
        
    private:
        void CreateDeviceResources(); // only used for debug
        void CreateDMLResources(const std::wstring& path_to_onnx);
        void InitializeDMLResource();
    public:
        void CreateBufferFromData(Microsoft::WRL::ComPtr<ID3D12Resource>, const std::optional<std::vector<uint16_t>> data, unsigned int bufferSizeInByte, bool needReadback = false); // only used for debug
        void ForceCPUSync(); // only used for debug
        void CPUReadBack(std::vector<char>& cpuImgData);
        void InitializeNewModel(const std::wstring& path_to_onnx, const std::string& modelName);
        void RunDMLInfer(const std::vector<ID3D12Resource*>& inputs, ID3D12Resource* outputs, const std::string& modelName);
    private:
        Microsoft::WRL::ComPtr<IDMLDevice>              m_dmlDevice;
        Microsoft::WRL::ComPtr<IDMLCommandRecorder>     m_dmlCommandRecorder;
        std::unique_ptr<DirectX::DescriptorHeap>        m_dmlDescriptorHeap;
        
        std::unordered_map<std::string, ModelInfo>      m_modelNameToResourceInfo; // model info (for supporting different models)
    private:// only needed for debug
        D3D_FEATURE_LEVEL m_d3dMinFeatureLevel;
        DWORD                                               m_dxgiFactoryFlags;

        Microsoft::WRL::ComPtr<ID3D12Device>                m_d3dDevice;
        Microsoft::WRL::ComPtr<ID3D12CommandQueue>          m_commandQueue;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList>   m_commandList;
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator>      m_commandAllocator;
        Microsoft::WRL::ComPtr<IDXGIFactory4>               m_dxgiFactory;
        Microsoft::WRL::ComPtr<ID3D12Fence>                 m_fence;
        UINT64                                              m_fenceValue;
        Microsoft::WRL::Wrappers::Event                     m_fenceEvent;
   
    };

    struct ModelInfo{
        IDMLCompiledOperator*       dmlGraph; 
        IDMLOperatorInitializer*    dmlOpInitializer;
        ID3D12Resource*             modelPersistentResource;
        ID3D12Resource*             modelTemporaryResource;
        ID3D12Resource*             modelOperatorWeights;
        IDMLBindingTable*           dmlBindingTable;
    };

    // in rhi thread


    


}

// void RHIDmlInitialize(){

// }



// void RHIDmlExecution(){

// }