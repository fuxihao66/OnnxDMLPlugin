#pragma once
#include "../helper/pch.h"
// global resources for all dml model
namespace ODI{

#define MAX_DESCRIPTOR_COUNT 100000
    struct ModelInfo {
        unsigned int                                        modelInputNum;
        unsigned int                                        modelOutputNum;
        /*unsigned int                                        descriptorCPUOffset;
        unsigned int                                        descriptorGPUOffset;*/
        std::vector<DML_BINDING_DESC>                       inputBindings;
        Microsoft::WRL::ComPtr<IDMLCompiledOperator>        dmlGraph;
        Microsoft::WRL::ComPtr<IDMLOperatorInitializer>     dmlOpInitializer;
        Microsoft::WRL::ComPtr<ID3D12Resource>              modelPersistentResource;
        Microsoft::WRL::ComPtr<ID3D12Resource>              modelTemporaryResource;
        Microsoft::WRL::ComPtr<ID3D12Resource>              modelOperatorWeights;
        Microsoft::WRL::ComPtr<IDMLBindingTable>            dmlBindingTable;
        ModelInfo() : modelInputNum(0), modelOutputNum(0), dmlGraph(nullptr), dmlOpInitializer(nullptr),
            modelPersistentResource(nullptr), modelTemporaryResource(nullptr), modelOperatorWeights(nullptr), dmlBindingTable(nullptr)
        {}
    };

    class DescriptorHeapWrapper {
    private:
        void Create(
            ID3D12Device* pDevice,
            const D3D12_DESCRIPTOR_HEAP_DESC* pDesc)
        {
            assert(pDesc != nullptr);

            m_desc = *pDesc;
            m_increment = pDevice->GetDescriptorHandleIncrementSize(pDesc->Type);

            if (pDesc->NumDescriptors == 0)
            {
                m_pHeap.Reset();
                m_hCPU.ptr = 0;
                m_hGPU.ptr = 0;
            }
            else
            {
                pDevice->CreateDescriptorHeap(
                    pDesc,
                    IID_PPV_ARGS(m_pHeap.ReleaseAndGetAddressOf()));

                m_hCPU = m_pHeap->GetCPUDescriptorHandleForHeapStart();

                if (pDesc->Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE)
                    m_hGPU = m_pHeap->GetGPUDescriptorHandleForHeapStart();

            }
        }

    public:
        DescriptorHeapWrapper(
            _In_ ID3D12Device* device,
            D3D12_DESCRIPTOR_HEAP_TYPE type,
            D3D12_DESCRIPTOR_HEAP_FLAGS flags,
            size_t count) :
            m_desc{},
            m_hCPU{},
            m_hGPU{},
            m_increment(0)
        {
            if (count > UINT32_MAX)
                throw std::exception("Too many descriptors");

            D3D12_DESCRIPTOR_HEAP_DESC desc = {};
            desc.Flags = flags;
            desc.NumDescriptors = static_cast<UINT>(count);
            desc.Type = type;
            Create(device, &desc);
        }
        ID3D12DescriptorHeap* Heap() {
            return m_pHeap.Get();
        }
        D3D12_CPU_DESCRIPTOR_HANDLE GetCpuHandle(_In_ size_t index) const
        {
            assert(m_pHeap != nullptr);
            if (index >= m_desc.NumDescriptors)
            {
                throw std::out_of_range("D3DX12_CPU_DESCRIPTOR_HANDLE");
            }

            D3D12_CPU_DESCRIPTOR_HANDLE handle;
            handle.ptr = static_cast<SIZE_T>(m_hCPU.ptr + UINT64(index) * UINT64(m_increment));
            return handle;
        }
        D3D12_GPU_DESCRIPTOR_HANDLE GetGpuHandle(_In_ size_t index) const
        {
            assert(m_pHeap != nullptr);
            if (index >= m_desc.NumDescriptors)
            {
                throw std::out_of_range("D3DX12_GPU_DESCRIPTOR_HANDLE");
            }
            assert(m_desc.Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);

            D3D12_GPU_DESCRIPTOR_HANDLE handle;
            handle.ptr = m_hGPU.ptr + UINT64(index) * UINT64(m_increment);
            return handle;
        }
    private:
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap>    m_pHeap;
        D3D12_DESCRIPTOR_HEAP_DESC                      m_desc;
        D3D12_CPU_DESCRIPTOR_HANDLE                     m_hCPU;
        D3D12_GPU_DESCRIPTOR_HANDLE                     m_hGPU;
        uint32_t                                        m_increment;
    };


    class D3D12RHIContext{
    public:
        D3D12RHIContext();
        ~D3D12RHIContext();
    private:
        void CreateDeviceResources(); // only used for debug
        void CreateDMLResources();
        //void InitializeDMLResource();
    public:
        void CreateBufferFromData(Microsoft::WRL::ComPtr<ID3D12Resource>, const std::optional<std::vector<uint16_t>> data, unsigned int bufferSizeInByte, bool needReadback = false); // only used for debug
        void ForceCPUSync(); // only used for debug
        void CPUReadBack(ID3D12Resource* resourcePointer, std::vector<uint16_t>& outputData, unsigned int outputSizeInByte);
        void CopyForReadBack(ID3D12Resource* readbackInput, ID3D12Resource* readbackOutput);
        void InitializeNewModel(const std::wstring& path_to_onnx, const std::string& modelName);
        void RunDMLInfer(const std::map<std::string, ID3D12Resource*> inputs, ID3D12Resource* outputs, const std::string& modelName);
    private:
        Microsoft::WRL::ComPtr<IDMLDevice>              m_dmlDevice;
        Microsoft::WRL::ComPtr<IDMLCommandRecorder>     m_dmlCommandRecorder;
        //Microsoft::WRL::ComPtr<DirectX::DescriptorHeap>        m_dmlDescriptorHeap;
        std::unique_ptr<DescriptorHeapWrapper>        m_dmlDescriptorHeap;
        
        std::unordered_map<std::string, ModelInfo>      m_modelNameToResourceInfo; // model info (for supporting different models)
        UINT                                            m_currentDescriptorTopIndex;
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
        D3D_FEATURE_LEVEL                                   m_d3dFeatureLevel;

    };

    

    // in rhi thread


    


}

// void RHIDmlInitialize(){

// }



// void RHIDmlExecution(){

// }