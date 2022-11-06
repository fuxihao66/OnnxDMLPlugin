// class ENGINE_API FSceneViewFamily{
    //FOnnxModelManager* m_modelManager;
// }
enum ONNX_MODEL_RESULT{
    O_OK,
    O_ERROR
}
class FOnnxModelManager{
public:
    static FOnnxModelManager* CreateModelManager(){
        if (m_manager == nullptr){
            free(m_manager);
            m_manager = FOnnxModelManager();
        }
    }
    FOnnxModel * GetModelFromFile(const std::wstring& path_to_onnx, const std::string& model_name) {
        if (m_modelMapping.count(model_name) == 0){
            m_modelMapping[model_name] = new FOnnxModel(path_to_onnx);
        }
        return m_modelMapping[model_name];
    }
    FOnnxModel * GetModelFromName(const std::string& model_name) const {
        if (m_modelMapping.count(model_name))
            return m_modelMapping[model_name];
        else
            return nullptr;    
    }
    void AddModelFromFile(const std::wstring& path_to_onnx, const std::string& model_name){
        if (m_modelMapping.count(model_name) == 0)
            m_modelMapping[model_name] = new FOnnxModel(path_to_onnx);
    }
    FOnnxModelManager() = default;
    virtual ~FOnnxModelManager(){
        free(m_manager);
    }
private:
    std::unordered_map<std::string, FOnnxModel*> m_modelMapping;
    static FOnnxModelManager * m_manager;
}
class FOnnxModel{

private:
    bool uploadWeightViaGraphBuilder;
    std::wstring m_model_filename;
    std::string m_model_name;
public:
    
    // bool IsNewModel(const std::wstring& path_to_onnx){
    //     if (m_model_filename == path_to_onnx)
    //         return false;
    //     else
    //         return true;
    // }
    FOnnxModel(const std::wstring& path_to_onnx)
        : uploadWeightViaGraphBuilder(true)
    {

    }
    bool IfUploadWeightsViaGraphBuilder() const{
        return uploadWeightViaGraphBuilder;
    }
    std::string GetModelName() const{
        return m_model_name;
    }
};



// in game thread

ENQUEUE_RENDER_COMMAND(FInitFXSystemCommand)(
[OnnxModel](FRHICommandListImmediate& RHICmdList)
{
    if (OnnxModel->IfUploadWeightsViaGraphBuilder())
        UploadNetworkResource_RenderThread(RHICmdList, OnnxModel);
    InitializeResources_RenderThread(RHICmdList, OnnxModel);
    InitializeOperators_RenderThread(RHICmdList, OnnxModel);
});

ONNX_MODEL_RESULT UploadNetworkResource_RenderThread(RHICmdList, OnnxModel){
    if (OnnxModel->IfUploadWeightsViaGraphBuilder())
        return ONNX_MODEL_RESULT::O_ERROR;
    FRDGBuilder GraphBuilder(RHICmdList, RDG_EVENT_NAME("Upload Model %s Weights To GPU", OnnxModel->GetName()));

    // Set parameters
    const TRefCountPtr<FRDGPooledBuffer>& PooledBuffer = InputTensor.GetPooledBuffer();
    // FRDGBufferRef InputBufferRef = GraphBuilder.RegisterExternalBuffer(PooledBuffer);
    FRDGBufferRef WeightsBufferRef = GraphBuilder.CreateBuffer();
    
    OnnxModel.SetNetworkWeightsHandle(WeightsBufferRef);
    FUploadTensorParameters* UploadParameters = GraphBuilder.AllocParameters<FUploadTensorParameters>();

    UploadParameters->Input = InputBufferRef;

    GraphBuilder.AddPass(
        RDG_EVENT_NAME("UNeuralNetwork-UploadTensor-%s", *InputTensor.GetName()),
        FUploadTensorParameters::FTypeInfo::GetStructMetadata(),
        UploadParameters,
        ERDGPassFlags::Copy | ERDGPassFlags::NeverCull,
        [UploadParameters](FRHICommandListImmediate& RHICmdList)
        {
            FRHIBuffer* InputBuffer = UploadParameters->Input->GetRHI();

            // NOTE: We're using UAVMask to trigger the UAV barrier in RDG
            RHICmdList.Transition(FRHITransitionInfo(InputBuffer, ERHIAccess::CopyDest, ERHIAccess::UAVMask));
        }
    );

    GraphBuilder.Execute();
}

// interface in render thread
// implementation is in D3D12RHI
void InitializeResources_RenderThread(){
    GraphBuilder.AddPass(
			RDG_EVENT_NAME("NN Inference"),
			PassParameters,
			ERDGPassFlags::Compute,
			[PassParameters, InputTensor, textureWidth, textureHeight](FRHICommandList& RHICmdList)
			{
				// FDLSSArguments DLSSArguments(DLSSState);
                FCustomNNArguments NNArguments;
				FMemory::Memzero(&NNArguments, sizeof(NNArguments));

				// We reinitialize each frame to deal with changing parameters, most times, this will be a no-op
				RHICmdList.InitializeInference(NNArguments);  // TODO: 调用的是method.inl定义的接口
			});
}
void InitializeOperators_RenderThread(){
    GraphBuilder.AddPass(
			RDG_EVENT_NAME("NN Inference"),
			PassParameters,
			ERDGPassFlags::Compute,
			[PassParameters, InputTensor, textureWidth, textureHeight](FRHICommandList& RHICmdList)
			{
				// FDLSSArguments DLSSArguments(DLSSState);
                FCustomNNArguments NNArguments;
				FMemory::Memzero(&NNArguments, sizeof(NNArguments));

				// We reinitialize each frame to deal with changing parameters, most times, this will be a no-op
				RHICmdList.InitializeInference(NNArguments);  // TODO: 调用的是method.inl定义的接口
			});
}
void Execute_RenderThread(){

    if (model->IfUploadWeightsViaGraphBuilder()){
        GraphBuilder.RegisterExternalBuffer();
    }
    GraphBuilder.AddPass(
			RDG_EVENT_NAME("NN Inference"),
			PassParameters,
			ERDGPassFlags::Compute,
			[PassParameters, InputTensor, textureWidth, textureHeight](FRHICommandList& RHICmdList)
			{
				// FDLSSArguments DLSSArguments(DLSSState);
                FCustomNNArguments NNArguments;
				FMemory::Memzero(&NNArguments, sizeof(NNArguments));

                if (model->IfUploadWeightsViaGraphBuilder()){
                    NNArguments.Weights = ;
                }

				// We reinitialize each frame to deal with changing parameters, most times, this will be a no-op
				RHICmdList.ExecuteInference(NNArguments);  // TODO: 调用的是method.inl定义的接口
			});
}

// 如果需要修改渲染管线，则只需要在SceneRendering.cpp加入初始化，然后在创建renderer的时候把网络作为参数传进去。