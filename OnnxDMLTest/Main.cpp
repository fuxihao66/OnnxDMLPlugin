#include <iostream>
#include <vector>
#include <windows.h>

#include "helper/pch.h"
#include "Common/Float16Compressor.h"
#define TJE_IMPLEMENTATION
#include "thirdParty/tiny_jpeg.h"
#include "thirdParty/jpeg_decoder.h"
#include "OnnxDMLCore/OnnxDMLRHIModule.h"
// #include "OnnxParser.h"

void Uint8ToHalfCHWWithoutNormalization(const std::vector<uint8_t>& input, std::vector<uint16_t>& output, int width, int height) {
    output.resize(input.size());

    int index = 0;
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                output[index++] = Float16Compressor::compress((float)input[y * width * 3 + x * 3 + c]);
            }
        }
    }
}
void HalfCHW2Uint8WithoutNormalization(const std::vector<uint16_t>& input, std::vector<uint8_t>& output, int width, int height) {
    output.resize(input.size());

    int index = 0;
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                output[y * width * 3 + x * 3 + c] = static_cast<uint8_t>(Float16Compressor::decompress(input[index++]));
            }
        }
    }
}

void Uint8ToHalfCHW(const std::vector<uint8_t>& input, std::vector<uint16_t>& output, int width, int height){
    output.resize(input.size());

    int index = 0;
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                output[index++] = Float16Compressor::compress((float)input[y * width * 3 + x * 3 + c] / 255.f);
            }
        }
    }
}
void HalfCHW2Uint8(const std::vector<uint16_t>& input, std::vector<uint8_t>& output, int width, int height) {
    output.resize(input.size());

    int index = 0;
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                output[y * width * 3 + x * 3 + c] = static_cast<uint8_t>(Float16Compressor::decompress(input[index++]) * 255.f);
            }
        }
    }
}
bool SaveImageToFile(const std::vector<uint8_t>& imgData, const std::string& fileName, unsigned int width, unsigned int height){
    
    tje_encode_to_file(fileName.c_str(), height, width, 3, imgData.data());
    return true;
}
bool LoadImageFromFile(std::vector<uint8_t>& data, const std::string& fileName, int& width, int & height){
    FILE *f = fopen(fileName.c_str(), "rb");
    if (!f) { return false; }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    unsigned char *buf = (unsigned char*)malloc(size);
    fseek(f, 0, SEEK_SET);
    size_t read = fread(buf, 1, size, f);
    fclose(f);
    Jpeg::Decoder decoder(buf, size);
    if (decoder.GetResult() != Jpeg::Decoder::OK) {
        static const std::vector<std::string> error_msgs = { "OK", "NotAJpeg", "Unsupported", "OutOfMemory", "InternalError", "SyntaxError", "Internal_Finished" };
        std::cout << "Error decoding the input file " << error_msgs[decoder.GetResult()] << std::endl;
        return false;
    }
    if (!decoder.IsColor()) {
        std::cout << "Need a color image for this demo" << std::endl;
        return false;
    }
    width = decoder.GetWidth();
    height = decoder.GetHeight();
    data.resize(width*height*3);
    std::memcpy(data.data(), decoder.GetImage(), data.size());
    return true;
}
void testModel()
{
    ODI::D3D12RHIContext context;

    /*ID3D12Resource * modelInput;
    ID3D12Resource * modelOutput;*/

    Microsoft::WRL::ComPtr<ID3D12Resource> modelInput;
    Microsoft::WRL::ComPtr<ID3D12Resource> modelOutput;
    Microsoft::WRL::ComPtr<ID3D12Resource> readbackOutput;


    std::vector<uint8_t> jpgInputData;
    int width, height;
    LoadImageFromFile(jpgInputData, "../data/testimg.jpg", width, height);

    std::vector<uint16_t> inputData;
    Uint8ToHalfCHW(jpgInputData, inputData, width, height);
    auto bufferSize = width * height * 3 * sizeof(uint16_t);
    context.CreateBufferFromData(modelInput, std::optional<std::vector<uint16_t>>{inputData}, bufferSize); // buffer for inference
    context.CreateBufferFromData(modelOutput, std::nullopt, bufferSize);
    context.CreateBufferFromData(readbackOutput, std::nullopt, bufferSize, true);
    std::vector<uint16_t> cpuImageData;
    std::vector<uint8_t> jpgOutputData;


    context.Prepare();

    context.ParseUploadModelData(L"D:/candy-9.onnx", "Candy");
    context.InitializeNewModel("Candy");
    /*context.RunDMLInfer(std::map<std::string, ID3D12Resource*>{ {"input1", modelInput.Get()} }, modelOutput.Get(), "Candy");
    context.CopyForReadBack(modelOutput.Get(), readbackOutput.Get());
    context.ForceCPUSync();

    context.CPUReadBack(readbackOutput.Get(), cpuImageData, bufferSize);

    HalfCHW2Uint8(cpuImageData, jpgOutputData, width, height);
    SaveImageToFile(jpgOutputData, "testOutput.png", width, height);*/

}


void testGather0() {
    ODI::D3D12RHIContext context;

    Microsoft::WRL::ComPtr<ID3D12Resource> modelInputUpload; //TODO: input and output requires alignment?
    Microsoft::WRL::ComPtr<ID3D12Resource> modelInput;
    Microsoft::WRL::ComPtr<ID3D12Resource> modelOutput;
    Microsoft::WRL::ComPtr<ID3D12Resource> readbackOutput;

    std::vector<uint16_t> inputData;

    inputData.push_back(Float16Compressor::compress(0.1f));
    inputData.push_back(Float16Compressor::compress(0.2f));
    inputData.push_back(Float16Compressor::compress(0.3f));
    inputData.push_back(Float16Compressor::compress(0.4f));

    auto inputSize = 4 * sizeof(uint16_t);
    auto outputSize = 6 * sizeof(uint16_t);

    context.Prepare(); // reset command list

    //context.CreateBufferFromData(modelInput, std::optional<std::vector<uint16_t>>{inputData}, inputSize); // buffer for inference
    context.CreateBufferFromDataSubresource(modelInput, modelInputUpload, inputData, inputSize); // make sure model input is on default heap
    context.CreateBufferFromData(modelOutput, std::nullopt, outputSize);
    context.CreateBufferFromData(readbackOutput, std::nullopt, outputSize, true);
    std::vector<uint16_t> cpuImageData;

    //context.ParseUploadModelData(L"../model/GeneratedOnnx/FP16/GatherTest0-fp16-13.onnx", "GatherTest");
    context.ParseUploadModelData(L"../model/GeneratedOnnx/FP32/GatherTest0-13.onnx", "GatherTest");
    context.ForceCPUSync();
    context.Prepare();

    context.InitializeNewModel("GatherTest");
    //context.ForceCPUSync();
    context.Prepare();

    context.RunDMLInfer(std::map<std::string, ID3D12Resource*>{ {"input1", modelInput.Get()} }, modelOutput.Get(), "GatherTest");
    context.CopyForReadBack(modelOutput.Get(), readbackOutput.Get());
    context.ForceCPUSync();

    context.CPUReadBack(readbackOutput.Get(), cpuImageData, outputSize);
    // output should be [0.4f, 0.2f, 0.4f, 0.1f, 0.3f]
    for (int i = 0; i < 6; i++) {
        std::cout << Float16Compressor::decompress(cpuImageData[i]) << " ";
    }
}

void testGather1(bool useFp16) {
    ODI::D3D12RHIContext context;

    Microsoft::WRL::ComPtr<ID3D12Resource> modelInputUpload; //TODO: input and output requires alignment?
    Microsoft::WRL::ComPtr<ID3D12Resource> modelInput;
    Microsoft::WRL::ComPtr<ID3D12Resource> modelOutput;
    Microsoft::WRL::ComPtr<ID3D12Resource> readbackOutput;

    std::vector<uint16_t> inputData;

    inputData.push_back(Float16Compressor::compress(1.0f));
    inputData.push_back(Float16Compressor::compress(1.2f));
    inputData.push_back(Float16Compressor::compress(2.3f));
    inputData.push_back(Float16Compressor::compress(3.4f));
    inputData.push_back(Float16Compressor::compress(4.5f));
    inputData.push_back(Float16Compressor::compress(5.7f));

    auto inputSize = 6 * sizeof(uint16_t);
    auto outputSize = 8 * sizeof(uint16_t);

    context.Prepare(); // reset command list

    //context.CreateBufferFromData(modelInput, std::optional<std::vector<uint16_t>>{inputData}, inputSize); // buffer for inference
    context.CreateBufferFromDataSubresource(modelInput, modelInputUpload, inputData, inputSize); // make sure model input is on default heap
    context.CreateBufferFromData(modelOutput, std::nullopt, outputSize);
    context.CreateBufferFromData(readbackOutput, std::nullopt, outputSize, true);
    std::vector<uint16_t> cpuImageData;

    if (useFp16)
        context.ParseUploadModelData(L"../model/GeneratedOnnx/FP16/GatherTest1-fp16-13.onnx", "GatherTest");
    else
        context.ParseUploadModelData(L"../model/GeneratedOnnx/FP32/GatherTest1-13.onnx", "GatherTest");
    context.ForceCPUSync();
    context.Prepare();

    context.InitializeNewModel("GatherTest");
    //context.ForceCPUSync();
    context.Prepare();

    context.RunDMLInfer(std::map<std::string, ID3D12Resource*>{ {"input1", modelInput.Get()} }, modelOutput.Get(), "GatherTest");
    context.CopyForReadBack(modelOutput.Get(), readbackOutput.Get());
    context.ForceCPUSync();

    context.CPUReadBack(readbackOutput.Get(), cpuImageData, outputSize);
    for (int i = 0; i < 8; i++) {
        std::cout << Float16Compressor::decompress(cpuImageData[i]) << " ";
    }
}

void testGather2(bool useFp16) {
    ODI::D3D12RHIContext context;

    Microsoft::WRL::ComPtr<ID3D12Resource> modelInputUpload; //TODO: input and output requires alignment?
    Microsoft::WRL::ComPtr<ID3D12Resource> modelInput;
    Microsoft::WRL::ComPtr<ID3D12Resource> modelOutput;
    Microsoft::WRL::ComPtr<ID3D12Resource> readbackOutput;

    std::vector<uint16_t> inputData;

    inputData.push_back(Float16Compressor::compress(1.0f));
    inputData.push_back(Float16Compressor::compress(1.2f));
    inputData.push_back(Float16Compressor::compress(1.9f));
    inputData.push_back(Float16Compressor::compress(2.3f));
    inputData.push_back(Float16Compressor::compress(3.4f));
    inputData.push_back(Float16Compressor::compress(3.9f));
    inputData.push_back(Float16Compressor::compress(4.5f));
    inputData.push_back(Float16Compressor::compress(5.7f));
    inputData.push_back(Float16Compressor::compress(5.9f));

    auto inputSize = 9 * sizeof(uint16_t);
    auto outputSize = 6 * sizeof(uint16_t);

    context.Prepare(); // reset command list

    //context.CreateBufferFromData(modelInput, std::optional<std::vector<uint16_t>>{inputData}, inputSize); // buffer for inference
    context.CreateBufferFromDataSubresource(modelInput, modelInputUpload, inputData, inputSize); // make sure model input is on default heap
    context.CreateBufferFromData(modelOutput, std::nullopt, outputSize);
    context.CreateBufferFromData(readbackOutput, std::nullopt, outputSize, true);
    std::vector<uint16_t> cpuImageData;

    if (useFp16)
        context.ParseUploadModelData(L"../model/GeneratedOnnx/FP16/GatherTest2-fp16-13.onnx", "GatherTest");
    else
        context.ParseUploadModelData(L"../model/GeneratedOnnx/FP32/GatherTest2-13.onnx", "GatherTest");

    context.ForceCPUSync();
    context.Prepare();

    context.InitializeNewModel("GatherTest");
    //context.ForceCPUSync();
    context.Prepare();

    context.RunDMLInfer(std::map<std::string, ID3D12Resource*>{ {"input1", modelInput.Get()} }, modelOutput.Get(), "GatherTest");
    context.CopyForReadBack(modelOutput.Get(), readbackOutput.Get());
    context.ForceCPUSync();

    context.CPUReadBack(readbackOutput.Get(), cpuImageData, outputSize);
    for (int i = 0; i < 6; i++) {
        std::cout << Float16Compressor::decompress(cpuImageData[i]) << " ";
    }
}

void UnitTest(const std::wstring& onnxFile, const std::string& modelName, 
              const unsigned int inputSize, const unsigned int outputSize, 
              const std::vector<uint16_t>& inputData, std::vector<uint16_t>& cpuImageData) {
    ODI::D3D12RHIContext context;

    Microsoft::WRL::ComPtr<ID3D12Resource> modelInputUpload; //TODO: input and output requires alignment?
    Microsoft::WRL::ComPtr<ID3D12Resource> modelInput;
    Microsoft::WRL::ComPtr<ID3D12Resource> modelOutput;
    Microsoft::WRL::ComPtr<ID3D12Resource> readbackOutput;

    //std::vector<uint16_t> inputData;

    /*inputData.push_back(Float16Compressor::compress(1.0f));
    inputData.push_back(Float16Compressor::compress(1.2f));
    inputData.push_back(Float16Compressor::compress(2.3f));
    inputData.push_back(Float16Compressor::compress(3.4f));
    inputData.push_back(Float16Compressor::compress(4.5f));
    inputData.push_back(Float16Compressor::compress(5.7f));*/

    //auto inputSize = 6 * sizeof(uint16_t);
    //auto outputSize = 8 * sizeof(uint16_t);

    context.Prepare(); // reset command list

    //context.CreateBufferFromData(modelInput, std::optional<std::vector<uint16_t>>{inputData}, inputSize); // buffer for inference
    context.CreateBufferFromDataSubresource(modelInput, modelInputUpload, inputData, inputSize); // make sure model input is on default heap
    context.CreateBufferFromData(modelOutput, std::nullopt, outputSize);
    context.CreateBufferFromData(readbackOutput, std::nullopt, outputSize, true);
    //std::vector<uint16_t> cpuImageData;

    context.ParseUploadModelData(onnxFile, modelName);
    context.ForceCPUSync();
    context.Prepare();

    context.InitializeNewModel(modelName);
    //context.ForceCPUSync();
    context.Prepare();

    context.RunDMLInfer(std::map<std::string, ID3D12Resource*>{ {"input1", modelInput.Get()} }, modelOutput.Get(), modelName);
    context.CopyForReadBack(modelOutput.Get(), readbackOutput.Get());
    context.ForceCPUSync();

    context.CPUReadBack(readbackOutput.Get(), cpuImageData, outputSize);
    // output should be [0.4f, 0.2f, 0.4f, 0.1f, 0.3f]
    
}

void UnitTest2Params(const std::wstring& onnxFile, const std::string& modelName,
    const unsigned int inputSize, const unsigned int outputSize,
    const std::vector<uint16_t>& inputData0, const std::vector<uint16_t>& inputData1, std::vector<uint16_t>& cpuImageData) {
    ODI::D3D12RHIContext context;

    Microsoft::WRL::ComPtr<ID3D12Resource> modelInputUpload0; //TODO: input and output requires alignment?
    Microsoft::WRL::ComPtr<ID3D12Resource> modelInput0;
    Microsoft::WRL::ComPtr<ID3D12Resource> modelInputUpload1; //TODO: input and output requires alignment?
    Microsoft::WRL::ComPtr<ID3D12Resource> modelInput1;
    Microsoft::WRL::ComPtr<ID3D12Resource> modelOutput;
    Microsoft::WRL::ComPtr<ID3D12Resource> readbackOutput;

    context.Prepare(); // reset command list

    //context.CreateBufferFromData(modelInput, std::optional<std::vector<uint16_t>>{inputData}, inputSize); // buffer for inference
    context.CreateBufferFromDataSubresource(modelInput0, modelInputUpload0, inputData0, inputSize); // make sure model input is on default heap
    context.CreateBufferFromDataSubresource(modelInput1, modelInputUpload1, inputData1, inputSize); // make sure model input is on default heap
    context.CreateBufferFromData(modelOutput, std::nullopt, outputSize);
    context.CreateBufferFromData(readbackOutput, std::nullopt, outputSize, true);
    //std::vector<uint16_t> cpuImageData;

    context.ParseUploadModelData(onnxFile, modelName);
    context.ForceCPUSync();
    context.Prepare();

    context.InitializeNewModel(modelName);
    //context.ForceCPUSync();
    context.Prepare();

    context.RunDMLInfer(std::map<std::string, ID3D12Resource*>{ {"input0", modelInput0.Get()}, { "input1", modelInput1.Get() } }, modelOutput.Get(), modelName);
    context.CopyForReadBack(modelOutput.Get(), readbackOutput.Get());
    context.ForceCPUSync();

    context.CPUReadBack(readbackOutput.Get(), cpuImageData, outputSize);
}

void UnitTest3Params(const std::wstring& onnxFile, const std::string& modelName,
    const unsigned int inputSize, const unsigned int otherInputSize, const unsigned int outputSize,
    const std::vector<uint16_t>& inputData0, const std::vector<uint16_t>& inputData1, const std::vector<uint16_t>& inputData2, std::vector<uint16_t>& cpuImageData) {
    ODI::D3D12RHIContext context;

    Microsoft::WRL::ComPtr<ID3D12Resource> modelInputUpload0; //TODO: input and output requires alignment?
    Microsoft::WRL::ComPtr<ID3D12Resource> modelInput0;
    Microsoft::WRL::ComPtr<ID3D12Resource> modelInputUpload1; //TODO: input and output requires alignment?
    Microsoft::WRL::ComPtr<ID3D12Resource> modelInput1;
    Microsoft::WRL::ComPtr<ID3D12Resource> modelInputUpload2; //TODO: input and output requires alignment?
    Microsoft::WRL::ComPtr<ID3D12Resource> modelInput2;
    Microsoft::WRL::ComPtr<ID3D12Resource> modelOutput;
    Microsoft::WRL::ComPtr<ID3D12Resource> readbackOutput;

    context.Prepare(); // reset command list

    //context.CreateBufferFromData(modelInput, std::optional<std::vector<uint16_t>>{inputData}, inputSize); // buffer for inference
    context.CreateBufferFromDataSubresource(modelInput0, modelInputUpload0, inputData0, inputSize); // make sure model input is on default heap
    context.CreateBufferFromDataSubresource(modelInput1, modelInputUpload1, inputData1, otherInputSize); // make sure model input is on default heap
    context.CreateBufferFromDataSubresource(modelInput2, modelInputUpload2, inputData2, otherInputSize); // make sure model input is on default heap
    context.CreateBufferFromData(modelOutput, std::nullopt, outputSize);
    context.CreateBufferFromData(readbackOutput, std::nullopt, outputSize, true);
    //std::vector<uint16_t> cpuImageData;

    context.ParseUploadModelData(onnxFile, modelName);
    context.ForceCPUSync();
    context.Prepare();

    context.InitializeNewModel(modelName);
    //context.ForceCPUSync();
    context.Prepare();

    context.RunDMLInfer(std::map<std::string, ID3D12Resource*>{ {"input0", modelInput0.Get()}, { "input1", modelInput1.Get() }, { "input2", modelInput2.Get() }  }, modelOutput.Get(), modelName);
    context.CopyForReadBack(modelOutput.Get(), readbackOutput.Get());
    context.ForceCPUSync();

    context.CPUReadBack(readbackOutput.Get(), cpuImageData, outputSize);
}

void INTest0() {
    std::vector<uint16_t> inputData;

    std::vector<uint16_t> inputDataScale = { Float16Compressor::compress(0.4f), Float16Compressor::compress(0.4f) };
    std::vector<uint16_t> inputDataBias = { Float16Compressor::compress(0.4f), Float16Compressor::compress(0.4f) };


    std::vector<uint16_t> cpuImageData;

    inputData.push_back(Float16Compressor::compress(-1.0f));
    inputData.push_back(Float16Compressor::compress(1.0f));
    inputData.push_back(Float16Compressor::compress(0.3f));
    inputData.push_back(Float16Compressor::compress(0.4f));
    inputData.push_back(Float16Compressor::compress(0.8f));
    inputData.push_back(Float16Compressor::compress(0.2f));
    inputData.push_back(Float16Compressor::compress(0.3f));
    inputData.push_back(Float16Compressor::compress(0.4f));
    /*inputData.push_back(Float16Compressor::compress(0.8f));
    inputData.push_back(Float16Compressor::compress(0.1f));
    inputData.push_back(Float16Compressor::compress(0.2f));
    inputData.push_back(Float16Compressor::compress(0.9f));*/
    
    UnitTest3Params(L"../model/GeneratedOnnx/FP16/InstanceNormalizationTest0-fp16-7.onnx", "TestIN",
        8 * sizeof(uint16_t), 2 * sizeof(uint16_t), 8 * sizeof(uint16_t), inputData, inputDataScale, inputDataBias, cpuImageData);
    for (int i = 0; i < 8; i++) {
        std::cout << Float16Compressor::decompress(cpuImageData[i]) << " ";
    }
}
void INTest1() {
    std::vector<uint16_t> inputData;

    std::vector<uint16_t> inputDataScale = { Float16Compressor::compress(0.4f), Float16Compressor::compress(0.4f), Float16Compressor::compress(0.4f) };
    std::vector<uint16_t> inputDataBias = { Float16Compressor::compress(0.4f), Float16Compressor::compress(0.4f), Float16Compressor::compress(0.4f) };


    std::vector<uint16_t> cpuImageData;

    inputData.push_back(Float16Compressor::compress(-1.0f));
    inputData.push_back(Float16Compressor::compress(1.0f));
    inputData.push_back(Float16Compressor::compress(0.3f));
    inputData.push_back(Float16Compressor::compress(0.4f));
    inputData.push_back(Float16Compressor::compress(0.8f));
    inputData.push_back(Float16Compressor::compress(0.2f));
    inputData.push_back(Float16Compressor::compress(0.3f));
    inputData.push_back(Float16Compressor::compress(0.4f));
    inputData.push_back(Float16Compressor::compress(0.8f));
    inputData.push_back(Float16Compressor::compress(0.1f));
    inputData.push_back(Float16Compressor::compress(0.2f));
    inputData.push_back(Float16Compressor::compress(0.9f));

    UnitTest3Params(L"../model/GeneratedOnnx/FP16/InstanceNormalizationTest1-fp16-7.onnx", "TestIN",
        12 * sizeof(uint16_t), 3 * sizeof(uint16_t), 12 * sizeof(uint16_t), inputData, inputDataScale, inputDataBias, cpuImageData);
    for (int i = 0; i < 12; i++) {
        std::cout << Float16Compressor::decompress(cpuImageData[i]) << " ";
    }
}

void PadTest0() {
    std::vector<uint16_t> inputData;
    std::vector<uint16_t> cpuImageData;

    inputData.push_back(Float16Compressor::compress(1.0f));
    inputData.push_back(Float16Compressor::compress(1.2f));
    inputData.push_back(Float16Compressor::compress(2.3f));
    inputData.push_back(Float16Compressor::compress(3.4f));
    inputData.push_back(Float16Compressor::compress(4.5f));
    inputData.push_back(Float16Compressor::compress(5.7f));
    //UnitTest(L"../model/GeneratedOnnx/FP16/PadTest0-fp16-13.onnx", "TestPad",
    UnitTest(L"../model/GeneratedOnnx/FP32/PadTest0-13.onnx", "TestPad",
    //UnitTest(L"../model/GeneratedOnnx/FP32/PadTest0-7.onnx", "TestPad",
    //UnitTest(L"../model/GeneratedOnnx/FP16/PadTest0-fp16-7.onnx", "TestPad",
        6 * sizeof(uint16_t), 12 * sizeof(uint16_t), inputData, cpuImageData);
    for (int i = 0; i < 12; i++) {
        std::cout << Float16Compressor::decompress(cpuImageData[i]) << " ";
    }
}
void PadTest1() {
    std::vector<uint16_t> inputData;
    std::vector<uint16_t> cpuImageData;

    inputData.push_back(Float16Compressor::compress(1.0f));
    inputData.push_back(Float16Compressor::compress(1.2f));
    inputData.push_back(Float16Compressor::compress(2.3f));
    inputData.push_back(Float16Compressor::compress(3.4f));
    inputData.push_back(Float16Compressor::compress(4.5f));
    inputData.push_back(Float16Compressor::compress(5.7f));
    UnitTest(L"../model/GeneratedOnnx/FP16/PadTest1-fp16-13.onnx", "TestPad",
    //UnitTest(L"../model/GeneratedOnnx/FP16/PadTest1-fp16-7.onnx", "TestPad",
        6 * sizeof(uint16_t), 12 * sizeof(uint16_t), inputData, cpuImageData);
    for (int i = 0; i < 12; i++) {
        std::cout << Float16Compressor::decompress(cpuImageData[i]) << " ";
    }
}
void PadTest2() {
    std::vector<uint16_t> inputData;
    std::vector<uint16_t> cpuImageData;

    inputData.push_back(Float16Compressor::compress(1.0f));
    inputData.push_back(Float16Compressor::compress(1.2f));
    inputData.push_back(Float16Compressor::compress(2.3f));
    inputData.push_back(Float16Compressor::compress(3.4f));
    inputData.push_back(Float16Compressor::compress(4.5f));
    inputData.push_back(Float16Compressor::compress(5.7f));
    UnitTest(L"../model/GeneratedOnnx/FP16/PadTest2-fp16-13.onnx", "TestPad",
    //UnitTest(L"../model/GeneratedOnnx/FP16/PadTest2-fp16-7.onnx", "TestPad",
        6 * sizeof(uint16_t), 12 * sizeof(uint16_t), inputData, cpuImageData);
    for (int i = 0; i < 12; i++) {
        std::cout << Float16Compressor::decompress(cpuImageData[i]) << " ";
    }
}
void SliceTest0() {
    std::vector<uint16_t> inputData;
    std::vector<uint16_t> cpuImageData;

    inputData.push_back(Float16Compressor::compress(1.0f));
    inputData.push_back(Float16Compressor::compress(2.0f));
    inputData.push_back(Float16Compressor::compress(3.0f));
    inputData.push_back(Float16Compressor::compress(4.0f));
    inputData.push_back(Float16Compressor::compress(5.0f));
    inputData.push_back(Float16Compressor::compress(6.0f));
    inputData.push_back(Float16Compressor::compress(7.0f));
    inputData.push_back(Float16Compressor::compress(8.0f));
    UnitTest(L"../model/GeneratedOnnx/FP16/SliceTest0-fp16-10.onnx", "TestSlice",
        8 * sizeof(uint16_t), 2 * sizeof(uint16_t), inputData, cpuImageData);
    for (int i = 0; i < 2; i++) {
        std::cout << Float16Compressor::decompress(cpuImageData[i]) << " ";
    }
}
void SliceTest1() {
    std::vector<uint16_t> inputData;
    std::vector<uint16_t> cpuImageData;

    inputData.push_back(Float16Compressor::compress(1.0f));
    inputData.push_back(Float16Compressor::compress(2.0f));
    inputData.push_back(Float16Compressor::compress(3.0f));
    inputData.push_back(Float16Compressor::compress(4.0f));
    inputData.push_back(Float16Compressor::compress(5.0f));
    inputData.push_back(Float16Compressor::compress(6.0f));
    inputData.push_back(Float16Compressor::compress(7.0f));
    inputData.push_back(Float16Compressor::compress(8.0f));
    UnitTest(L"../model/GeneratedOnnx/FP16/SliceTest1-fp16-10.onnx", "TestSlice",
        8 * sizeof(uint16_t), 3 * sizeof(uint16_t), inputData, cpuImageData);
    for (int i = 0; i < 3; i++) {
        std::cout << Float16Compressor::decompress(cpuImageData[i]) << " ";
    }
}

void ReluTest0() {
    std::vector<uint16_t> inputData;
    std::vector<uint16_t> cpuImageData;

    inputData.push_back(Float16Compressor::compress(-1.0f));
    inputData.push_back(Float16Compressor::compress(1.0f));
    inputData.push_back(Float16Compressor::compress(0.3f));
    inputData.push_back(Float16Compressor::compress(0.4f));
    UnitTest(L"../model/GeneratedOnnx/FP16/ReluTest0-fp16-13.onnx", "TestRelu",
             4 * sizeof(uint16_t), 4 * sizeof(uint16_t), inputData, cpuImageData);
    for (int i = 0; i < 4; i++) {
        std::cout << Float16Compressor::decompress(cpuImageData[i]) << " ";
    }
}

void ReluTest1() {
    std::vector<uint16_t> inputData;
    std::vector<uint16_t> cpuImageData;

    inputData.push_back(Float16Compressor::compress(-1.0f));
    inputData.push_back(Float16Compressor::compress(1.0f));
    inputData.push_back(Float16Compressor::compress(0.3f));
    inputData.push_back(Float16Compressor::compress(0.4f));
    inputData.push_back(Float16Compressor::compress(0.7f));
    inputData.push_back(Float16Compressor::compress(0.8f));
    UnitTest(L"../model/GeneratedOnnx/FP16/ReluTest1-fp16-13.onnx", "TestRelu",
        6 * sizeof(uint16_t), 6 * sizeof(uint16_t), inputData, cpuImageData);
    for (int i = 0; i < 6; i++) {
        std::cout << Float16Compressor::decompress(cpuImageData[i]) << " ";
    }
}

void CastTest0() {
    std::vector<uint16_t> inputData;
    std::vector<uint16_t> cpuImageData;

    inputData.push_back(Float16Compressor::compress(-1.0f));
    inputData.push_back(Float16Compressor::compress(1.0f));
    inputData.push_back(Float16Compressor::compress(0.3f));
    inputData.push_back(Float16Compressor::compress(0.4f));
    UnitTest(L"../model/GeneratedOnnx/FP16/CastTest0-fp16-9.onnx", "TestRelu",
        4 * sizeof(uint16_t), 8 * sizeof(uint16_t), inputData, cpuImageData);
    for (int i = 0; i < 4; i++) {
        unsigned int temp;
        memcpy(&temp, cpuImageData.data() + i * 2, sizeof(int));
        std::cout << temp << " ";
    }
}

void CastTest1() {
    std::vector<uint16_t> inputData;
    std::vector<uint16_t> cpuImageData;

    inputData.push_back(Float16Compressor::compress(-1.0f));
    inputData.push_back(Float16Compressor::compress(1.0f));
    inputData.push_back(Float16Compressor::compress(0.3f));
    inputData.push_back(Float16Compressor::compress(0.4f));
    inputData.push_back(Float16Compressor::compress(0.7f));
    inputData.push_back(Float16Compressor::compress(21.8f));
    UnitTest(L"../model/GeneratedOnnx/FP16/CastTest1-fp16-9.onnx", "TestRelu",
        6 * sizeof(uint16_t), 12 * sizeof(uint16_t), inputData, cpuImageData);
    for (int i = 0; i < 6; i++) {
        unsigned int temp;
        memcpy(&temp, cpuImageData.data() + i * 2, sizeof(int));
        std::cout << temp << " ";
    }
}

void ConvTest0() {
    std::vector<uint16_t> inputData;
    std::vector<uint16_t> cpuImageData;

    std::vector<uint8_t> jpgInputData;
    int width, height;
    LoadImageFromFile(jpgInputData, "../data/testimg.jpg", width, height);

    Uint8ToHalfCHW(jpgInputData, inputData, width, height);
    UnitTest(L"../model/GeneratedOnnx/FP16/ConvTest0-fp16-11.onnx", "TestConv",
        224 * 224 * 3 * sizeof(uint16_t), 224 * 224 * 3 * sizeof(uint16_t), inputData, cpuImageData);
    

    std::vector<uint8_t> jpgOutputData;

    HalfCHW2Uint8(cpuImageData, jpgOutputData, width, height);
    SaveImageToFile(jpgOutputData, "ConvTestOutput.png", width, height);
}

void ConvTest1() {
    std::vector<uint16_t> inputData;
    std::vector<uint16_t> cpuImageData;

    std::vector<uint8_t> jpgInputData;
    int width, height;
    LoadImageFromFile(jpgInputData, "../data/testimg.jpg", width, height);

    Uint8ToHalfCHW(jpgInputData, inputData, width, height);
    UnitTest(L"../model/GeneratedOnnx/FP32/ConvTest0-7.onnx", "TestConv",
        224 * 224 * 3 * sizeof(uint16_t), 224 * 224 * 3 * sizeof(uint16_t), inputData, cpuImageData);


    std::vector<uint8_t> jpgOutputData;

    HalfCHW2Uint8(cpuImageData, jpgOutputData, width, height);
    SaveImageToFile(jpgOutputData, "ConvTestOutput.png", width, height);
}

void UpsampleTest0() {
    std::vector<uint16_t> inputData;
    std::vector<uint16_t> cpuImageData;

    inputData.push_back(Float16Compressor::compress(-1.0f));
    inputData.push_back(Float16Compressor::compress(1.0f));
    inputData.push_back(Float16Compressor::compress(0.3f));
    inputData.push_back(Float16Compressor::compress(0.4f));
    UnitTest(L"../model/GeneratedOnnx/FP32/UpsampleTest0-9.onnx", "TestUpsample",
        4 * sizeof(uint16_t), 6 * sizeof(uint16_t), inputData, cpuImageData);
    for (int i = 0; i < 6; i++) {
        std::cout << Float16Compressor::decompress(cpuImageData[i]) << " ";
    }
}

void UpsampleTest1() {
    std::vector<uint16_t> inputData;
    std::vector<uint16_t> cpuImageData;

    inputData.push_back(Float16Compressor::compress(-1.0f));
    inputData.push_back(Float16Compressor::compress(1.0f));
    inputData.push_back(Float16Compressor::compress(0.3f));
    inputData.push_back(Float16Compressor::compress(0.4f));
    UnitTest(L"../model/GeneratedOnnx/FP32/UpsampleTest1-9.onnx", "TestUpsample",
        4 * sizeof(uint16_t), 6 * sizeof(uint16_t), inputData, cpuImageData);
    for (int i = 0; i < 6; i++) {
        std::cout << Float16Compressor::decompress(cpuImageData[i]) << " ";
    }
}

void UpsampleTest2() {
    std::vector<uint16_t> inputData;
    std::vector<uint16_t> cpuImageData;

    inputData.push_back(Float16Compressor::compress(1.0f));
    inputData.push_back(Float16Compressor::compress(2.0f));
    inputData.push_back(Float16Compressor::compress(3.0f));
    inputData.push_back(Float16Compressor::compress(4.0f));
    UnitTest(L"../model/GeneratedOnnx/FP32/UpsampleTest2-9.onnx", "TestUpsample",
        4 * sizeof(uint16_t), 6 * sizeof(uint16_t), inputData, cpuImageData);
    for (int i = 0; i < 6; i++) {
        std::cout << Float16Compressor::decompress(cpuImageData[i]) << " ";
    }
}

void UpsampleTest3() {
    std::vector<uint16_t> inputData;
    std::vector<uint16_t> cpuImageData;

    inputData.push_back(Float16Compressor::compress(-1.0f));
    inputData.push_back(Float16Compressor::compress(1.0f));
    inputData.push_back(Float16Compressor::compress(0.3f));
    inputData.push_back(Float16Compressor::compress(0.4f));
    inputData.push_back(Float16Compressor::compress(0.8f));
    inputData.push_back(Float16Compressor::compress(0.3f));
    inputData.push_back(Float16Compressor::compress(0.2f));
    inputData.push_back(Float16Compressor::compress(0.7f));
    inputData.push_back(Float16Compressor::compress(0.1f));
    inputData.push_back(Float16Compressor::compress(0.2f));
    inputData.push_back(Float16Compressor::compress(0.5f));
    inputData.push_back(Float16Compressor::compress(0.4f));
    UnitTest(L"../model/GeneratedOnnx/FP32/UpsampleTest3-9.onnx", "TestUpsample",
        12 * sizeof(uint16_t), 48 * sizeof(uint16_t), inputData, cpuImageData);
    for (int i = 0; i < 48; i++) {
        std::cout << Float16Compressor::decompress(cpuImageData[i]) << " ";
    }
}

void AddTest0() {
    std::vector<uint16_t> inputData0;
    std::vector<uint16_t> inputData1;
    std::vector<uint16_t> cpuImageData;

    inputData0.push_back(Float16Compressor::compress(-1.0f));
    inputData0.push_back(Float16Compressor::compress(1.0f));
    inputData0.push_back(Float16Compressor::compress(0.3f));
    inputData0.push_back(Float16Compressor::compress(0.4f));

    inputData1.push_back(Float16Compressor::compress(1.0f));
    inputData1.push_back(Float16Compressor::compress(-1.0f));
    inputData1.push_back(Float16Compressor::compress(0.3f));
    inputData1.push_back(Float16Compressor::compress(-0.4f));

    UnitTest2Params(L"../model/GeneratedOnnx/FP16/AddTest0-fp16-13.onnx", "TestRelu",
        4 * sizeof(uint16_t), 4 * sizeof(uint16_t), inputData0, inputData1, cpuImageData);
    for (int i = 0; i < 4; i++) {
        std::cout << Float16Compressor::decompress(cpuImageData[i]) << " ";
    }
}

void AddTest1() {
    std::vector<uint16_t> inputData0;
    std::vector<uint16_t> inputData1;
    std::vector<uint16_t> cpuImageData;

    inputData0.push_back(Float16Compressor::compress(-1.0f));
    inputData0.push_back(Float16Compressor::compress(1.0f));
    inputData0.push_back(Float16Compressor::compress(0.3f));
    inputData0.push_back(Float16Compressor::compress(0.4f));
    inputData0.push_back(Float16Compressor::compress(0.9f));
    inputData0.push_back(Float16Compressor::compress(0.7f));

    inputData1.push_back(Float16Compressor::compress(1.0f));
    inputData1.push_back(Float16Compressor::compress(-1.0f));
    inputData1.push_back(Float16Compressor::compress(0.3f));
    inputData1.push_back(Float16Compressor::compress(-0.4f));
    inputData1.push_back(Float16Compressor::compress(-0.2f));
    inputData1.push_back(Float16Compressor::compress(-0.3f));

    UnitTest2Params(L"../model/GeneratedOnnx/FP16/AddTest1-fp16-13.onnx", "TestRelu",
        6 * sizeof(uint16_t), 6 * sizeof(uint16_t), inputData0, inputData1, cpuImageData);
    for (int i = 0; i < 6; i++) {
        std::cout << Float16Compressor::decompress(cpuImageData[i]) << " ";
    }
}

void ConcatTest0() {
    std::vector<uint16_t> inputData0;
    std::vector<uint16_t> inputData1;
    std::vector<uint16_t> cpuImageData;

    inputData0.push_back(Float16Compressor::compress(-1.0f));
    inputData0.push_back(Float16Compressor::compress(1.0f));
    inputData0.push_back(Float16Compressor::compress(0.3f));
    inputData0.push_back(Float16Compressor::compress(0.4f));

    inputData1.push_back(Float16Compressor::compress(1.0f));
    inputData1.push_back(Float16Compressor::compress(-1.0f));
    inputData1.push_back(Float16Compressor::compress(0.3f));
    inputData1.push_back(Float16Compressor::compress(-0.4f));

    UnitTest2Params(L"../model/GeneratedOnnx/FP16/ConcatTest0-fp16-13.onnx", "TestRelu",
        4 * sizeof(uint16_t), 8 * sizeof(uint16_t), inputData0, inputData1, cpuImageData);
    for (int i = 0; i < 8; i++) {
        std::cout << Float16Compressor::decompress(cpuImageData[i]) << " ";
    }
}

void ConcatTest1() {
    std::vector<uint16_t> inputData0;
    std::vector<uint16_t> inputData1;
    std::vector<uint16_t> cpuImageData;

    inputData0.push_back(Float16Compressor::compress(-1.0f));
    inputData0.push_back(Float16Compressor::compress(1.0f));
    inputData0.push_back(Float16Compressor::compress(0.3f));
    inputData0.push_back(Float16Compressor::compress(0.4f));
    inputData0.push_back(Float16Compressor::compress(0.9f));
    inputData0.push_back(Float16Compressor::compress(0.7f));

    inputData1.push_back(Float16Compressor::compress(1.0f));
    inputData1.push_back(Float16Compressor::compress(-1.0f));
    inputData1.push_back(Float16Compressor::compress(0.3f));
    inputData1.push_back(Float16Compressor::compress(-0.4f));
    inputData1.push_back(Float16Compressor::compress(-0.2f));
    inputData1.push_back(Float16Compressor::compress(-0.3f));

    UnitTest2Params(L"../model/GeneratedOnnx/FP16/ConcatTest1-fp16-13.onnx", "TestRelu",
        6 * sizeof(uint16_t), 12 * sizeof(uint16_t), inputData0, inputData1, cpuImageData);
    for (int i = 0; i < 12; i++) {
        std::cout << Float16Compressor::decompress(cpuImageData[i]) << " ";
    }
}

void ConcatTest2() {
    std::vector<uint16_t> inputData0;
    std::vector<uint16_t> inputData1;
    std::vector<uint16_t> cpuImageData;

    inputData0.push_back(Float16Compressor::compress(-1.0f));
    inputData0.push_back(Float16Compressor::compress(1.0f));
    inputData0.push_back(Float16Compressor::compress(0.3f));
    inputData0.push_back(Float16Compressor::compress(0.4f));
    inputData0.push_back(Float16Compressor::compress(0.9f));
    inputData0.push_back(Float16Compressor::compress(0.7f));

    inputData1.push_back(Float16Compressor::compress(1.0f));
    inputData1.push_back(Float16Compressor::compress(-1.0f));
    inputData1.push_back(Float16Compressor::compress(0.3f));
    inputData1.push_back(Float16Compressor::compress(-0.4f));
    inputData1.push_back(Float16Compressor::compress(-0.2f));
    inputData1.push_back(Float16Compressor::compress(-0.3f));

    UnitTest2Params(L"../model/GeneratedOnnx/FP16/ConcatTest2-fp16-13.onnx", "TestRelu",
        6 * sizeof(uint16_t), 12 * sizeof(uint16_t), inputData0, inputData1, cpuImageData);
    for (int i = 0; i < 12; i++) {
        std::cout << Float16Compressor::decompress(cpuImageData[i]) << " ";
    }
}

void ConcatTest3() {
    std::vector<uint16_t> inputData0;
    std::vector<uint16_t> inputData1;
    std::vector<uint16_t> cpuImageData;

    inputData0.push_back(Float16Compressor::compress(-1.0f));
    inputData0.push_back(Float16Compressor::compress(1.0f));
    inputData0.push_back(Float16Compressor::compress(0.3f));
    inputData0.push_back(Float16Compressor::compress(0.4f));
    inputData0.push_back(Float16Compressor::compress(0.9f));
    inputData0.push_back(Float16Compressor::compress(0.7f));

    inputData1.push_back(Float16Compressor::compress(1.0f));
    inputData1.push_back(Float16Compressor::compress(-1.0f));
    inputData1.push_back(Float16Compressor::compress(0.3f));
    inputData1.push_back(Float16Compressor::compress(-0.4f));
    inputData1.push_back(Float16Compressor::compress(-0.2f));
    inputData1.push_back(Float16Compressor::compress(-0.3f));

    UnitTest2Params(L"../model/GeneratedOnnx/FP16/ConcatTest3-fp16-13.onnx", "TestRelu",
        6 * sizeof(uint16_t), 12 * sizeof(uint16_t), inputData0, inputData1, cpuImageData);
    for (int i = 0; i < 12; i++) {
        std::cout << Float16Compressor::decompress(cpuImageData[i]) << " ";
    }
}
void testReadBack() // legacy, fail 
{
    ODI::D3D12RHIContext context;

    /*ID3D12Resource * modelInput;
    ID3D12Resource * modelOutput;*/

    Microsoft::WRL::ComPtr<ID3D12Resource> modelInput;
    Microsoft::WRL::ComPtr<ID3D12Resource> modelOutput;
    Microsoft::WRL::ComPtr<ID3D12Resource> readbackOutput;


    std::vector<uint8_t> jpgInputData;
    int width, height;
    LoadImageFromFile(jpgInputData, "../data/testimg.jpg", width, height);

    std::vector<uint16_t> inputData;
    Uint8ToHalfCHW(jpgInputData, inputData, width, height);


    auto bufferSize = width * height * 3 * sizeof(uint16_t);
    context.CreateBufferFromData(modelInput, std::optional<std::vector<uint16_t>>{inputData}, bufferSize); // buffer for inference
    //context.CreateBufferFromData(modelOutput, std::nullopt, bufferSize);
    context.CreateBufferFromData(readbackOutput, std::nullopt, bufferSize, true);

    std::vector<uint16_t> cpuImageData;
    std::vector<uint8_t> jpgOutputData;

    context.Prepare();
    context.CopyForReadBack(modelInput.Get(), readbackOutput.Get());
    context.ForceCPUSync();

    context.CPUReadBack(readbackOutput.Get(), cpuImageData, bufferSize);
    
    HalfCHW2Uint8(cpuImageData, jpgOutputData, width, height);
    SaveImageToFile(jpgOutputData, "testOutput.png", width, height);

}


void FinalTest() {
    std::vector<uint16_t> inputData;
    std::vector<uint16_t> cpuImageData;

    std::vector<uint8_t> jpgInputData;
    int width, height;
    LoadImageFromFile(jpgInputData, "../data/testimg.jpg", width, height);

    Uint8ToHalfCHWWithoutNormalization(jpgInputData, inputData, width, height);
    UnitTest(L"../model/optimized-candy-9.onnx", "FinalTest",
        224 * 224 * 3 * sizeof(uint16_t), 224 * 224 * 3 * sizeof(uint16_t), inputData, cpuImageData);


    std::vector<uint8_t> jpgOutputData;

    HalfCHW2Uint8WithoutNormalization(cpuImageData, jpgOutputData, width, height);
    SaveImageToFile(jpgOutputData, "FinalTestOutput.png", width, height);
}

int main() {
    //testReadBack();
    //testModel();
    //testGather0();
    //testGather1(true);
    //testGather2(false);
    //ReluTest0();
    //ReluTest1();
    //AddTest0();
    //AddTest1();
    //ConcatTest0();
    //ConcatTest1();
    //ConcatTest2();
    //ConcatTest3();
    //UpsampleTest0();
    //UpsampleTest3();
    //CastTest0();
    //CastTest1();
    //INTest0();
    //INTest1();
    //ConvTest0();
    //ConvTest1();
    //SliceTest0();
    //SliceTest1();
    //PadTest0();
    //PadTest1();
    //PadTest2();

    FinalTest();

    return 0;
}