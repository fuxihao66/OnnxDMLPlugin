#include <iostream>
#include <vector>
#include <windows.h>

#include "helper/pch.h"
#include "Common/Float16Compressor.h"
#include "thirdParty/tiny_jpeg.h"
#include "thirdParty/jpeg_decoder.h"
#include "OnnxDMLCore/OnnxDMLRHIModule.h"
// #include "OnnxParser.h"

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
int main()
{
    ODI::D3D12RHIContext context;

    /*ID3D12Resource * modelInput;
    ID3D12Resource * modelOutput;*/

    Microsoft::WRL::ComPtr<ID3D12Resource> modelInput;
    Microsoft::WRL::ComPtr<ID3D12Resource> modelOutput;
    Microsoft::WRL::ComPtr<ID3D12Resource> readbackOutput;


    std::vector<uint8_t> jpgInputData;
    int width, height;
    LoadImageFromFile(jpgInputData, "data/testimg.jpg", width, height);

    std::vector<uint16_t> inputData;
    Uint8ToHalfCHW(jpgInputData, inputData, width, height);
    auto bufferSize = width * height * 3 * sizeof(uint16_t);
    context.CreateBufferFromData(modelInput, std::optional<std::vector<uint16_t>>{inputData}, bufferSize); // buffer for inference
    context.CreateBufferFromData(modelOutput, std::nullopt, bufferSize);
    context.CreateBufferFromData(readbackOutput, std::nullopt, bufferSize, true);

    context.InitializeNewModel(L"D:/candy-9.onnx", "Candy");
    context.RunDMLInfer(std::map<std::string, ID3D12Resource*>{ {"input1", modelInput.Get()} }, modelOutput.Get(), "Candy");
    context.CopyForReadBack(modelOutput.Get(), readbackOutput.Get());
    context.ForceCPUSync();

    std::vector<uint16_t> cpuImageData;
    context.CPUReadBack(readbackOutput.Get(), cpuImageData, bufferSize);
    std::vector<uint8_t> jpgOutputData;

    HalfCHW2Uint8(cpuImageData, jpgOutputData, width, height);
    SaveImageToFile(jpgOutputData, "testOutput.png", width, height);

}