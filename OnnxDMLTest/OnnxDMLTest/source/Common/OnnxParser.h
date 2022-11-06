


#include "Common.h"
#include <memory>
#include <iostream>
#include <xstring>
#include <fcntl.h>
#include <map>
#include <vector>


// onnx.proto3
//message TensorProto{
//  enum DataType {
//	UNDEFINED = 0;
//	// Basic types.
//	FLOAT = 1;   // float
//	UINT8 = 2;   // uint8_t
//	INT8 = 3;    // int8_t
//	UINT16 = 4;  // uint16_t
//	INT16 = 5;   // int16_t
//	INT32 = 6;   // int32_t
//	INT64 = 7;   // int64_t
//	STRING = 8;  // string
//	BOOL = 9;    // bool
//
//	// IEEE754 half-precision floating-point format (16 bits wide).
//	// This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
//	FLOAT16 = 10;
//
//	DOUBLE = 11;
//	UINT32 = 12;
//	UINT64 = 13;
//	COMPLEX64 = 14;     // complex with float32 real and imaginary components
//	COMPLEX128 = 15;    // complex with float64 real and imaginary components
//
//	// Non-IEEE floating-point format based on IEEE754 single-precision
//	// floating-point number truncated to 16 bits.
//	// This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
//	BFLOAT16 = 16;
//
//	// Future extensions go here.
//}

	
namespace onnx {
	class NodeProto;
	class TensorProto;
	class ModelProto;
}


namespace ONNX_PARSER {
	class NodeAttrHelper;
 
	struct ONNXPARSER_API TensorInfo {
		TensorType tensorType;
		std::string name;
		unsigned int  dims;
		std::vector<uint32_t> shapes;   // TODO: different from onnx Shape definition (which is int64_t)

		TensorInfo() : dims(0) {}
		TensorInfo(const std::string& n, unsigned int d, TensorType t);

		uint64_t GetSize() const;
		void SetShape(unsigned int dim, uint32_t v);

	};
	struct ONNXPARSER_API InitializerTensorInfo : public TensorInfo {

		unsigned int index; // index + modelInputNum == graph tensor index (used for binding resources at runtime)

		InitializerTensorInfo() : TensorInfo() {};
		InitializerTensorInfo(const std::string& n, unsigned int d, TensorType t, unsigned int i);

	};

	struct ONNXPARSER_API Op {
		std::vector<std::string> inputNames;

		// std::vector<TensorInfo> inputInfo;

		std::string outputName;
		// TensorInfo outputInfo;
		/*std::optional<std::vector<int64_t>> inputShape;
		std::optional<std::vector<int64_t>> outputShape;*/
		std::string opName;
		std::string opType;

		/*TensorType inputTensorType;
		TensorType outputTensorType;*/
		unsigned int opIndex; // operator index inside network graph
		Op();
		Op(const onnx::NodeProto& node, unsigned int i);
		Op(const std::vector<std::string>& input, const std::string& output, const std::string& name, const std::string& type, const unsigned int index);
		~Op();
		bool GetAttribute(const std::string& attriName, AttributeType attriType, std::vector<char>& returnVal);
		// void AppendIOInfo(std::map<std::string, TensorInfo>&, std::map<std::string, TensorInfo>&, std::map<std::string, InitializerTensorInfo>&);
		void AppendAdditionAttribute(const onnx::TensorProto& attriTensor, const  std::string& name);
		/*Op& operator = (Op&& op);
		Op(Op&& op);*/
		Op& operator = (const Op& op);
		Op(const Op& op);
	private:
		std::map<std::string, onnx::TensorProto> additionalAtrribute; // TODO: some operator need attribute from initializer
		//std::unique_ptr<NodeAttrHelper> attriHelper;
		NodeAttrHelper* attriHelper;
	};



	struct ONNXPARSER_API BindingInfo {
		unsigned int stride; // all initializer data is stored in a single buffer, use stride to indicate
		unsigned int byteSize;
		
		BindingInfo() = default;
		BindingInfo(unsigned int s, unsigned int w);
	};

	class ONNXPARSER_API OnnxParser {
	private:
		onnx::ModelProto * model;
		std::vector<char> weightValues;

		std::map<std::string, TensorInfo> inputMap;
		std::map<std::string, TensorInfo> outputMap;
		std::map<std::string, Op> nodeMap;
		std::map<std::string, InitializerTensorInfo> initializerMap;
		std::vector<BindingInfo> bindings;

		std::map<std::string, onnx::TensorProto> initializerMetaData; // used for graph node parsing

		void ParseInputs();
		void ParseOutputs();
		void ParseGraphNodes();
		//void ParseGraphNodes(std::map<std::string, Op>&); // TODO: only support single output node
		void ParseGraphInitializers();
	public:
		OnnxParser(const std::wstring& path_to_onnx);
		~OnnxParser();
		//OnnxParser(google::protobuf::io::FileInputStream* fileStream);
		int64_t GetIrVersion() const;
		std::string GetProducerName() const;
		int64_t GetOpsetVersion() const;

		// return const reference for avoiding passing memory allocated in DLL heap which might be released in application heap space
		const std::map<std::string, TensorInfo>& GetInputs() const;
		const std::map<std::string, TensorInfo>& GetOutputs() const;
		//void GetGraphNodes(std::map<std::string, Op>&);
		const std::map<std::string, Op>& GetGraphNodes() const;
		const std::map<std::string, InitializerTensorInfo>& GetGraphInitializers() const;
		const std::vector<BindingInfo>& GetBindings() const;
		// data
		const std::vector<char>& GetWeights() const;

	};
	//ONNXPARSER_API PERROR CreateParserFromFile(const std::wstring& path_to_onnx, OnnxParser** pOnnxParser);
	//ONNXPARSER_API PERROR GetNetworkInputs(const OnnxParser& pOnnxParser);
	ONNXPARSER_API PERROR ParseFromFile(const std::wstring& path_to_onnx, std::map<std::string, TensorInfo>& inputMap, std::map<std::string, TensorInfo>& outputMap, std::map<std::string, Op>& graphNodes, std::map<std::string, InitializerTensorInfo>& graphInitializers, std::vector<BindingInfo>& bindings, std::vector<char>& weights, unsigned int& opsetVersion);

	/*ONNXPARSER_API PERROR TestFunction(const std::wstring& path_to_onnx, std::map<std::string, TensorInfo>& inputMap, std::map<std::string, TensorInfo>& outputMap, std::map<std::string, Op>& graphNodes, std::map<std::string, InitializerTensorInfo>& graphInitializers, std::vector<BindingInfo>& bindings, std::vector<char>& weights, unsigned int& opsetVersion);*/
}
