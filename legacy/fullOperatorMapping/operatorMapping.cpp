// TODO: Windows10 上应该要支持的（注释的部分需要0x5100，可能只支持Win11）
// 目前只尝试支持了16种operator
// https://github.com/onnx/onnx/blob/main/docs/Operators.md
// onnxruntime\core\providers\dml\DmlExecutionProvider\src\Operators\OperatorRegistration.cpp

// operator that has different parameter with different ir version
#define REG_INFO_VER(operatorName, version) \
    #operatorName, Create##operatorName##version,
#define REG_INFO(operatorName) \
    #operatorName, Create##operatorName,
// operaters that can be mapped to DmlOperatorCopy
#define REG_INFO_COPY(operatorName) \
    #operatorName, CreateCopy,

    
#define DML_OP_EXTERN_CREATION_FUNCTION(operatorName) CALLBACK Create##operatorName(const std::map<std::string, dml::Expression>& expressionMap, const Op& node)


DML_OP_EXTERN_CREATION_FUNCTION(Copy);
DML_OP_EXTERN_CREATION_FUNCTION(FC);
DML_OP_EXTERN_CREATION_FUNCTION(Conv);
DML_OP_EXTERN_CREATION_FUNCTION(ConvTranspose);
// DML_OP_EXTERN_CREATION_FUNCTION(ConvTransposeWithDynamicPads);
DML_OP_EXTERN_CREATION_FUNCTION(AveragePool);
DML_OP_EXTERN_CREATION_FUNCTION(GlobalAveragePool);
DML_OP_EXTERN_CREATION_FUNCTION(MaxPool);
DML_OP_EXTERN_CREATION_FUNCTION(GlobalMaxPool);
DML_OP_EXTERN_CREATION_FUNCTION(LpPool);
DML_OP_EXTERN_CREATION_FUNCTION(GlobalLpPool);
DML_OP_EXTERN_CREATION_FUNCTION(MaxRoiPool);
DML_OP_EXTERN_CREATION_FUNCTION(RoiAlign10);
DML_OP_EXTERN_CREATION_FUNCTION(InstanceNormalization);
DML_OP_EXTERN_CREATION_FUNCTION(BatchNormalization);
DML_OP_EXTERN_CREATION_FUNCTION(BatchNormalization15);
DML_OP_EXTERN_CREATION_FUNCTION(LRN);
DML_OP_EXTERN_CREATION_FUNCTION(MeanVarianceNormalization);
DML_OP_EXTERN_CREATION_FUNCTION(LpNormalization);
DML_OP_EXTERN_CREATION_FUNCTION(RNN);
DML_OP_EXTERN_CREATION_FUNCTION(GRU);
DML_OP_EXTERN_CREATION_FUNCTION(LSTM);
DML_OP_EXTERN_CREATION_FUNCTION(Gather);
DML_OP_EXTERN_CREATION_FUNCTION(Flatten);
DML_OP_EXTERN_CREATION_FUNCTION(Split7);
DML_OP_EXTERN_CREATION_FUNCTION(Split11);
DML_OP_EXTERN_CREATION_FUNCTION(Split13);
DML_OP_EXTERN_CREATION_FUNCTION(Transpose);
DML_OP_EXTERN_CREATION_FUNCTION(Tile);
DML_OP_EXTERN_CREATION_FUNCTION(Concat);
DML_OP_EXTERN_CREATION_FUNCTION(Slice7);
DML_OP_EXTERN_CREATION_FUNCTION(Slice10);
DML_OP_EXTERN_CREATION_FUNCTION(Slice11);
DML_OP_EXTERN_CREATION_FUNCTION(Slice13);
DML_OP_EXTERN_CREATION_FUNCTION(Pad7);
DML_OP_EXTERN_CREATION_FUNCTION(Pad11);
DML_OP_EXTERN_CREATION_FUNCTION(Pad13);
DML_OP_EXTERN_CREATION_FUNCTION(SpaceToDepth);
DML_OP_EXTERN_CREATION_FUNCTION(DepthToSpace);
DML_OP_EXTERN_CREATION_FUNCTION(Sqrt);
DML_OP_EXTERN_CREATION_FUNCTION(Reciprocal);
DML_OP_EXTERN_CREATION_FUNCTION(Pow);
DML_OP_EXTERN_CREATION_FUNCTION(Exp);
DML_OP_EXTERN_CREATION_FUNCTION(Log);
DML_OP_EXTERN_CREATION_FUNCTION(Abs);
DML_OP_EXTERN_CREATION_FUNCTION(Ceil);
DML_OP_EXTERN_CREATION_FUNCTION(Floor);
DML_OP_EXTERN_CREATION_FUNCTION(Clip7);
DML_OP_EXTERN_CREATION_FUNCTION(Clip11);
DML_OP_EXTERN_CREATION_FUNCTION(Clip12);
DML_OP_EXTERN_CREATION_FUNCTION(Clip13);
DML_OP_EXTERN_CREATION_FUNCTION(Greater);
DML_OP_EXTERN_CREATION_FUNCTION(Less);
DML_OP_EXTERN_CREATION_FUNCTION(GreaterOrEqual);
DML_OP_EXTERN_CREATION_FUNCTION(LessOrEqual);
DML_OP_EXTERN_CREATION_FUNCTION(Equal);
DML_OP_EXTERN_CREATION_FUNCTION(Not);
DML_OP_EXTERN_CREATION_FUNCTION(And);
DML_OP_EXTERN_CREATION_FUNCTION(Or);
DML_OP_EXTERN_CREATION_FUNCTION(Xor);
DML_OP_EXTERN_CREATION_FUNCTION(Add);
DML_OP_EXTERN_CREATION_FUNCTION(Sub);
DML_OP_EXTERN_CREATION_FUNCTION(Mul);
DML_OP_EXTERN_CREATION_FUNCTION(Div);
DML_OP_EXTERN_CREATION_FUNCTION(Sum);
DML_OP_EXTERN_CREATION_FUNCTION(Mean);
DML_OP_EXTERN_CREATION_FUNCTION(Max);
DML_OP_EXTERN_CREATION_FUNCTION(Min);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceSum);
DML_OP_EXTERN_CREATION_FUNCTION(Einsum12);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceMean);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceProd);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceLogSum);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceLogSumExp);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceSumSquare);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceL1);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceL2);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceMax);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceMin);
DML_OP_EXTERN_CREATION_FUNCTION(ArgMax);
DML_OP_EXTERN_CREATION_FUNCTION(ArgMin);
DML_OP_EXTERN_CREATION_FUNCTION(Gemm);
DML_OP_EXTERN_CREATION_FUNCTION(Neg);
DML_OP_EXTERN_CREATION_FUNCTION(Crop);
DML_OP_EXTERN_CREATION_FUNCTION(ImageScaler);
DML_OP_EXTERN_CREATION_FUNCTION(Upsample7);
DML_OP_EXTERN_CREATION_FUNCTION(Upsample9);
DML_OP_EXTERN_CREATION_FUNCTION(Upsample10);
DML_OP_EXTERN_CREATION_FUNCTION(Sigmoid);
DML_OP_EXTERN_CREATION_FUNCTION(HardSigmoid);
DML_OP_EXTERN_CREATION_FUNCTION(Tanh);
DML_OP_EXTERN_CREATION_FUNCTION(ScaledTanh);
DML_OP_EXTERN_CREATION_FUNCTION(Relu);
DML_OP_EXTERN_CREATION_FUNCTION(LeakyRelu);
DML_OP_EXTERN_CREATION_FUNCTION(PRelu);
DML_OP_EXTERN_CREATION_FUNCTION(ThresholdedRelu);
DML_OP_EXTERN_CREATION_FUNCTION(Elu);
DML_OP_EXTERN_CREATION_FUNCTION(Celu);
DML_OP_EXTERN_CREATION_FUNCTION(Selu);
DML_OP_EXTERN_CREATION_FUNCTION(Softmax);
// DML_OP_EXTERN_CREATION_FUNCTION(Softmax13);
DML_OP_EXTERN_CREATION_FUNCTION(LogSoftmax);
// DML_OP_EXTERN_CREATION_FUNCTION(LogSoftmax13);
DML_OP_EXTERN_CREATION_FUNCTION(Hardmax);
// DML_OP_EXTERN_CREATION_FUNCTION(Hardmax13);
DML_OP_EXTERN_CREATION_FUNCTION(Softsign);
DML_OP_EXTERN_CREATION_FUNCTION(Softplus);
DML_OP_EXTERN_CREATION_FUNCTION(ParametricSoftplus);
DML_OP_EXTERN_CREATION_FUNCTION(Affine);
DML_OP_EXTERN_CREATION_FUNCTION(Dropout);
DML_OP_EXTERN_CREATION_FUNCTION(MatMul);
DML_OP_EXTERN_CREATION_FUNCTION(Cast);
DML_OP_EXTERN_CREATION_FUNCTION(CastLike15);
// DML_OP_EXTERN_CREATION_FUNCTION(MemcpyFromHost);
// DML_OP_EXTERN_CREATION_FUNCTION(MemcpyToHost);
DML_OP_EXTERN_CREATION_FUNCTION(TopK7);
DML_OP_EXTERN_CREATION_FUNCTION(TopK10);
DML_OP_EXTERN_CREATION_FUNCTION(TopK11);
DML_OP_EXTERN_CREATION_FUNCTION(Expand);
DML_OP_EXTERN_CREATION_FUNCTION(Cos);
DML_OP_EXTERN_CREATION_FUNCTION(Sin);
DML_OP_EXTERN_CREATION_FUNCTION(Tan);
DML_OP_EXTERN_CREATION_FUNCTION(Acos);
DML_OP_EXTERN_CREATION_FUNCTION(Asin);
DML_OP_EXTERN_CREATION_FUNCTION(Atan);
// DML_OP_EXTERN_CREATION_FUNCTION(FusedConv);
// DML_OP_EXTERN_CREATION_FUNCTION(FusedConvTranspose);
// DML_OP_EXTERN_CREATION_FUNCTION(FusedInstanceNormalization);
// DML_OP_EXTERN_CREATION_FUNCTION(FusedBatchNormalization);
// DML_OP_EXTERN_CREATION_FUNCTION(FusedMeanVarianceNormalization);
// DML_OP_EXTERN_CREATION_FUNCTION(FusedGemm);
// DML_OP_EXTERN_CREATION_FUNCTION(FusedMatMul);
// DML_OP_EXTERN_CREATION_FUNCTION(FusedAdd);
// DML_OP_EXTERN_CREATION_FUNCTION(FusedSum);
DML_OP_EXTERN_CREATION_FUNCTION(QuantizeLinear);
DML_OP_EXTERN_CREATION_FUNCTION(DequantizeLinear);
DML_OP_EXTERN_CREATION_FUNCTION(Sign);
DML_OP_EXTERN_CREATION_FUNCTION(IsNaN);
DML_OP_EXTERN_CREATION_FUNCTION(Sinh);
DML_OP_EXTERN_CREATION_FUNCTION(Cosh);
DML_OP_EXTERN_CREATION_FUNCTION(Tanh);
DML_OP_EXTERN_CREATION_FUNCTION(Asinh);
DML_OP_EXTERN_CREATION_FUNCTION(Acosh);
DML_OP_EXTERN_CREATION_FUNCTION(Atanh);
DML_OP_EXTERN_CREATION_FUNCTION(Erf);
DML_OP_EXTERN_CREATION_FUNCTION(Where);
DML_OP_EXTERN_CREATION_FUNCTION(Shrink);
// DML_OP_EXTERN_CREATION_FUNCTION(Gelu);
DML_OP_EXTERN_CREATION_FUNCTION(OneHot);
DML_OP_EXTERN_CREATION_FUNCTION(EyeLike);
DML_OP_EXTERN_CREATION_FUNCTION(MaxUnpool);
DML_OP_EXTERN_CREATION_FUNCTION(Scatter9);
DML_OP_EXTERN_CREATION_FUNCTION(Scatter11);
DML_OP_EXTERN_CREATION_FUNCTION(Scatter13);
DML_OP_EXTERN_CREATION_FUNCTION(Resize10);
DML_OP_EXTERN_CREATION_FUNCTION(Resize11);
DML_OP_EXTERN_CREATION_FUNCTION(Resize13);
DML_OP_EXTERN_CREATION_FUNCTION(ConstantOfShape);
DML_OP_EXTERN_CREATION_FUNCTION(IsInf);
DML_OP_EXTERN_CREATION_FUNCTION(Mod);
DML_OP_EXTERN_CREATION_FUNCTION(BitShift);
DML_OP_EXTERN_CREATION_FUNCTION(CumSum11);
DML_OP_EXTERN_CREATION_FUNCTION(CumSum14);
DML_OP_EXTERN_CREATION_FUNCTION(GatherElements);
DML_OP_EXTERN_CREATION_FUNCTION(GatherND);
DML_OP_EXTERN_CREATION_FUNCTION(Range);
DML_OP_EXTERN_CREATION_FUNCTION(ReverseSequence);
DML_OP_EXTERN_CREATION_FUNCTION(Round);
DML_OP_EXTERN_CREATION_FUNCTION(ScatterElements);
DML_OP_EXTERN_CREATION_FUNCTION(ScatterND);
// DML_OP_EXTERN_CREATION_FUNCTION(QLinearAdd);
DML_OP_EXTERN_CREATION_FUNCTION(QLinearConv);
DML_OP_EXTERN_CREATION_FUNCTION(QLinearMatMul);
DML_OP_EXTERN_CREATION_FUNCTION(DynamicQuantizeLinear);
DML_OP_EXTERN_CREATION_FUNCTION(MatMulInteger);
DML_OP_EXTERN_CREATION_FUNCTION(ConvInteger);
// DML_OP_EXTERN_CREATION_FUNCTION(Trilu);



DML_OP_DEFINE_CREATION_FUNCTION(Concat, DmlOperatorConcat);

#define DML_OP_DEFINE_CREATION_FUNCTION(operatorName, ...)\
extern dml::Expression CALLBACK Create##operatorName(const std::map<std::string, dml::Expression>& expressionMap, const Op& node)\
{\
    using T = __VA_ARGS__; \
    OperatorGenerator<T>::CreateDmlExpression(expressionMap, node);\
}

using CreateFn = dml::Expression(CALLBACK*)(const std::map<std::string, dml::Expression>& expressionMap, const Op& node);


constexpr static std::unordered_map<std::string, CreateFn> g_operatorRegistrationMap = 
{
    {REG_INFO(      Conv,   )},
    {REG_INFO(     ConvTranspose, )},
    {REG_INFO(     AveragePool,  )},
    {REG_INFO(     GlobalAveragePool,   )},
    {REG_INFO(      MaxPool,      )},
    {REG_INFO(      GlobalMaxPool,)},
    {REG_INFO(      LpPool,       )},
    {REG_INFO(      GlobalLpPool, )},
    {REG_INFO(      MaxRoiPool,   )},
    {REG_INFO_VER(  RoiAlign, 10, )},
    {REG_INFO(      InstanceNormalization,     )},
    {REG_INFO(      BatchNormalization,  )},
    // {REG_INFO(      BatchNormalization,  )},  // v9 just removes 'spatial' attribute.
    // {REG_INFO(      BatchNormalization, )},  // v14 adds training_mode attribute
    // {REG_INFO(      BatchNormalization, )},  // v15 adds differing types for scale and bias vs input.
    {REG_INFO(      LRN, )},
    {REG_INFO(      MeanVarianceNormalization, )},
    {REG_INFO(      LpNormalization,     )},
    {REG_INFO(      RNN,  )},
    {REG_INFO(      GRU,  )},
    {REG_INFO(      LSTM, )},
    // {REG_INFO_MS(     ConvTransposeWithDynamicPads,  )},

    // Data Reorganization Layers
    {REG_INFO_VER(  Split, 7, )},
    {REG_INFO_VER( Split, 11, )},  // Adds negative axis.
    {REG_INFO_VER(  Split, 13,      )},  // Moves splits from constant parameter to dynamic input.
    {REG_INFO(      Transpose,  )},
    {REG_INFO(      Concat,     )},  // Adds negative axis.
    {REG_INFO_VER(  Slice, 7, )},
    {REG_INFO_VER( Slice, 10, )},  // Adds negative axes.
    {REG_INFO_VER( Slice, 11, )},  // Adds negative axes.
    {REG_INFO_VER( Slice, 13, )},  // Adds negative axes.
    {REG_INFO_VER(  Pad,  7,      )},
    {REG_INFO_VER( Pad,  11,      )}, // https://microsoft.visualstudio.com/OS/_workitems/edit/26007728
    {REG_INFO_VER( Pad,  13,      )}, // https://microsoft.visualstudio.com/OS/_workitems/edit/26007728
    {REG_INFO(      SpaceToDepth,        )},
    {REG_INFO(      DepthToSpace,        )},
    {REG_INFO(      Tile, ,      )},
    {REG_INFO(      Expand,     ,      )},
    {REG_INFO(        ConstantOfShape,    )},
    {REG_INFO(      Gather,     )},
    {REG_INFO(      GatherElements,      )},
    {REG_INFO(     GatherND,   )},
    {REG_INFO_VER(    Scatter, 9, )},
    {REG_INFO_VER( Scatter, 11,  )},
    {REG_INFO_VER(  Scatter, 13, )},
    {REG_INFO(     ScatterElements,     )},
    {REG_INFO(      ScatterND,  )},
    {REG_INFO(        EyeLike,    )},
    // {REG_INFO(       Trilu,,     )},

    // Data reorganization that merely changes the dimensions while keeping the data identical.
    {REG_INFO_COPY( Identity,   )},
    {REG_INFO_COPY( Flatten,    )},
    {REG_INFO_COPY( Squeeze,    )},
    {REG_INFO_COPY( Unsqueeze,  )},
    {REG_INFO_COPY( Reshape,    ,     )},

    // Elementwise
    {REG_INFO(      Sqrt,)},
    {REG_INFO(      Reciprocal,   )},
    {REG_INFO(       Pow,  )},  // 15 added bfloat16 to T1.
    {REG_INFO(      Exp, )},
    {REG_INFO(      Log, )},
    {REG_INFO(      Abs,  )},
    {REG_INFO(      Ceil,)},
    {REG_INFO(      Floor,        )},
    {REG_INFO_VER(  Clip, 7, )},
    {REG_INFO_VER( Clip, 11, )},
    {REG_INFO_VER(   Clip, 12, )},
    {REG_INFO_VER(  Clip, 13, )},
    {REG_INFO(      Add,  )},
    {REG_INFO(       Sub,  )},
    {REG_INFO(       Mul,  )},
    {REG_INFO(       Div,  )},
    {REG_INFO(      Sum, },
    {REG_INFO(      Mean,},
    {REG_INFO(      Max,  },
    {REG_INFO(      Min,  },
    {REG_INFO(      Cos, )},
    {REG_INFO(      Sin, )},
    {REG_INFO(      Tan, )},
    {REG_INFO(      Acos,)},
    {REG_INFO(      Asin,)},
    {REG_INFO(      Atan,)},
    {REG_INFO(      Affine,       )},
    {REG_INFO(      QuantizeLinear,      )},
    {REG_INFO(      DequantizeLinear,    )},
    // {REG_INFO_MS(     QuantizeLinear,      )},
    // {REG_INFO_MS(     DequantizeLinear,    )},
    {REG_INFO(      Sign, )},
    {REG_INFO(      IsNaN,)},
    {REG_INFO(        Sinh,)},
    {REG_INFO(        Cosh,)},
    {REG_INFO(        Asinh,        )},
    {REG_INFO(        Acosh,        )},
    {REG_INFO(        Atanh,        )},
    {REG_INFO(      Erf, )},
    {REG_INFO(        Where,)},
    {REG_INFO(      ReduceSum,  ,     )},
    {REG_INFO_VER(   Einsum,    12   },
    {REG_INFO(     ReduceMean,   )},
    {REG_INFO(     ReduceProd, )},
    {REG_INFO(     ReduceLogSum, )},
    {REG_INFO(     ReduceLogSumExp,     )},
    {REG_INFO(     ReduceSumSquare,     )},
    {REG_INFO(     ReduceL1,   )},
    {REG_INFO(     ReduceL2,     )},
    {REG_INFO(     ReduceMax,    )},
    {REG_INFO(     ReduceMin,    )},
    {REG_INFO(     ArgMax,     },
    {REG_INFO(     ArgMin,     },
    {REG_INFO(     Gemm,)},
    {REG_INFO(      Neg,  )},
    {REG_INFO(      Greater,    )},
    {REG_INFO(      Less, )},
    {REG_INFO(       GreaterOrEqual,      )},
    {REG_INFO(       LessOrEqual,)},
    {REG_INFO(     Equal,)},
    {REG_INFO(      Not,  )},
    {REG_INFO(      And,  )},
    {REG_INFO(      Or,   )},
    {REG_INFO(      Xor,  )},

    // Imaging Operators
    {REG_INFO(      Crop,)},
    {REG_INFO(      ImageScaler,  )},
    {REG_INFO_VER(  Upsample, 7,    )},
    {REG_INFO_VER(  Upsample, 9,           /*scales*/)},
    {REG_INFO_VER( Upsample,  10,       /*scales*/)},
    {REG_INFO_VER( Resize,   10,       /*scales*/)},
    {REG_INFO_VER( Resize,   11, )},
    {REG_INFO_VER(  Resize,  13, )},

    // Activation Functions
    {REG_INFO(      Sigmoid,      )},
    {REG_INFO(      HardSigmoid,  )},
    {REG_INFO(      Tanh,)},
    {REG_INFO(      ScaledTanh,   )},
    {REG_INFO(      Relu,)},
    {REG_INFO(      LeakyRelu,    )},
    {REG_INFO(      PRelu,        )},
    {REG_INFO(      ThresholdedRelu,     )},
    {REG_INFO(      Elu, )},
    {REG_INFO(       Celu,)},
    {REG_INFO(      Selu,)},
    {REG_INFO(      Softmax,      )},
    // {REG_INFO_VER(  Softmax,      )},// TODO: require DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1, which requires DML feature level 0x5100, not supported on windows10 yet
    {REG_INFO(      LogSoftmax,   )},
    // {REG_INFO_VER(  LogSoftmax,   )},// TODO: require DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1, which requires DML feature level 0x5100, not supported on windows10 yet
    {REG_INFO(     Hardmax,      )},
    // {REG_INFO_VER(  Hardmax,      )},// TODO: require DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1, which requires DML feature level 0x5100, not supported on windows10 yet
    {REG_INFO(      Softsign,     )},
    {REG_INFO(      Softplus,     )},
    {REG_INFO(      ParametricSoftplus,  )},
    {REG_INFO(      Dropout,      )},
    {REG_INFO(        Shrink,     )},
    // {REG_INFO_MS(     Gelu,)},// TODO: require DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1, which requires DML feature level 0x5100, not supported on windows10 yet

    // Uncategorized
    {REG_INFO(      MatMul,       )},
    {REG_INFO(      Cast, )},
    {REG_INFO_VER(   CastLike,  15 )},
    {REG_INFO(      MemcpyFromHost,      )},
    {REG_INFO(      MemcpyToHost,        )},
    {REG_INFO_VER(  TopK, 7, )},
    {REG_INFO_VER( TopK, 10, )},
    {REG_INFO_VER( TopK, 11, )},
    {REG_INFO(     OneHot,     )},
    // Shape- Shape- Shape-15 rely on CPU.
    // Size-1 relies on CPU.

    // // Fused operators
    // {REG_INFO_MSDML(  FusedConv,    )},
    // {REG_INFO_MSDML(  FusedConvTranspose,  )},
    // {REG_INFO_MSDML(  FusedInstanceNormalization,)},
    // {REG_INFO_MSDML(  FusedBatchNormalization,   )},
    // {REG_INFO_MSDML(  FusedMeanVarianceNormalization,     )},
    // {REG_INFO_MSDML(  FusedGemm,    )},
    // {REG_INFO_MSDML(  FusedMatMul,  )},
    // {REG_INFO_MSDML(  FusedAdd,     )},
    // {REG_INFO_MSDML(  FusedSum,     },

    {REG_INFO(     IsInf,)},
    {REG_INFO(     Mod,  )},
    {REG_INFO(      Mod,  )},
    {REG_INFO(     BitShift,   )},
    {REG_INFO(     Round,        )},
    {REG_INFO(     ReverseSequence,     )},
    {REG_INFO_VER( CumSum,    11 ,     )},
    {REG_INFO_VER( CumSum,    14 ,     )},
    {REG_INFO(     Range,)},

    {REG_INFO(     MaxUnpool,  )},  // 11 is identical to 9.

    // {REG_INFO_MS(  QLinearAdd, )},
    {REG_INFO(     QLinearConv,)},
    {REG_INFO(     QLinearMatMul,       )},
    {REG_INFO(     MatMulInteger,       )},
    {REG_INFO(     ConvInteger,)},
    {REG_INFO(     DynamicQuantizeLinear, )},


};

dml::Expression CreateExpression(const std::map<std::string, dml::Expression>& expressionMap, const Op& node){
    return g_operatorRegistrationMap[node.opType](expressionMap, node, opsetVersion);
}
