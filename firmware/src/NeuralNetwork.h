#ifndef __NeuralNetwork__
#define __NeuralNetwork__

#include <stdint.h>

namespace tflite
{
    template <unsigned int tOpCount>
    class MicroMutableOpResolver;
    class ErrorReporter;
    class Model;
    class MicroInterpreter;
} // namespace tflite

struct TfLiteTensor;

class NeuralNetwork
{
private:
    tflite::MicroMutableOpResolver<10> *resolver;
    // tflite::AllOpsResolver resolver;
    tflite::ErrorReporter *error_reporter;
    const tflite::Model *model;
    tflite::MicroInterpreter *interpreter;
    uint8_t *tensor_arena;
    TfLiteTensor *input;
    TfLiteTensor *output;

public:
    int8_t *getInputBuffer();
    NeuralNetwork();
    int8_t predict();
    int8_t convert_float2int8(float x);
    float convert_int82float(int8_t x);
};
#endif
