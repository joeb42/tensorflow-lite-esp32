// #include "NeuralNetwork.h"
// #include "model_data.h"
// #include "tensorflow/lite/micro/all_ops_resolver.h"
// #include "tensorflow/lite/micro/micro_error_reporter.h"
// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/schema/schema_generated.h"
// #include "tensorflow/lite/version.h"

// const int kArenaSize = 200000;

// NeuralNetwork::NeuralNetwork()
// {
//     error_reporter = new tflite::MicroErrorReporter();

//     model = tflite::GetModel(model);
//     if (model->version() != TFLITE_SCHEMA_VERSION)
//     {
//         TF_LITE_REPORT_ERROR(error_reporter, "Model provided is schema version %d not equal to supported version %d.",
//                              model->version(), TFLITE_SCHEMA_VERSION);
//         return;
//     }
//     // This pulls in the operators implementations we need
//     resolver = new tflite::MicroMutableOpResolver<10>();
//     resolver->AddFullyConnected();
//     resolver->AddMul();
//     resolver->AddAdd();
//     resolver->AddLogistic();
//     resolver->AddReshape();
//     resolver->AddQuantize();
//     resolver->AddDequantize();
//     static tflite::AllOpsResolver resolver;

//   // Build an interpreter to run the model with.
//   static tflite::MicroInterpreter static_interpreter(
//       model, resolver, tensor_arena, kArenaSize, error_reporter);
//   interpreter = &static_interpreter;

//     tensor_arena = (uint8_t *)malloc(kArenaSize);
//     if (!tensor_arena)
//     {
//         TF_LITE_REPORT_ERROR(error_reporter, "Could not allocate arena");
//         return;
//     }

//     // // Build an interpreter to run the model with.
//     // interpreter = new tflite::MicroInterpreter(
//     //     model, *resolver, tensor_arena, kArenaSize, error_reporter);

//     // Allocate memory from the tensor_arena for the model's tensors.
//     TfLiteStatus allocate_status = interpreter->AllocateTensors();
//     if (allocate_status != kTfLiteOk)
//     {
//         TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
//         return;
//     }

//     size_t used_bytes = interpreter->arena_used_bytes();
//     TF_LITE_REPORT_ERROR(error_reporter, "Used bytes %d\n", used_bytes);

//     // Obtain pointers to the model's input and output tensors.
//     input = interpreter->input(0);
//     output = interpreter->output(0);
// }

// float *NeuralNetwork::getInputBuffer()
// {
//     return input->data.f;
// }

// float NeuralNetwork::predict()
// {
//     interpreter->Invoke();
//     return output->data.f[0];
// }

#include "NeuralNetwork.h"
#include "model_data.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

const int kArenaSize = 20000;

NeuralNetwork::NeuralNetwork()
{
    error_reporter = new tflite::MicroErrorReporter();

    model = tflite::GetModel(converted_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        TF_LITE_REPORT_ERROR(error_reporter, "Model provided is schema version %d not equal to supported version %d.",
                             model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }
    // This pulls in the operators implementations we need
    resolver = new tflite::MicroMutableOpResolver<10>();
    resolver->AddFullyConnected();
    resolver->AddMul();
    resolver->AddAdd();
    resolver->AddLogistic();
    resolver->AddReshape();
    resolver->AddQuantize();
    resolver->AddDequantize();

    tensor_arena = (uint8_t *)malloc(kArenaSize);
    if (!tensor_arena)
    {
        TF_LITE_REPORT_ERROR(error_reporter, "Could not allocate arena");
        return;
    }

    // Build an interpreter to run the model with.
    interpreter = new tflite::MicroInterpreter(
        model, *resolver, tensor_arena, kArenaSize, error_reporter);

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        return;
    }

    size_t used_bytes = interpreter->arena_used_bytes();
    TF_LITE_REPORT_ERROR(error_reporter, "Used bytes %d\n", used_bytes);

    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);
}

int8_t *NeuralNetwork::getInputBuffer()
{
    return input->data.int8;
}

int8_t NeuralNetwork::predict()
{
    interpreter->Invoke();
    return output->data.int8[0];
}

int8_t NeuralNetwork::convert_float2int8(float x){
   int8_t x_quantized = x / input->params.scale + input->params.zero_point;
   return x_quantized;
}

float NeuralNetwork::convert_int82float(int8_t x){
    float y = (x - output->params.zero_point) * output->params.scale;
    return y;
}