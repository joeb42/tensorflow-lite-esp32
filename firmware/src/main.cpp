#include <Arduino.h>
#include "NeuralNetwork.h"
#include <math.h>


NeuralNetwork *nn;
int i;
float losses [630];

void setup()
{
  Serial.begin(115200);
  nn = new NeuralNetwork();
  i = 0;
  // Serial.println("setup");
}

void loop()
{
  // Serial.println("starting loop");


  // float number1 = random(100) / 100.0;
  // float number2 = random(100) / 100.0;

  // nn->getInputBuffer()[0] = number1;
  // nn->getInputBuffer()[1] = number2;

  // float result = nn->predict();

  // const char *expected = number2 > number1 ? "True" : "False";

  // const char *predicted = result > 0.5 ? "True" : "False";

  // Serial.printf("%.2f %.2f - result %.2f - Expected %s, Predicted %s\n", number1, number2, result, expected, predicted);

  if (i >= 630){
    Serial.println("Printing losses: ");
    Serial.println("===================================");
    for (int i = 0; i < 630; i++){
      Serial.printf("%.2f, ", losses[i]);
      if (i % 25 == 0){
        Serial.printf("\n");
      }
    }
    return;
  }

  float x = static_cast<float> (i) / static_cast<float> (100);

  // int8_t n1 = nn->convert_float2int8(0.1);

  nn->getInputBuffer()[0] = nn->convert_float2int8(x);
  float result = nn->convert_int82float(nn->predict());
  float loss = abs(cos(x) - result);
  losses[i] = loss;
  Serial.printf("x = %.2f, prediction = %.2f, loss = %.2f\n", x, result, loss);
  i += 1;
  delay(1000);
}