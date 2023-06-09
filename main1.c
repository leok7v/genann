#include "nnn.h"
#include "rt.h"
#include <math.h>
#include "genann.h"
#include "fffc.h"
// https://github.com/codeplea/genann
// https://github.com/libfann/fann
// https://github.com/fabiooshiro/redesepstemicas/tree/master/RedesEpstemicas/lwneuralnet-0.8/source
// https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam
// https://github.com/loshchil/AdamW-and-SGDW  ( https://arxiv.org/abs/1711.05101 )

// https://stackoverflow.com/questions/13897316/approximating-the-sine-function-with-a-neural-network

enum { nn_inputs = 1, nn_layers = 2, nn_hidden = 10, nn_outputs = 1 };

static genann *nn0;

#define dump_weights(nn) do { for (int i = 0; i < nn->total_weights; i++) { printf("%25.17e ", nn->weight[i]); } printf("\n"); } while (0)

#define PI 3.14159265358979323846
#define r2d(r) ((r) * 180 / PI)

enum { epochs = 1000 };

#define learning_rate 0.01

int main1(int argc, const char* argv[]) {
    (void)argc; (void)argv;
    double inputs[360];
    double truth[countof(inputs)];
    double output[countof(inputs)];
    for (int i = 0; i < countof(inputs); i++) {
        double x = 2 * PI * (i / (double)countof(inputs));
        inputs[i] = x;
        truth[i] = sin(x);
    }
    nn0 = genann_init(nn_inputs, nn_layers, nn_hidden, nn_outputs);
    nn0->activation_hidden = genann_act_tanh;
    nn0->activation_output = genann_act_tanh;
    /* Train on the four labeled data points many times. */
    for (int e = 0; e < epochs; e++) {
        for (int j = 0; j < countof(inputs); j++) {
            genann_train(nn0, (double*)&inputs[j], &truth[j], learning_rate);
//          dump_weights(nn0);
        }
    }
    /* Run the network and see what it predicts. */
    double e = 0;
    for (int i = 0; i < countof(inputs); i++) {
        output[i] = *genann_run(nn0, (double*)&inputs[i]);
        double error = truth[i] - output[i];
        e += error * error;
        double rms = sqrt(error * error);
        printf("sin(%6.3f): %6.3f truth %6.3f rms %6.3f\n", r2d(inputs[i]), output[i], truth[i], rms);
    }
    e /= countof(inputs); // length of [countof[input]] dimensional vector
    printf("error: %.16f\n", e);
    genann_free(nn0);
    return 0;
}
