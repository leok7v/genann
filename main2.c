#include "nnn.h"
#include "rt.h"
#include <math.h>
#include "genann.h"
// https://github.com/codeplea/genann
// https://github.com/libfann/fann
// https://github.com/fabiooshiro/redesepstemicas/tree/master/RedesEpstemicas/lwneuralnet-0.8/source
// https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam
// https://github.com/loshchil/AdamW-and-SGDW  ( https://arxiv.org/abs/1711.05101 )

// https://stackoverflow.com/questions/13897316/approximating-the-sine-function-with-a-neural-network

enum { nn_inputs = 2, nn_layers = 3, nn_hidden = 10, nn_outputs = 1 };

static genann *nn0;

#define dump_weights(nn) do { for (int i = 0; i < nn->total_weights; i++) { printf("%.3e ", nn->weight[i]); } printf("\n"); } while (0)

enum { n = 8 };

enum { epochs = 1000 };

#define learning_rate 0.001

int main2(int argc, const char* argv[]) {
    (void)argc; (void)argv;
    double inputs[n * n * 2];
    double truth[n * n];
    double output[n * n];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double x = (double)i;
            double y = (double)j;
            int ox = i * n + j;
            int ix = ox * 2;
            inputs[ix + 0] = x;
            inputs[ix + 1] = y;
            truth[ox] = x + y;
        }
    }
    nn0 = genann_init(nn_inputs, nn_layers, nn_hidden, nn_outputs);
    nn0->activation_hidden = genann_act_linear;
    nn0->activation_output = genann_act_linear;
#if 0 // will NOT work with tanh because it is -1..1 bounded
    nn0->activation_hidden = genann_act_tanh;
    nn0->activation_output = genann_act_tanh;
#endif
    /* Train on the four labeled data points many times. */
    for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int ox = i * n + j;
                int ix = ox * 2;
                genann_train(nn0, (double*)&inputs[ix], &truth[ox], learning_rate);
            }
        }
    }
    /* Run the network and see what it predicts. */
    double e = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int ox = i * n + j;
            int ix = ox * 2;
            output[ox] = *genann_run(nn0, (double*)&inputs[ix]);
            double error = truth[ox] - output[ox];
            e += error * error;
            double rms = sqrt(error * error);
            printf("%10.6f + %10.6f = %10.6f truth %10.6f rms %10.6f\n",
                inputs[ix + 0], inputs[ix + 1], output[ox], truth[ox],
                rms);
        }
    }
    e /= (double)n * (double)n; // length of [countof[input]] dimensional vector
    printf("error: %.16e\n", e);
    double in0[2] = {9.0, 9.0};
    double r9p9 = *genann_run(nn0, in0);
    printf("%10.6f * %10.6f = %10.6f %10.6f delta=%.16e\n",
        in0[0], in0[1], r9p9, in0[1] + in0[1], r9p9 - (in0[1] + in0[1]));

    double in1[2] = {13.0, 15.0};
    double r13p15 = *genann_run(nn0, in1);
    printf("%10.6f * %10.6f = %10.6f %10.6f delta=%.16e\n",
        in1[0], in1[1], r13p15, in1[1] + in1[1], r13p15 - (in1[1] + in1[1]));

//  dump_weights(nn0);
    genann_free(nn0);
    return 0;
}
