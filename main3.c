#include "rt.h"
#include <math.h>
#include "genann.h"

enum { epochs = 1 };

static genann* nn;

static double learning_rate = 0.025;

int main3(int argc, const char* argv[]) {
    (void)argc; (void)argv;
    double input[16] = {0};
    double output[16] = {0};
    for (int i = 0; i < countof(input); i++) {
        input[i] = rand() / (double)RAND_MAX;
        output[i] = rand() / (double)RAND_MAX;
    }
    for (int inputs = 1; inputs < countof(input); inputs++) {
        for (int layers = 1; layers < 16; layers++) {
            for (int hidden = 1; hidden < 16; hidden++) {
                for (int outputs = 2; outputs < countof(input); outputs++) {
                    nn = genann_init(inputs, layers, hidden, outputs);
                    nn->activation_hidden = genann_act_linear;
                    nn->activation_output = genann_act_linear;
                    traceln("inputs: %d layers: %d hidden: %d outputs: %d",
                             inputs, layers, hidden, outputs);
                    for (int e = 0; e < epochs; e++) {
                        genann_train(nn, input, output, learning_rate);
                    }
                    const double* o = genann_run(nn, input);
                    (void)o;
                    genann_free(nn);
                }
            }
        }
    }
    traceln("done");
    return 0;
}
