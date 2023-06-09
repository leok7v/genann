#include "rt.h"
#include <math.h>
#include "genann.h"

enum { epochs = 1 };

static genann* nn;

static void addition() {
    enum { n = 10 };
    double input[2] = {0, 0};
    double truth[1] = {0};
    nn = genann_init(2, 1, 2, 1);
    nn->activation_hidden = genann_act_linear;
    nn->activation_output = genann_act_linear;
    static uint32_t seed = 1;
    for (int epoch = 0; epoch < 300; epoch++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                input[0] = i;
                input[1] = j;
                truth[0] = i + j;
                static const double learning_rate = 0.001;
                genann_train(nn, input, truth, learning_rate);
            }
        }
        if (epoch % 10 == 0) {
            double test[2] = {0};
            test[0] = (double)(int)((100.0 * random32(&seed)) / (double)UINT32_MAX);
            test[1] = (double)(int)((100.0 * random32(&seed)) / (double)UINT32_MAX);
            const double* output = genann_run(nn, test);
            double loss = output[0] - (test[0] + test[1]);
            loss = sqrt(loss * loss);
            println("%d loss: %.7f", epoch, loss);
        }
    }
    for (int i = 0; i < 24; i++) {
        double test[2] = {0};
        test[0] = (double)(int)((100.0 * random32(&seed)) / (double)UINT32_MAX);
        test[1] = (double)(int)((100.0 * random32(&seed)) / (double)UINT32_MAX);
        const double* output = genann_run(nn, test);
        println("%2.0f + %2.0f = %3.0f", test[0], test[1], output[0]);
    }
    genann_free(nn);
}

int main3(int argc, const char* argv[]) {
    (void)argc; (void)argv;
    static double learning_rate = 0.025;
    double input[16] = {0};
    double output[countof(input)] = {0};
    for (int i = 0; i < countof(input); i++) {
        input[i] = rand() / (double)RAND_MAX;
        output[i] = rand() / (double)RAND_MAX;
    }
    // all permutations of input, layers, hidden, output 1..16
    for (int inputs = 1; inputs < countof(input); inputs++) {
        for (int layers = 0; layers < 16; layers++) {
            const int from = layers == 0 ? 0 : 1;
            const int to = layers == 0 ? 0 : countof(input) - 1;
            for (int hidden = from; hidden <= to; hidden++) {
                for (int outputs = 1; outputs < countof(input); outputs++) {
                    nn = genann_init(inputs, layers, hidden, outputs);
                    nn->activation_hidden = genann_act_linear;
                    nn->activation_output = genann_act_linear;
//                  println("inputs: %d layers: %d hidden: %d outputs: %d",
//                           inputs, layers, hidden, outputs);
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
    println("done");
    addition();
    return 0;
}
