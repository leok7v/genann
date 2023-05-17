#include "nnn.h"
#define RT_IMPLEMENTATION
#include "rt.h"
#include <memory.h>
#include "genann.h"
#include "ann.h"

typedef struct {
    double a;
    double b;
} input_t;

enum { n = 2 };

static input_t inputs[n * n];
static double  output[countof(inputs)];

static void ground_truth() {
    // 20x20 multiplication table
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int ix = i * n + j;
            inputs[ix].a = i;
            inputs[ix].b = j;
            output[ix] = (double)(i | j);
            printf("%f ^ %f = %f\n", inputs[ix].a, inputs[ix].b, output[ix]);

        }
    }
}

double randoms[1024];
double* verify;

enum { nn_inputs = 3, nn_layers = 4, nn_hidden = 10, nn_outputs = 2 };

enum { epochs = 5 }; // 500

static genann *nn0;
static ann_t*  nn1;

#define dump_weights(nn) do { for (int i = 0; i < nn->total_weights; i++) { printf("%25.17e ", nn->weight[i]); } printf("\n"); } while (0)

static int train_and_test() {
    /* This will make the neural network initialize differently each run. */
    /* If you don't get a good result, try again for a different result. */
    srand(0);
    /* New network with 2 inputs,
     * 1 hidden layer of 2 neurons,
     * and 1 output. */
    nn0 = genann_init(nn_inputs, nn_layers, nn_hidden, nn_outputs);
    verify = nn0->weight;
#if 0
    static uint8_t memory[sizeof(ann_t) + ann_sizeof(nn_inputs, nn_layers, nn_hidden, nn_outputs)];
    nn1 = (ann_t*)memory;
    uint64_t seed = nanoseconds();
    ann.init(nn1, nn_inputs, nn_layers, nn_hidden, nn_outputs, ann.activation_sigmoid, ann.activation_sigmoid, seed,
        memory + sizeof(ann_t));
    ann.randomize(nn1);

    /* Train on the four labeled data points many times. */
    for (int i = 0; i < epochs; ++i) {
        for (int j = 0; j < countof(inputs); j++) {
            genann_train(nn0, (double*)&inputs[j], &output[j], 3.0);
            ann.train(nn1, (double*)&inputs[j], &output[j], 3.0);
//          dump_weights(nn0);
        }
    }
    /* Run the network and see what it predicts. */
    for (int j = 0; j < countof(inputs); j++) {
        printf("%1.f  %1.f is %1.f.\n", inputs[j].a, inputs[j].b, *genann_run(nn0, (double*)&inputs[j]));
    }
    for (int j = 0; j < countof(inputs); j++) {
        printf("%1.f  %1.f is %1.f.\n", inputs[j].a, inputs[j].b, *ann.inference(nn1, (double*)&inputs[j]));
    }
//  genann_free(nn0);
#endif
    return 0;
}

#if 0
static int ann_main() {
    static uint8_t memory[sizeof(ann_t) + ann_sizeof(nn_inputs, nn_layers, nn_hidden, nn_outputs)];
    nn1 = (ann_t*)memory;
    uint64_t seed = nanoseconds();
    ann.init(nn1, nn_inputs, nn_layers, nn_hidden, nn_outputs, ann.activation_sigmoid, ann.activation_sigmoid, seed,
        memory + sizeof(ann_t));
    ann.randomize(nn1);
    /* Train on the four labeled data points many times. */
    for (int i = 0; i < epochs; ++i) {
        for (int j = 0; j < countof(inputs); j++) {
            ann.train(nn1, (double*)&inputs[j], &output[j], 3.0);
//          dump_weights(nn1);
        }
    }
    /* Run the network and see what it predicts. */
    for (int j = 0; j < countof(inputs); j++) {
        printf("%1.f  %1.f is %1.f.\n", inputs[j].a, inputs[j].b, *ann.inference(nn1, (double*)&inputs[j]));
    }
    return 0;
}
#endif
static nnn_t net;

static void train() {
//  uint32_t seed = (uint32_t)nanoseconds();
    uint32_t seed = 0;
//  nnn.init(&net, nnn_tanh, seed);
    nnn.init(&net, nnn_sigmoid, seed);
    static const double learning_rate = 0.01;
    static const double weight_decay  = 0.01;
    static const double nudge         = 0.0000; // avoid local minima
    enum { epochs = 10000 };
    for (int i = 0; i < epochs; i++) {
        for (int j = 0; j < countof(inputs); j++) {
            static_assertion(sizeof(net.input)  == sizeof(inputs[j]));
            static_assertion(sizeof(net.output) == sizeof(output[j]));
            memcpy(net.input, &inputs[j], sizeof(net.input));
            nnn.train(&net, &output[j], learning_rate, weight_decay, nudge);
        }
        if (i == epochs - 1) {
            printf("last\n");
        }
        if (i % 1000 == 0) {
            double max_rma = 0;
            double avg_rma = 0;
            for (int j = 0; j < countof(inputs); j++) {
                memcpy(net.input, &inputs[j], sizeof(net.input));
                double max_err = 0;
                avg_rma += nnn.rma(&net, &output[j], &max_err);
                max_rma = max(max_rma, max_err);
            }
            avg_rma /= countof(inputs);
            printf("epoch: %d rma avg: %.15f max: %.15f\n", i, avg_rma, max_rma);
        }
    }
    printf("training done\n");
}

static void test() {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            net.input[0] = i;
            net.input[1] = j;
            nnn.inference(&net);
            printf("%d ^ %d = %f\n", i, j, net.output[0]);
        }
    }
//  nnn.dump(&net);
    printf("test done\n");
}

void main() {
    for (int i = 0; i < countof(randoms); i++) { randoms[i] = rand() / (double)RAND_MAX - 0.5; }
    ground_truth();
    train_and_test();
//  ann_main();
    for (int i = 0; i < nn1->total_weights; i++) {
        assertion(nn0->weight[i] == nn1->weight[i],
            "nn0.weight[%d] != nn1.weight[%d]\n nn0: %.17e\n nn1: %.17e\n", i, i,
            nn0->weight[i], nn1->weight[i]);
    }

//  train();
//  test();
}
