#pragma once
#include "stdint.h"

// naive neural network

#ifdef cplusplus
extern "C" {
#endif

enum { // activation functions
    nnn_relu = 1,
    nnn_sigmoid = 2,
    nnn_tanh = 3
};

enum { // network dimensions
    nnn_input  = 2,
    nnn_layers = 1,
    nnn_hidden = 2,
    nnn_output = 1
};

typedef double vector_input_t[nnn_input];
typedef double vector_hidden_t[nnn_hidden];
typedef vector_hidden_t matrix_hidden_t[nnn_hidden];
typedef double vector_output_t[nnn_output];

typedef struct nnn_s {
    vector_input_t  input;
    vector_input_t  weights_ih[nnn_hidden];
    matrix_hidden_t weights_hh[nnn_layers];
    vector_output_t weights_ho[nnn_hidden];
    vector_hidden_t biases_ih;
    vector_hidden_t biases_hh[nnn_layers]; // only [0..nnn_layers-1] are used
    vector_output_t biases_ho;
    double output[nnn_output];
    vector_hidden_t hidden[nnn_layers + 1]; // state
    double (*activation)(double x);
    double (*derivative)(double x); // derivative of activation
    uint32_t seed;
} nnn_t;

typedef struct nnn_if {
    void (*init)(nnn_t* net, int activation_function, uint32_t seed);
    void (*inference)(nnn_t* net);
    void (*train)(nnn_t* net, vector_output_t target,
        double learning_rate, double weight_decay, double nudge);
    // rma() returns average of root mean square error and maximum
    double (*rma)(nnn_t* net, vector_output_t target, double *max_rma);
    void (*dump)(nnn_t* net);
} nnn_if;

extern nnn_if nnn;

/*
Usage:
    enum { n = ??? };
    double inputs[n][nn_count] = { ??? };
    double targets[n][nn_count] = { ??? };
    nn_t net = {0};
    #ifdef DEBUG
    uint32_t seed = 0; // deterministic from run to next run
    #else
    uint32_t seed = (uint32_t)crt.nanoseconds();
    #endif
    nnn.init(&net, nnn_tanh, seed);
    // avoid local minima:
    static const double nudge         = 0.0001; // can be 0
    static const double learning_rate = 0.01;
    static const double weight_decay  = 0.01;
    enum { epochs = 10000 };
    for (int i = 0; i < epochs; i++) {
        double max_rma = 0;
        double avg_rma = 0;
        for (int j = 0; j < n; j++) {
            memcpy(net.input, inputs[j], sizeof(net.input));
            nnn.train(&net, targets[j], learning_rate, weight_decay, nudge);
        }
        for (int j = 0; j < countof(inputs); j++) {
            memcpy(net.input, inputs[j], sizeof(net.input));
            double max_err = 0;
            avg_rma += nnn.rma(&net, target[j], &max_err); // will call nnn.inference()
            max_rma = max(max_rma, max_err);
        }
        printf("epoch: %d rma avg: %.15f max: %.15f\n", i, avg_rma / countof(inputs), max_rma);
    }
*/

#ifdef cplusplus
} // extern "C"
#endif

