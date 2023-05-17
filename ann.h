#pragma once
#ifndef ANN_H
#define ANN_H
/* based on https://github.com/codeplea/genann
 * GENANN - Minimal C Artificial Neural Network
 * Copyright (c) 2015-2018 Lewis Van Winkle
 * http://CodePlea.com
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgement in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef double (*ann_activation_t)(double a);

typedef struct ann_s {
    /* How many inputs, outputs, and hidden neurons. */
    int64_t inputs, layers, hidden, outputs;
    /* Which activation function to use for hidden neurons. Default: gennann_act_sigmoid_cached*/
    ann_activation_t activation_hidden;
    ann_activation_t derivative_hidden;
    /* Which activation function to use for output. Default: gennann_act_sigmoid_cached*/
    ann_activation_t activation_output;
    ann_activation_t derivative_output;
    /* Total number of weights, and size of weights buffer. */
    int64_t total_weights; // [layers][hidden][hiden+1] first element is bias
    /* Total number of neurons + inputs and size of output buffer. */
    int64_t total_neurons;
    /* All weights (total_weights long). */
    double *weight;
    /* Stores input array and output of each neuron (total_neurons long). */
    double *output;
    /* Stores delta of each hidden and output neuron (total_neurons - inputs long). */
    double *delta;
    uint64_t seed; // initial seed for random64() generator used for this network
} ann_t;

#define ann_sizeof(inputs, layers, hidden, output) (                                  \
    sizeof(double) * (                                                                \
        /* weights: */                                                                \
        (inputs + 1) * hidden + (layers - 1) * (hidden + 1) * hidden + (hidden + 1) + \
        /* output */                                                                  \
        (inputs + hidden * layers + output) +                                         \
        /* deltas */                                                                  \
        (hidden * layers + output))                                                   \
)

typedef struct ann_if { // interface
    // uint8_t memory[sizeof(ann_t) + ann_sizeof(inputs, layers, hidden, outputs)];
    // ann_t* nn = (ann_t*)memory;
    // seed = nanoseconds();
    // ann.init(nn, memory, ann.activation_sigmoid, ann.activation_sigmoid, seed);
    // train:
    // ann.randomize(nn);
    // deloy:
    //
    void (*init)(ann_t* nn,
        int64_t inputs, int64_t layers, int64_t hidden, int64_t outputs,
        ann_activation_t activation_hidden, ann_activation_t activation_output,
        uint64_t seed,
        void* memory);
    void (*randomize)(ann_t* nn);
    double const* (*inference)(const ann_t* ann, const double* inputs); // returns output
    void (*train)(ann_t* nn, const double* inputs, const double* truth, double learning_rate);
    // available activation functions:
    const ann_activation_t activation_sigmoid;
    const ann_activation_t activation_relu;
    const ann_activation_t activation_tanh;
    const ann_activation_t activation_linear;
} ann_if;

extern ann_if ann;

#ifdef __cplusplus
}
#endif

#endif /*ann_H*/
