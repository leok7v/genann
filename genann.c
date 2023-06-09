/*
 * GENANN - Minimal C Artificial Neural Network
 *
 * Copyright (c) 2015-2018 Lewis Van Winkle
 *
 * http://CodePlea.com
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgement in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 */

#include "genann.h"
#include "fffc.h"

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if 0
#ifndef genann_act
#define genann_act_hidden genann_act_hidden_indirect
#define genann_act_output genann_act_output_indirect
#else
#define genann_act_hidden genann_act
#define genann_act_output genann_act
#endif

#define LOOKUP_SIZE 4096

double genann_act_hidden_indirect(const struct genann *ann, double a) {
    return ann->activation_hidden(ann, a);
}

double genann_act_output_indirect(const struct genann *ann, double a) {
    return ann->activation_output(ann, a);
}

const double sigmoid_dom_min = -15.0;
const double sigmoid_dom_max = 15.0;
double interval;
double lookup[LOOKUP_SIZE];

#ifdef __GNUC__
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#define unused          __attribute__((unused))
#else
#define likely(x)       x
#define unlikely(x)     x
#define unused
#pragma warning(disable : 4996) /* For fscanf */
#endif

double genann_act_sigmoid(const genann *ann unused, double a) { (void)ann;
    if (a < sigmoid_dom_min) return 0;
    if (a > sigmoid_dom_max) return 1;
    return 1.0 / (1 + exp(-a));
}

void genann_init_sigmoid_lookup(const genann *ann) { (void)ann;
        const double f = (sigmoid_dom_max - sigmoid_dom_min) / LOOKUP_SIZE;
        int i;

        interval = LOOKUP_SIZE / (sigmoid_dom_max - sigmoid_dom_min);
        for (i = 0; i < LOOKUP_SIZE; ++i) {
            lookup[i] = genann_act_sigmoid(ann, sigmoid_dom_min + f * i);
        }
}

double genann_act_sigmoid_cached(const genann *ann unused, double a) { (void)ann;
    assertion(!isnan(a));

    if (a < sigmoid_dom_min) return lookup[0];
    if (a >= sigmoid_dom_max) return lookup[LOOKUP_SIZE - 1];

    size_t j = (size_t)((a-sigmoid_dom_min)*interval+0.5);

    /* Because floating point... */
    if (unlikely(j >= LOOKUP_SIZE)) return lookup[LOOKUP_SIZE - 1];

    return lookup[j];
}
#endif

double genann_act_sigmoid(double x) {
    assertion(!isnan(x));
//  if (x < -45.0) { // ??? 2.8625186e-20
//      return 0;
//  } else if (x > 45.0) { // ??? 3.4934271e+19
//      return 1;
//  }
    double r = 1.0 / (1.0 + exp(-x));
    assertion(!isnan(r));
    return r;
}

double genann_derivative_sigmoid(double s) {
    assertion(!isnan(s));
    return s * (1 - s);
}

double genann_act_relu(double x) {
    assertion(!isnan(x));
    return x > 0 ? x : 0;
}

double genann_derivative_relu(double r) {
    assertion(!isnan(r));
    return r > 0 ? 1 : 0;
}

double genann_act_tanh(double x) {
    assertion(!isnan(x));
    return tanh(x);
}

double genann_derivative_tanh(double t) {
    assertion(!isnan(t));
    return 1.0 - t * t;
}

double genann_act_linear(double x) {
    assertion(!isnan(x));
    return x;
}

double genann_derivative_linear(double y) {
    assertion(!isnan(y));
    return 1; (void)y; // unused
}

double genann_act_threshold(double a) {
    return a > 0;
}

genann *genann_init(int inputs, int hidden_layers, int hidden, int outputs) {
    assertion(hidden_layers >= 0);
    assertion(inputs >= 1);
    assertion(outputs >= 1);
    assertion(hidden_layers > 0 && hidden > 0 || hidden_layers == 0 && hidden == 0);
//  traceln("inputs: %d", inputs);
//  traceln("hidden_layers: %d", hidden_layers);
//  traceln("hidden: %d", hidden);
//  traceln("outputs: %d", outputs);
    const int hidden_weights = hidden_layers ? (inputs+1) * hidden + (hidden_layers-1) * (hidden+1) * hidden : 0;
//  traceln("hidden_weights: %d", hidden_weights);
    const int output_weights = (hidden_layers ? (hidden+1) : (inputs+1)) * outputs;
//  traceln("output_weights: %d", output_weights);
    const int total_weights = (hidden_weights + output_weights);
//  traceln("total_weights: %d", total_weights);
    const int total_neurons = (inputs + hidden * hidden_layers + outputs);
    /* Allocate extra size for weights, outputs, and deltas. */
    const int size = sizeof(genann) + sizeof(double) * (total_weights + total_neurons + (total_neurons - inputs));
    genann *ret = malloc(size);
    if (!ret) return 0;
    ret->inputs = inputs;
    ret->hidden_layers = hidden_layers;
    ret->hidden = hidden;
    ret->outputs = outputs;
    ret->total_weights = total_weights;
    ret->total_neurons = total_neurons;
    /* Set pointers. */
    ret->weight = (double*)((char*)ret + sizeof(genann));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;
    ret->activation_hidden = genann_act_sigmoid; // genann_act_sigmoid_cached;
    ret->activation_output = genann_act_sigmoid; // genann_act_sigmoid_cached;
//  genann_init_sigmoid_lookup(ret);
#if 0
    int64_t bytes = fffc_inference_memory_size(inputs, hidden_layers, hidden, outputs);
#else
    int64_t bytes = fffc_training_memory_size(inputs, hidden_layers, hidden, outputs);
#endif
    ret->nn1 = (fffc_t*)malloc(bytes);
    uint64_t seed = 1;
    fffc.init(ret->nn1, bytes, seed, inputs, hidden_layers, hidden, outputs);
    assertion(ret->total_weights == fffc_weights_count(inputs, hidden_layers, hidden, outputs),
              "total_weights: %d fffc_weights_count(%d, %d, %d, %d): %d",
              ret->total_weights, inputs, hidden_layers, hidden, outputs,
              fffc_weights_count(inputs, hidden_layers, hidden, outputs));
    genann_randomize(ret);
    return ret;
}

genann *genann_read(FILE *in) {
    int inputs, hidden_layers, hidden, outputs;
    int rc;

    errno = 0;
    rc = fscanf(in, "%d %d %d %d", &inputs, &hidden_layers, &hidden, &outputs);
    if (rc < 4 || errno != 0) {
        perror("fscanf");
        return NULL;
    }

    genann *ann = genann_init(inputs, hidden_layers, hidden, outputs);

    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        errno = 0;
        rc = fscanf(in, " %le", ann->weight + i);
        if (rc < 1 || errno != 0) {
            perror("fscanf");
            genann_free(ann);

            return NULL;
        }
    }

    return ann;
}


genann *genann_copy(genann const *ann) {
    const int size = sizeof(genann) + sizeof(double) * (ann->total_weights + ann->total_neurons + (ann->total_neurons - ann->inputs));
    genann *ret = malloc(size);
    if (!ret) return 0;

    memcpy(ret, ann, size);

    /* Set pointers. */
    ret->weight = (double*)((char*)ret + sizeof(genann));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;

    return ret;
}


void genann_randomize(genann *ann) {
    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        double r = GENANN_RANDOM();
        /* Sets weights from -0.5 to 0.5. */
        ann->weight[i] = r - 0.5;
        if (ann->nn1->layers != 0) {
            ann->nn1->iw[i] = ann->weight[i]; // TODO: (temp) sync randomization
        } else {
            ann->nn1->ow[i] = ann->weight[i];
            assert(ann->nn1->iw == null);
            assert(ann->nn1->hw == null);
        }
    }
}

void genann_free(genann *ann) {
    /* The weight, output, and delta pointers go to the same buffer. */
    free(ann->nn1);
    free(ann);
}

static genann_actfun derivative_of(genann_actfun af) {
    if (af == genann_act_sigmoid) {
        return genann_derivative_sigmoid;
    } else if (af == genann_act_tanh) {
        return genann_derivative_tanh;
    } else if (af == genann_act_relu) {
        return genann_derivative_relu;
    } else if (af == genann_act_linear) {
        return genann_derivative_linear;
    } else {
        assertion(false);
        return NULL;
    }
}

// TODO: remove me - hack
static genann_actfun fffc_activation_of(genann_actfun af) {
    if (af == genann_act_sigmoid) {
        return fffc.sigmoid;
    } else if (af == genann_act_tanh) {
        return fffc.tanh;
    } else if (af == genann_act_relu) {
        return fffc.relu;
    } else if (af == genann_act_linear) {
        return fffc.linear;
    } else {
        assertion(false);
        return NULL;
    }
}


// TODO: make static move to fffc.h

// fffc_dot_product_w_x_i(w[n + 1], i[n], n) treats first element of w as bias

static inline fffc_type fffc_w_x_i(const fffc_type* restrict w,
        const fffc_type* restrict i, int64_t n) {
    assertion(n > 0);
    fffc_type sum = *w++ * -1.0; // bias;
    const fffc_type* e = i + n;
    while (i < e) { sum += *w++ * *i++; }
    return sum;
}

// TODO: move to fffc.h
/* static */ static fffc_activation_t fffc_derivative_of(fffc_activation_t af) {
    if (af == fffc.sigmoid) {
        return fffc_derivative_sigmoid;
    } else if (af == fffc.tanh) {
        return fffc_derivative_tanh;
    } else if (af == fffc.relu) {
        return fffc_derivative_relu;
    } else if (af == fffc.linear) {
        return fffc_derivative_linear;
    } else {
        assertion(false);
        return NULL;
    }
}

double const *genann_run(genann *ann, double const *inputs) {
    ann->nn1->activation_hidden = fffc_activation_of(ann->activation_hidden);
    ann->nn1->activation_output = fffc_activation_of(ann->activation_output);
    ann->derivative_hidden = derivative_of(ann->activation_hidden);
    ann->derivative_output = derivative_of(ann->activation_output);
    double const *w = ann->weight;
    double *o = ann->output + ann->inputs;
    double const *i = ann->output;
    /* Copy the inputs to the scratch area, where we also store each neuron's
     * output, for consistency. This way the first layer isn't a special case. */
    memcpy(ann->output, inputs, sizeof(double) * ann->inputs);
    int h, j, k;
    if (!ann->hidden_layers) {
        double *ret = o;
        for (j = 0; j < ann->outputs; ++j) {
//          traceln("ouput[%d] w[0]: %.16e", j, *w);
            double sum = *w++ * -1.0;
            for (k = 0; k < ann->inputs; ++k) {
//              traceln("ouput[%d] w[%d]: %.16e i[%d]: %.16e", j, k + 1, *w, k, i[k]);
                sum += *w++ * i[k];
            }
            *o++ = ann->activation_output(sum);
//          traceln("ouput[%d]: %.16e", j, *(o - 1));
        }
        fffc.inference(ann->nn1, inputs);
        bool equal = memcmp(ret, ann->nn1->output, ann->outputs * sizeof(double)) == 0;
        if (!equal) {
            traceln("output: ann: %.15e nn: %.15e", *ret, *ann->nn1->output);
        }
        assertion(equal);
        return ret;
    }
    /* Figure input layer */
    for (j = 0; j < ann->hidden; ++j) {

//      traceln("w[0] %.16e", w[0]);
//      for (k = 0; k < ann->inputs; ++k) {
//          traceln("w[%d] %.16e i[%d]: %.16e", k + 1, w[k + 1], k, i[k]);
//      }

        double sum = *w++ * -1.0;
        for (k = 0; k < ann->inputs; ++k) {
            sum += *w++ * i[k];
        }
        *o++ = ann->activation_hidden(sum);
//      traceln("o[0][%d]: %.16e", j, *(o - 1));
    }
    i += ann->inputs;
    /* Figure hidden layers, if any. */
    for (h = 1; h < ann->hidden_layers; ++h) {
        for (j = 0; j < ann->hidden; ++j) {
            double sum = *w++ * -1.0;
            for (k = 0; k < ann->hidden; ++k) {
                sum += *w++ * i[k];
            }
            *o++ = ann->activation_hidden(sum);
//          traceln("o[%d][%d]: %.16e", h, j, *(o - 1));
        }
        i += ann->hidden;
    }
    double const *ret = o;
    /* Figure output layer. */
    for (j = 0; j < ann->outputs; ++j) {
//      traceln("ouput[%d] w[0]: %.16e", j, *w);
        double sum = *w++ * -1.0;
        for (k = 0; k < ann->hidden; ++k) {
//          traceln("ouput[%d] w[%d]: %.16e i[%d]: %.16e", j, k + 1, *w, k, i[k]);
            sum += *w++ * i[k];
        }
        *o++ = ann->activation_output(sum);
//      traceln("ouput[%d]: %.16e", j, *(o - 1));
    }
    /* Sanity check that we used all weights and wrote all outputs. */
    assertion(w - ann->weight == ann->total_weights);
    assertion(o - ann->output == ann->total_neurons);
    // call fffc_inference and compare results bit to bit
//  traceln("fffc_iw_count: %d", fffc_iw_count(ann->inputs, ann->hidden));
//  traceln("fffc_hw_count: %d", fffc_hw_count(ann->hidden_layers, ann->hidden));
//  traceln("total_weights: %d", ann->total_weights);
    assertion(ann->total_weights ==
        fffc_weights_count(ann->inputs, ann->hidden_layers, ann->hidden, ann->outputs));
    memcpy(ann->nn1->iw, ann->weight, sizeof(fffc_type) * ann->total_weights);
    fffc.inference(ann->nn1, inputs);
    bool equal = memcmp(ret, ann->nn1->output, ann->outputs * sizeof(double)) == 0;
    if (!equal) {
        traceln("output: ann: %.15e nn: %.15e", *ret, *ann->nn1->output);
    }
    assertion(equal);
    return ret;
}

// TODO: move to fffc.c

void genann_train(genann *ann, double const *inputs, double const *desired_outputs, double learning_rate) {
    /* To begin with, we must run the network forward. */
    const double* ret = genann_run(ann, inputs);
    int h, j, k;
    /* First set the output layer deltas. */
    {
        double const *o = ann->output + ann->inputs + ann->hidden * ann->hidden_layers; /* First output. */
        assertion(ret == o);
        double *d = ann->delta + ann->hidden * ann->hidden_layers; /* First delta. */
        double const *t = desired_outputs; /* First desired output. */
        /* Set output layer deltas. */
        for (j = 0; j < ann->outputs; ++j) {
            *d++ = (*t - *o) * ann->derivative_output(*o);
//          traceln("o[%d]: %.17e od[%d]: %.17e", j, *o, j, *(d - 1));
            ++o; ++t;
        }
    }
    /* Set hidden layer deltas, start on last layer and work backwards. */
    /* Note that loop is skipped in the case of hidden_layers == 0. */
    for (h = ann->hidden_layers - 1; h >= 0; --h) {
        /* Find first output and delta in this layer. */
        double const *o = ann->output + ann->inputs + (h * ann->hidden);
        double *d = ann->delta + (h * ann->hidden);
        /* Find first delta in following layer (which may be hidden or output). */
        double const * const dd = ann->delta + ((h+1) * ann->hidden);
        /* Find first weight in following layer (which may be hidden or output). */
        double const * const ww = ann->weight + ((ann->inputs+1) * ann->hidden) + ((ann->hidden+1) * ann->hidden * (h));
        for (j = 0; j < ann->hidden; ++j) {
            double delta = 0;
            for (k = 0; k < (h == ann->hidden_layers-1 ? ann->outputs : ann->hidden); ++k) {
                const double forward_delta = dd[k];
//              traceln("dd[%d]: %25.17e", k, dd[k]);
                const int windex = k * (ann->hidden + 1) + (j + 1);
                const double forward_weight = ww[windex];
                delta += forward_delta * forward_weight;
//              traceln("delta: %25.17e := forward_delta: %25.17e "
//                      "forward_weight: %25.17e\n", delta, forward_delta,
//                      forward_weight);
            }
            *d = ann->derivative_hidden(*o) * delta;
//          traceln("d[%d]: %25.17e o: %25.17e delta: %25.17e", j, *d, *o, delta);
            ++d; ++o;
        }
    }
    /* Train the outputs. */
    {
        /* Find first output delta. */
        double const *d = ann->delta + ann->hidden * ann->hidden_layers; /* First output delta. */
        /* Find first weight to first output delta. */
        double *w = ann->weight + (ann->hidden_layers
                ? ((ann->inputs+1) * ann->hidden + (ann->hidden+1) * ann->hidden * (ann->hidden_layers-1))
                : (0));
        /* Find first output in previous layer. */
        double const * const i = ann->output + (ann->hidden_layers
                ? (ann->inputs + (ann->hidden) * (ann->hidden_layers-1))
                : 0);
        /* Set output layer weights. */
        for (j = 0; j < ann->outputs; ++j) {
            *w++ += *d * learning_rate * -1.0;
            for (k = 1; k < (ann->hidden_layers ? ann->hidden : ann->inputs) + 1; ++k) {
                *w++ += *d * learning_rate * i[k - 1];
//              traceln("output[%d] i[%d] %25.17e w %25.17e", j, k - 1, i[k - 1], *(w - 1));
            }
            ++d;
        }
        assertion(w - ann->weight == ann->total_weights);
    }
    /* Train the hidden layers. */
    for (h = ann->hidden_layers - 1; h >= 0; --h) {
        /* Find first delta in this layer. */
        double const *d = ann->delta + (h * ann->hidden);
        /* Find first input to this layer. */
        double const *i = ann->output + (h
                ? (ann->inputs + ann->hidden * (h-1))
                : 0);
        /* Find first weight to this layer. */
        double *w = ann->weight + (h
                ? ((ann->inputs+1) * ann->hidden + (ann->hidden+1) * (ann->hidden) * (h-1))
                : 0);
        for (j = 0; j < ann->hidden; ++j) {
//          traceln("hidden layer[%d][%d] weights w.ofs=%lld\n", h, j, w - ann->weight);
            *w++ += *d * learning_rate * -1.0;
//          traceln("w[0] (bias)=%25.17e d=%25.17e", *(w - 1), *d);
            for (k = 1; k < (h == 0 ? ann->inputs : ann->hidden) + 1; ++k) {
                *w++ += *d * learning_rate * i[k - 1];
//              traceln("i[%d] %25.17e w %25.17e ", k - 1, i[k - 1], *(w - 1));
            }
            ++d;
        }
    }
    /////////////////////////////////////////////
    fffc.train(ann->nn1, inputs, desired_outputs, learning_rate);
    for (int i = 0; i < ann->total_weights; i++) {
        fffc_type* weights = ann->nn1->iw != null ? ann->nn1->iw : ann->nn1->ow;
        assertion(ann->weight[i] == weights[i],
            "ann.weight[%d] != nn[%d]\n ann: %25.17e\n nn1: %25.17e\n",
            i, i, ann->weight[i], weights[i]);
    }
}


void genann_write(genann const *ann, FILE *out) {
    fprintf(out, "%d %d %d %d", ann->inputs, ann->hidden_layers, ann->hidden, ann->outputs);

    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        fprintf(out, " %.20e", ann->weight[i]);
    }
}

#define FFFC_IMPLEMENTATION
#include "fffc.h"

