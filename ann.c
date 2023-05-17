#include "ann.h"
#include "rt.h"
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double ann_activation_sigmoid(double x) {
//  assertion(!isnan(x));
//  if (x < -45.0) { // ??? 2.8625186e-20
//      return 0;
//  } else if (x > 45.0) { // ??? 3.4934271e+19
//      return 1;
//  }
    double r = 1.0 / (1.0 + exp(-x));
    assertion(!isnan(r));
    return r;
}

static double ann_derivative_sigmoid(double s) {
    return s * (1 - s);
}

static double ann_activation_relu(double x) {
    return x > 0 ? x : 0;
}

static double ann_derivative_relu(double x) {
    return x > 0 ? 1 : 0;
}

static double ann_activation_tanh(double x) {
    return tanh(x);
}

static double ann_derivative_tanh(double x) {
    return 1.0 - x * x;
}

static double ann_activation_linear(double x) {
    return x;
}

static double ann_derivative_linear(double x) {
    return 1; (void)x; // unused
}

#if 0
static double ann_activation_threshold(double a) {
    return a > 0;
}

static double ann_derivative_threshold(double a) {
    return ???;
}

// if f(a) is the activation_threshold function, then its derivative
// f'(a) does not exist except at a = 0 where it is undefined.
// This is because a step function has an abrupt change in its value
// at the threshold, which means the function is not continuous at
// that point and therefore its derivative is undefined.
#endif

static uint64_t random64(ann_t* nn) {
 	const uint64_t s = nn->seed;
	const uint64_t z = (s ^ s >> 25) * (nn->seed += 0x6A5D39EAE12657AAULL);
	return z ^ (z >> 22);
}

static double random(ann_t* nn) {
    return (double)random64(nn) / (double)UINT64_MAX;
}

extern double randoms[1024];

static void ann_randomize(ann_t* nn) {
#if 1
    for (int64_t i = 0; i < nn->total_weights; i++) {
        nn->weight[i] = random(nn) - 0.5; // weights from -0.5 to 0.5
    }
#else
    for (int64_t i = 0; i < nn->total_weights; i++) {
        assertion(i < _countof(randoms));
        nn->weight[i] = randoms[i];
//      printf("%10.6f ", nn->weight[i]);
    }
//  printf("\n");
#endif
}

static double const* ann_inference(const ann_t* nn, const double* inputs) {
    double const* w = nn->weight;
    double* o = nn->output;
    double const* i = inputs;
    /* input -> hidden layer */
    for (int64_t j = 0; j < nn->hidden; j++) {
        double sum = *w++ * -1.0; // bias
        for (int64_t k = 0; k < nn->inputs; k++) {
            sum += *w++ * i[k];
        }
        *o++ = nn->activation_hidden(sum);
//      printf("o[0][%lld]=%.16e (ann)\n", j, *(o - 1));
    }
    // hidden layers, if any more than 1
    i = nn->output; // "i" is now first row of outputs
    for (int64_t h = 1; h < nn->layers; ++h) {
        for (int64_t j = 0; j < nn->hidden; j++) {
            double sum = *w++ * -1.0; // bias
            for (int64_t k = 0; k < nn->hidden; k++) {
                sum += *w++ * i[k];
            }
            *o++ = nn->activation_hidden(sum);
 //         printf("o[%lld][%lld]=%.16e (ann)\n", h, j, *(o - 1));
        }
        i += nn->hidden;
    }
    double const* output = o; // actuall last row of output
    // output layer is [nn->outputs] not the same as preceeding layers
    for (int64_t j = 0; j < nn->outputs; j++) {
        double sum = *w++ * -1.0; // bias
        for (int64_t k = 0; k < nn->hidden; k++) {
//          printf("i[%lld]=%.16e ", k, i[k]);
            sum += *w++ * i[k];
        }
//      printf("\nsum=%.16e (gen)\n", sum);
        *o++ = nn->activation_output(sum);
//      printf("%p output[%lld]=%.16e (ann)\n", o - 1, j, *(o - 1));
    }
    /* Sanity check that we used all weights and wrote all outputs. */
    assertion(w - nn->weight == nn->total_weights);
    assertion(o - nn->output == nn->total_neurons);
    assertion(output == nn->output + nn->hidden * nn->layers);
//  printf("%lld (gen)\n", nn->delta - nn->weight);
    assertion(o == nn->delta);
//  printf("return %p %.16e (gen)\n", output, *output);
    return output;
}

// extern double* verify;

void ann_train(ann_t* nn, double const* inputs, const double* truth, double learning_rate) {
    double const* output = ann.inference(nn, inputs);
    {   // calculate the output layer deltas.
        double const *o = output; /* First output. */
        assertion(o == nn->output + nn->hidden * nn->layers);
        double *d = nn->delta + nn->hidden * nn->layers; // first delta
        double const *t = truth; // pointer to the first grownd truth value
        for (int64_t j = 0; j < nn->outputs; j++) {
            *d++ = (*t - *o) * nn->derivative_output(*o);
//          printf("%10.6f ", *(d - 1));
            o++; t++;
        }
//      printf("\n");
    }
//  printf("output=%10.6f ***(1)\n", *nn->output);
    // hidden layer deltas, start at the last layer and work backwards.
    for (int64_t h = (int64_t)nn->layers - 1; h >= 0; --h) {
        /* Find first output and delta in this layer. */
        double const *o = nn->output + (h * nn->hidden);
        double *d = nn->delta + (h * nn->hidden);
        /* Find first delta in following layer (which may be hidden or output). */
        double const * const dd = nn->delta + ((h + 1) * nn->hidden);
        /* Find first weight in following layer (which may be hidden or output). */
        double const * const ww = nn->weight + ((nn->inputs + 1) * nn->hidden) + ((nn->hidden + 1) * nn->hidden * (h));
        printf("hidden[%lld] o.ofs=%lld\n", h, o - nn->output);
        for (int64_t j = 0; j < nn->hidden; j++) {
            double delta = 0;
            const int64_t n = h == nn->layers-1 ? nn->outputs : nn->hidden;
            for (int64_t k = 0; k < n; k++) {
                const double forward_delta = dd[k];
                printf("dd[%lld]: %25.17e ", k, dd[k]);
                const int64_t windex = k * (nn->hidden + 1) + (j + 1);
                const double forward_weight = ww[windex];
                delta += forward_delta * forward_weight;
//              printf("delta: %25.17e := forward_delta: %25.17e forward_weight: %25.17e\n", delta, forward_delta, forward_weight);
            }
            *d = nn->derivative_hidden(*o) * delta;
//          printf("d: %25.17e o: %25.17e delta: %25.17e", *d, *o, delta);
            d++; o++;
        }
        printf("(ann)\n");
    }
//  printf("output=%10.6f ***(2)\n", *nn->output);
    {   // Train the outputs.
        /* Find first output delta. */
        double const *d = nn->delta + nn->hidden * nn->layers; /* First output delta. */
        /* Find first weight to first output delta. */
        double *w = nn->weight +
                   (nn->inputs + 1) * nn->hidden + (nn->hidden + 1) * nn->hidden * (nn->layers - 1);
        /* Find first output in previous layer. */
//      double const * const i = nn->output + nn->inputs + (nn->hidden) * (nn->layers - 1);
        double const * const i = nn->output + (nn->hidden) * (nn->layers - 1);
        /* Set output layer weights. */
        for (int64_t j = 0; j < nn->outputs; j++) {
            *w++ += *d * learning_rate * -1.0; // bias
//          printf("output layer weights\n");
            for (int64_t k = 1; k < nn->hidden + 1; k++) {
                *w++ += *d * learning_rate * i[k - 1];
//              printf("i[%lld] %10.6f w %10.6f ", k-1, i[k - 1], *(w - 1));
            }
//          printf("\n");
            ++d;
        }
        assertion(w - nn->weight == nn->total_weights);
    }
    /* Train the hidden layers. */
    for (int64_t h = nn->layers - 1; h >= 0; h--) {
        /* Find first delta in this layer. */
        double const* d = nn->delta + (h * nn->hidden);
        /* Find first input to this layer. */
//      double const* i = nn->output + (h != 0
//              ? (nn->inputs + nn->hidden * (h - 1))
//              : 0);
        double const* i = (h == 0 ? inputs :
                  nn->output + nn->hidden * (h - 1));
        /* Find first weight to this layer. */
        double *w = nn->weight + (h != 0
                ? ((nn->inputs + 1) * nn->hidden + (nn->hidden + 1) * (nn->hidden) * (h - 1))
                : 0);
//      printf("hidden layer[%lld] i.ofs %lld\n", h, i - nn->output);
        for (int64_t j = 0; j < nn->hidden; j++) {
//          printf("hidden layer[%lld][%lld] weights w.ofs=%lld\n", h, j, w - nn->weight);
            *w++ += *d * learning_rate * -1.0; // bias
//          printf("w[0] (bias)=%25.17e d=%25.17e", *(w - 1), *d);
            for (int64_t k = 1; k < (h == 0 ? nn->inputs : nn->hidden) + 1; k++) {
                *w++ += *d * learning_rate * i[k - 1];
//              printf("i[%lld] %25.17e w %25.17e ", k - 1, i[k - 1], *(w - 1));
            }
//          printf(" (ann)\n");
            ++d;
        }
    }
//  for (int i = 0; i < nn->total_weights; i++) {
//      assertion(nn->weight[i] == verify[i],
//          "nn.weight[%d] != verify[%d]\n nn0: %.17e\n verify: %.17e\n", i, i,
//          nn->weight[i], verify[i]);
//  }
}

static ann_activation_t derivative_of(ann_activation_t af) {
    if (af == ann_activation_sigmoid) {
        return ann_derivative_sigmoid;
    } else if (af == ann_activation_tanh) {
        return ann_derivative_tanh;
    } else if (af == ann_activation_relu) {
        return ann_derivative_relu;
    } else if (af == ann_activation_linear) {
        return ann_derivative_linear;
    } else {
        assertion(false);
        return NULL;
    }
}

static void ann_init(ann_t* nn,
        int64_t inputs, int64_t layers, int64_t hidden, int64_t outputs,
        ann_activation_t activation_hidden, ann_activation_t activation_output,
        uint64_t seed,
        void* memory) {
    assertion(layers > 0);
    assertion(inputs > 0);
    assertion(outputs > 0);
    assertion(hidden > 0);
    memset(nn, 0, sizeof(*nn));
    nn->inputs = inputs;
    nn->layers = layers;
    nn->hidden = hidden;
    nn->outputs = outputs;

    const int64_t hidden_weights = (inputs + 1) * hidden + (layers - 1) * (hidden + 1) * hidden;
    const int64_t output_weights = (hidden + 1) * outputs;
    nn->total_weights = hidden_weights + output_weights;
    nn->total_neurons = inputs + hidden * layers + outputs;


    printf("total_weights : %lld\n", nn->total_weights );
    printf("hidden_weights: %lld\n", hidden_weights);
    printf("output_weights: %lld\n", output_weights);
    printf("total_neurons : %lld\n", nn->total_neurons );


    // https://gist.github.com/tommyettinger/e6d3e8816da79b45bfe582384c2fe14a
    nn->seed = seed | 1; // first seed must be odd
    uint64_t size = sizeof(double) * (nn->total_weights + nn->total_neurons * 2);
    assertion(size == ann_sizeof(inputs, layers, hidden, outputs));
    (void)size;
    double* p = (double*)memory;
    // memory layout
    // weight_ih[inputs + 1][hidden]:  input + bias -> first hidden layer
    // weight_hh[layers - 1][hidden + 1][hidden]: layers - 1 can be zero
    // output[layers][hidden]: result for each layer
    // output[outputs]: final reasults
    nn->weight = p;
    nn->output = nn->weight + nn->total_weights;
    nn->delta  = nn->output + nn->total_neurons;
    nn->activation_hidden = activation_hidden;
    nn->activation_output = activation_output;
    nn->derivative_hidden = derivative_of(activation_hidden);
    nn->derivative_output = derivative_of(activation_output);
    printf("size=%lld delta.ofs=%lld\n", size / sizeof(double), nn->delta - (double*)memory);
}

ann_if ann = {
    .init = ann_init,
    .randomize = ann_randomize,
    .inference = ann_inference,
    .train = ann_train,
    .activation_sigmoid = ann_activation_sigmoid,
    .activation_tanh    = ann_activation_tanh,
    .activation_relu    = ann_activation_relu,
    .activation_linear  = ann_activation_linear
};
