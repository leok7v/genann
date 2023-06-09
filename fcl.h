/* Copyright (c) 2023 Leo Kuznetsov
 * derived of
 * GENANN - Minimal C Artificial Neural Network
 * Copyright (c) 2015-2018 Lewis Van Winkle
 * http://CodePlea.com
 */

#ifndef FCL_H
#define FCL_H
#include <stdint.h>
#include <rt.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef fcl_type
#define fcl_type double // TODO: change to float when done with genann
#endif

typedef fcl_type (*fcl_activation_t)(fcl_type a);

// inference()
//
// state for first hidden layer is the output of input layer as:
//    state[0][i] = activation_hidden(sum(input[j] * iw[j + 1]) + ihw[0])
// where iw[0] is input bias
//
// for subsequent layers "k" > 0
//    state[next][i] = activation_hidden(sum(state[k - 1][j] * hw[k - 1][i][j + 1]) + hw[k - 1][i][0])
// hw[k - 1][i][0] is hidden layer neuron bias
//
// for inference next = (k + 1) % 2
// for training  next = k + 1
//
// output[i] = activation_output(sum(state[layers][j] * hw[layers][i][j + 1]) + hw[layers][i][0])
// note dimension of hw[layers + 1]

typedef /* begin_packed */ struct fcl_s {
    int64_t inputs;  // number elements in input vector
    int64_t layers;  // number of hidden layers, must be >= 1
    int64_t hidden;  // number of neurons in each hidden layer
    int64_t outputs; // number elements in output vector
    // iw and hw are located contiguously in memory and can be written out
    // as chunk of binary data
    // for each hidden neuron of first layer bias and weight for each input element
    fcl_type* iw; // [hidden][inputs + 1] == null if layers == 0 && hidden == 0
    // for each neuron of each hidden layer bias and weights vector of previous layer weights:
    fcl_type* hw;  // [layers][hidden][hidden + 1] or null for layers == 0
    fcl_type* ow;  // [outputs][hidden + 1] output weights
    fcl_type* output; // [outputs] elements of output vector
    // training only data:
    fcl_type* state; // [2][hidden] for inference() | [layers][hidden] for train()
    // delta == null for inference only fcl
    fcl_type* delta; // null inference() | [layers][hidden] for train()
    // output deltas:
    fcl_type* od;    // null inference() | [outputs] for train()
    // initial seed for random64() generator used for this network
    uint64_t seed;
    // for known fcl.sigmoid, tanh, relu, linear activation functions
    // nn.train() knows the derivatives
    fcl_activation_t activation_hidden; // activation function to use for hidden neurons
    fcl_activation_t derivative_hidden; // derivative of hidden activation function
    fcl_activation_t activation_output; // activation function to use for output elements
    fcl_activation_t derivative_output; // derivative of output activation function
} /* end_packed */ fcl_t;

typedef struct fcl_if { // interface
    void (*init)(fcl_t* nn, int64_t bytes, uint64_t seed,
                 int64_t inputs, int64_t layers, int64_t hidden, int64_t outputs);
    fcl_type (*random)(fcl_t* nn, fcl_type lo, fcl_type hi); // next random after seed
    // randomize initializes each wieight and bias [lo..hi] starting with seed
    void (*randomize)(fcl_t* nn, uint64_t seed, fcl_type lo, fcl_type hi);
    // inference() returns pointer to output array of elements
    fcl_type* (*inference)(fcl_t* nn, const fcl_type* input);
    void (*train)(fcl_t* nn, const fcl_type* input, const fcl_type* truth,
                  fcl_type learning_rate);
    // available activation functions:
    const fcl_activation_t sigmoid;
    const fcl_activation_t tanh;
    const fcl_activation_t relu;
    const fcl_activation_t linear;
} fcl_if;

extern fcl_if fcl;

// TODO: delete from here and make static
fcl_type fcl_derivative_sigmoid(fcl_type s);
fcl_type fcl_derivative_relu(fcl_type x);
fcl_type fcl_derivative_tanh(fcl_type x);
fcl_type fcl_derivative_linear(fcl_type x);

// Memory management helpers.
// All counts are a number of fcl_type elements not bytes

#define fcl_iw_count(inputs, hidden) ((hidden) == 0 ? \
    0 : ((hidden) * ((inputs) + 1)))
#define fcl_hw_count(layers, hidden) ((layers) == 0 ? \
    0 : (((layers) - 1)  * (hidden) * ((hidden) + 1)))
#define fcl_ow_count(inputs, hidden, outputs) ((hidden) == 0 ? \
    (outputs) * ((inputs) + 1) : ((outputs) * ((hidden) + 1)))

#define fcl_weights_count(inputs, layers, hidden, outputs) (             \
   (layers > 0 ?                                                          \
   (fcl_iw_count(inputs, hidden) + fcl_hw_count(layers, hidden)) : 0) + \
    fcl_ow_count(inputs, hidden, outputs)                  )

#define fcl_inference_state_count(hidden) (2 * (hidden))
#define fcl_training_state_count(layers, hidden) ((layers) * (hidden))
#define fcl_training_delta_count(layers, hidden) ((layers) * (hidden))

// input, [layers][hidden][hidden+1], output
#define fcl_weights_memory_iho(inputs, layers, hidden, outputs) \
    fcl_type iw[hidden][inputs + 1];                        \
    fcl_type hw[layers - 1][hidden][hidden + 1];            \
    fcl_type ow[outputs][hidden + 1]

// input, output
#define fcl_weights_memory_io(inputs, outputs) \
    fcl_type ow[outputs][inputs + 1]

#define fcl_weight_memory_size(inputs, layers, hidden, outputs) \
    ((fcl_iw_count(inputs, hidden) +                            \
      fcl_hw_count(layers, hidden) +                            \
      fcl_ow_count(inputs, hidden, outputs)) * sizeof(fcl_type))

#define fcl_base_memory_iho(inputs, layers, hidden, outputs) \
    fcl_weights_memory_iho(inputs, layers, hidden, outputs); \
    fcl_type output[outputs]

#define fcl_base_memory_io(inputs, outputs)  \
    fcl_weights_memory_io(inputs,  outputs); \
    fcl_type output[outputs]

#define fcl_base_memory_size(inputs, layers, hidden, outputs) ( \
    fcl_weight_memory_size(inputs, layers, hidden, outputs) +   \
    /* output: */(outputs) * sizeof(fcl_type)                   \
)

#define fcl_inference_memory_iho(inputs, layers, hidden, outputs) \
    fcl_base_memory_iho(inputs, layers, hidden, outputs);         \
    fcl_type state[2][hidden]

#define fcl_inference_memory_io(inputs, outputs) \
    fcl_base_memory_io(inputs, outputs)

#define fcl_inference_memory_size(inputs, layers, hidden, outputs) ( \
    fcl_base_memory_size(inputs, layers, hidden, outputs) +          \
    fcl_inference_state_count(hidden) * sizeof(fcl_type) +          \
    sizeof(fcl_t)                                                    \
)

#define fcl_training_memory_iho(inputs, layers, hidden, outputs) \
    fcl_base_memory_iho(inputs, layers, hidden, outputs);        \
    fcl_type state[layers][hidden];                              \
    fcl_type delta[layers][hidden];                              \
    fcl_type od[outputs]
//  ^^^ memory layout is important "od" must be imediately after "delta"

#define fcl_training_memory_io(inputs, outputs)    \
    fcl_base_memory_io(inputs, outputs);           \
    fcl_type od[outputs]
//  ^^^ memory layout is important "od" must be imediately after "ow"

#define fcl_training_memory_size(inputs, layers, hidden, outputs) ( \
    fcl_base_memory_size(inputs, layers, hidden, outputs) +         \
   (fcl_training_state_count(layers, hidden) +                      \
    fcl_training_delta_count(layers, hidden) +                      \
    /* od: */ (outputs)) * sizeof(fcl_type) +                       \
    sizeof(fcl_t)                                                   \
)

#define fcl_io_memory_size(inputs, layers, hidden, outputs) (        \
    fcl_base_memory_size(inputs, layers, hidden, outputs) +          \
    fcl_inference_state_count(hidden) * sizeof(fcl_type) +          \
    (uint8_t*)&((fcl_t*)(0))->iw - (uint8_t*)((fcl_t*)(0))          \
)

/*
    How to use:
    enum { inputs = 2, layers = 1, hidden = 2, outputs = 1};
    int64_t bytes = fcl_training_memory_size(inputs, layers, hidden, outputs);
    fcl_t* nn = (fcl_t*)malloc(bytes);
    if (nn != null) {
        uint64_t seed = 1;
        fcl.init(nn, bytes, seed, inputs, layers, hidden, outputs);
        nn->activation_hidden = fcl.sigmoid;
        fcl_type inputs[4][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
        fcl_type xor[4] = { 0, 1, 1, 0 };
        fcl_type learning_rate = 3; // no idea why 3?
        for (int32_t epoch = 0; epoch < 500; epoch++) {
            for (int32_t i = 0; i < countof(inputs); i++) {
                fcl.train(nn, inputs[i], &xor[i], learning_rate);
            }
        }
        for (int32_t i = 0; i < countof(inputs); i++) {
            fcl_type* output = nn->inference(nn, input);
            printf("%.1f ^ %.1f = %.1f\n", inputs[i][0], inputs[i][1], *output);
        }
        free(nn);
    }


*/


#endif // fcl_H

#ifdef fcl_IMPLEMENTATION

static inline void fcl_check_sizes_and_structs_correctness() {
    // C99 does not have constexpr thus just generic check for a match
    {
        enum { inputs = 123, layers = 234, hidden = 456, outputs = 567 };
        typedef struct {
            fcl_weights_memory_iho(inputs, layers, hidden, outputs);
        } fcl_weights_memory_iho_t;
        static_assertion(fcl_weight_memory_size(inputs, layers, hidden, outputs) ==
            sizeof(fcl_weights_memory_iho_t));
        typedef struct {
            fcl_inference_memory_iho(inputs, layers, hidden, outputs);
        } fcl_inference_memory_t;
        static_assertion(fcl_inference_memory_size(inputs, layers, hidden, outputs) ==
                  sizeof(fcl_inference_memory_t) + sizeof(fcl_t));
        typedef struct {
            fcl_training_memory_iho(inputs, layers, hidden, outputs);
        } fcl_training_memory_t;
        static_assertion(fcl_training_memory_size(inputs, layers, hidden, outputs) ==
                  sizeof(fcl_training_memory_t) + sizeof(fcl_t));
    }
    // Microsoft implementation of C99 and C17 does not allow zero size arrays
    {
        enum { inputs = 123, layers = 0, hidden = 0, outputs = 567 };
        typedef struct {
            fcl_weights_memory_io(inputs, outputs);
        } fcl_weights_memory_io_t;
        static_assertion(fcl_weight_memory_size(inputs, layers, hidden, outputs) ==
            sizeof(fcl_weights_memory_io_t));
        typedef struct {
            fcl_inference_memory_io(inputs, outputs);
        } fcl_inference_memory_t;
        static_assertion(fcl_inference_memory_size(inputs, layers, hidden, outputs) ==
                  sizeof(fcl_inference_memory_t) + sizeof(fcl_t));
        typedef struct {
            fcl_training_memory_io(inputs, outputs);
        } fcl_training_memory_io_t;
        static_assertion(fcl_training_memory_size(inputs, layers, hidden, outputs) ==
                  sizeof(fcl_training_memory_io_t) + sizeof(fcl_t));
    }
}

static void fcl_init(fcl_t* nn, int64_t bytes, uint64_t seed,
        int64_t inputs, int64_t layers, int64_t hidden, int64_t outputs) {
    fcl_check_sizes_and_structs_correctness();
    assertion(inputs >= 1);
    assertion(layers >= 0);
    assertion(layers > 0 && hidden > 0 || layers == 0 && hidden == 0);
    assertion(outputs >= 1);
    const int64_t training_size = fcl_training_memory_size(inputs, layers, hidden, outputs);
    const int64_t inference_size = fcl_inference_memory_size(inputs, layers, hidden, outputs);
    uint8_t* p = (uint8_t*)nn;
    memset(p, 0, sizeof(fcl_t)); // only zero out the header
    nn->inputs = inputs;
    nn->layers = layers;
    nn->hidden = hidden;
    nn->outputs = outputs;
    // iw, hw can be huge and deliberately left uninitialized
    nn->iw = (fcl_type*)(p + sizeof(fcl_t));
    nn->hw = nn->iw + fcl_iw_count(inputs, hidden);
    nn->ow = nn->hw + fcl_hw_count(layers, hidden);
    nn->output = nn->ow + fcl_ow_count(inputs, hidden, outputs);
    uint8_t* b = p + sizeof(fcl_t);  // beginning of counters memory
    uint8_t* e = p + bytes; // end of memory
//  println("total counters=%lld", (fcl_type*)e - (fcl_type*)b);
//  println("");
//  println("fcl_iw_count(inputs:%lld hidden:%lld)=%lld", inputs, hidden, fcl_iw_count(inputs, hidden));
//  println("fcl_hw_count(layers:%lld hidden:%lld)=%lld", layers, hidden, fcl_hw_count(layers, hidden));
//  println("fcl_ow_count(hidden:%lld outputs:%lld)=%lld", hidden, outputs, fcl_ow_count(hidden, outputs));
//  println("fcl_weights_count()=%lld", fcl_weights_count(inputs, layers, hidden, outputs));
//  println("");
//  println("fcl_output_count()=%lld", outputs);
    nn->state = nn->output + outputs; // .output[outputs]
//  println("iw - memory %lld (bytes) sizeof(fcl_t)=%lld", (uint8_t*)nn->iw - p, sizeof(fcl_t));
//  println("hw - iw %lld", nn->hw - nn->iw);
//  println("ow - hw %lld", nn->ow - nn->hw);
//  println("output - ow %lld", nn->output - nn->ow);
//  println("state - output %lld", nn->state - nn->output);
    assertion((uint8_t*)nn->iw - p == sizeof(fcl_t));
    assertion(p + sizeof(fcl_t) == (uint8_t*)nn->iw);
    if (bytes == inference_size) {
//      println("fcl_inference_state_count()=%lld", fcl_inference_state_count(hidden));
        nn->delta = null;
        nn->od = null;
        assertion((uint8_t*)(nn->state + fcl_inference_state_count(hidden)) == e);
    } else if (bytes == training_size) {
        nn->delta = nn->state + fcl_training_state_count(layers, hidden);
        nn->od = nn->delta + fcl_training_delta_count(layers, hidden);
//      println("fcl_training_state_count(layers, hidden): %d", fcl_training_state_count(layers, hidden));
//      println("fcl_training_delta_count(layers, hidden): %d", fcl_training_delta_count(layers, hidden));
//      println("end - nn->state %lld bytes %lld counts",
//           e - (uint8_t*)(nn->od + outputs),
//          (e - (uint8_t*)(nn->od + outputs)) / sizeof(fcl_type));
//      println("delta - state %lld", nn->delta - nn->state);
//      println("od - delta %lld", nn->od - nn->delta);
//      println("end - od %lld (bytes) %lld count", e - (uint8_t*)nn->od, (fcl_type*)e - nn->od);
        assertion((uint8_t*)(nn->od + outputs) == e);
    } else {
        assertion(false, "use fcl_infrence|training_memory_size()");
    }
    nn->seed = seed;
    nn->activation_hidden = null;
    nn->derivative_hidden = null;
    nn->activation_output = null;
    nn->derivative_output = null;
    if (nn->layers == 0) {
        nn->iw = null;
        nn->hw = null;
    }
    (void)b; (void)e;
}

fcl_type* fcl_inference(fcl_t* nn, const fcl_type* inputs);

static fcl_type fcl_sigmoid(fcl_type x) {
//  assertion(!isnan(x));
//  if (x < -45.0) { // ??? 2.8625186e-20
//      return 0;
//  } else if (x > 45.0) { // ??? 3.4934271e+19
//      return 1;
//  }
    fcl_type r = 1.0 / (1.0 + exp(-x));
    assertion(!isnan(r));
    return r;
}

// TODO: make static
/* static */ fcl_type fcl_derivative_sigmoid(fcl_type s) {
    return s * (1 - s);
}

/* static */ fcl_type fcl_relu(fcl_type x) {
    return x > 0 ? x : 0;
}

/* static */ fcl_type fcl_derivative_relu(fcl_type x) {
    return x > 0 ? 1 : 0;
}

/* static */ fcl_type fcl_tanh(fcl_type x) {
    return tanh(x);
}

/* static */ fcl_type fcl_derivative_tanh(fcl_type x) {
    return 1.0 - x * x;
}

/* static */ fcl_type fcl_linear(fcl_type x) {
    return x;
}

/* static */ fcl_type fcl_derivative_linear(fcl_type x) {
    return 1; (void)x; // unused
}

/* static */ fcl_type* fcl_inference(fcl_t* nn, const fcl_type* input) {
    nn->derivative_hidden = fcl_derivative_of(nn->activation_hidden);
    nn->derivative_output = fcl_derivative_of(nn->activation_output);
    // TODO: delete PARANOIDAL asserts
    assertion(fcl_inference_state_count(nn->hidden) == 2 * nn->hidden);
    fcl_type* i = (fcl_type*)input; // because i/o swap below
    fcl_type* s[2] = { nn->state, nn->state + nn->hidden };
    fcl_type* o = s[0];
    if (nn->layers > 0) {
        /* input -> hidden layer */
        const fcl_type* w = nn->iw;
        for (int64_t j = 0; j < nn->hidden; j++) {
//          println("w[0] %.16e", w[0]);
//          for (int64_t k = 0; k < nn->inputs; ++k) {
//              println("w[%lld] %.16e i[%lld]: %.16e", k + 1, w[k + 1], k, i[k]);
//          }
            *o++ = nn->activation_hidden(fcl_w_x_i(w, i, nn->inputs));
//          println("o[0][%lld]: %.16e", j, *(o - 1));
            w += nn->inputs + 1;
        }
        int ix = 0; // state index for inference only networks
        i = s[0]; // "o" already incremented above
        w = nn->layers > 0 ? nn->hw : nn->iw;
        for (int64_t h = 0; h < nn->layers - 1; h++) {
            for (int64_t j = 0; j < nn->hidden; j++) {
                *o++ = nn->activation_hidden(fcl_w_x_i(w, i, nn->hidden));
//              println("o[%lld][%lld]: %.16e", h, j, *(o - 1));
                w += nn->hidden + 1;
            }
            if (nn->delta == null) { // inference only network
                ix = !ix;
                i = s[ix];
                o = s[!ix];
            } else {
                i += nn->hidden; // "o" already incremented above
            }
        }
        if (nn->delta != null) { // training network
            assertion(o == nn->state + fcl_training_state_count(nn->layers, nn->hidden));
        }
    }
    // "n" number of output connections from "input" vector
    // if number of hidden layers is zero input is connected to output:
    const int64_t n = nn->layers > 0 ? nn->hidden : nn->inputs;
    const fcl_type* w = nn->ow;
    o = nn->output;
    for (int64_t j = 0; j < nn->outputs; j++) {
//      println("ouput[%d] w[0]: %.16e", j, *w);
//      for (int64_t k = 0; k < n; ++k) {
//          println("ouput[%lld] w[%lld]: %.16e i[%lld]: %.16e", j, k + 1, w[k + 1], k, i[k]);
//      }
        *o++ = nn->activation_output(fcl_w_x_i(w, i, n));
//      println("ouput[%lld]: %.16e", j, *(o - 1));
        w += n + 1;
    }
    // TODO: delete this PARANOIDAL assert
    assertion((nn->layers > 0 ? w - nn->iw : w - nn->ow) ==
        fcl_weights_count(nn->inputs, nn->layers, nn->hidden, nn->outputs));
    return nn->output;
}

/* static */ void fcl_train(fcl_t* nn, fcl_type const* inputs,
        const fcl_type* truth, fcl_type learning_rate) {
    fcl_type const* output = fcl.inference(nn, inputs);
    {   // calculate the output layer deltas.
        fcl_type const *o = output; /* First output. */
        fcl_type *d = nn->od; // output delta
        fcl_type const* t = truth; // pointer to the first grownd truth value
        for (int64_t j = 0; j < nn->outputs; j++) {
            *d++ = (*t - *o) * nn->derivative_output(*o);
//          println("o[%lld]: %.17e od[%lld]: %.17e", j, *o, j, *(d - 1));
            o++; t++;
        }
    }
    assertion(nn->delta + nn->layers * nn->hidden == nn->od);
    // hidden layer deltas, start at the last layer and work upward.
    const int64_t hh1 = (nn->hidden + 1) * nn->hidden;
    fcl_type* ww = nn->ow;
    assertion(nn->output + nn->outputs == nn->state);
    for (int64_t h = nn->layers - 1; h >= 0; h--) {
        if (h != nn->layers) {
            assertion(ww == nn->hw + hh1 * h);
        }
        /* Find first output and delta in this layer. */
        fcl_type *o = nn->state + nn->hidden * h;
        fcl_type *d = nn->delta + nn->hidden * h;
        /* Find first delta in following layer (which may be .delta[] or .od[]). */
        fcl_type const * const dd = (h == nn->layers - 1) ?
            nn->od : nn->delta + (h + 1) * nn->hidden;
        if (h == nn->layers - 1) {
            assertion(dd == nn->od);
        }
        for (int64_t j = 0; j < nn->hidden; j++) {
            fcl_type delta = 0;
            const int64_t n = h == nn->layers - 1 ? nn->outputs : nn->hidden;
            for (int64_t k = 0; k < n; k++) {
                const fcl_type forward_delta = dd[k];
//              println("dd[%lld]: %25.17e", k, dd[k]);
                const int64_t windex = k * (nn->hidden + 1) + (j + 1);
                const fcl_type forward_weight = ww[windex];
                delta += forward_delta * forward_weight;
//              println("delta: %25.17e := forward_delta: %25.17e "
//                      "forward_weight: %25.17e\n", delta, forward_delta,
//                      forward_weight);
            }
            *d = nn->derivative_hidden(*o) * delta;
//          println("d[%lld]: %25.17e o: %25.17e delta: %25.17e", j, *d, *o, delta);
            d++; o++;
        }
        ww = nn->hw + (h - 1) * hh1;
//      println("nn->hw - ww %lld", nn->hw - ww);
    }
    {   // Train the outputs.
        fcl_type const *d = nn->od; /* output delta. */
        /* Find first weight to first output delta. */
        fcl_type *w = nn->ow;
        /* Find first output in previous layer. */
        fcl_type const * const i =
            nn->layers == 0 ? inputs :
                              nn->state + nn->hidden * (nn->layers - 1);
        const int64_t n = nn->layers == 0 ? nn->inputs : nn->hidden;
        /* Set output layer weights. */
        for (int64_t j = 0; j < nn->outputs; j++) {
            *w++ += *d * learning_rate * -1.0; // bias
            for (int64_t k = 1; k < n + 1; k++) {
                *w++ += *d * learning_rate * i[k - 1];
//              println("output[%lld] i[%lld] %25.17e w %25.17e",
//                  j, k - 1, i[k - 1], *(w - 1));
            }
            ++d;
        }
        assertion((nn->layers > 0 ? w - nn->iw : w - nn->ow) ==
            fcl_weights_count(nn->inputs, nn->layers, nn->hidden, nn->outputs));
    }
    /* Train the hidden layers. */
    for (int64_t h = nn->layers - 1; h >= 0; h--) {
        fcl_type const* d = nn->delta + h * nn->hidden;
        fcl_type const* i = h == 0 ? inputs : nn->state + (h - 1) * nn->hidden;
        fcl_type *w = h == 0 ? nn->iw : nn->hw + hh1 * (h - 1);
        for (int64_t j = 0; j < nn->hidden; j++) {
//          println("hidden layer[%lld][%lld] weights w.ofs=%lld\n", h, j, w - nn->iw);
            *w++ += *d * learning_rate * -1.0; // bias
//          println("w[0] (bias)=%25.17e d=%25.17e", *(w - 1), *d);
            const int64_t n = (h == 0 ? nn->inputs : nn->hidden) + 1;
            for (int64_t k = 1; k < n; k++) {
                *w++ += *d * learning_rate * i[k - 1];
//              println("i[%lld] %25.17e w %25.17e ", k - 1, i[k - 1], *(w - 1));
            }
            ++d;
        }
    }
}


fcl_if fcl = {
    .init = fcl_init,
    .inference = fcl_inference,
    .train = fcl_train,
    .sigmoid = fcl_sigmoid,
    .relu = fcl_relu,
    .tanh = fcl_tanh,
    .linear = fcl_linear
};

#endif // fcl_IMPLEMENTATION


#ifdef __cplusplus
}
#endif

