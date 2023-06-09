#ifndef FFFC_H
#define FFFC_H
#include <stdint.h>
#include <rt.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef fffc_type
#define fffc_type double // TODO: change to float when done with genann
#endif

typedef fffc_type (*fffc_activation_t)(fffc_type a);

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

typedef /* begin_packed */ struct fffc_s {
    int64_t inputs;  // number elements in input vector
    int64_t layers;  // number of hidden layers, must be >= 1
    int64_t hidden;  // number of neurons in each hidden layer
    int64_t outputs; // number elements in output vector
    // iw and hw are located contiguously in memory and can be written out
    // as chunk of binary data
    // for each hidden neuron of first layer bias and weight for each input element
    fffc_type* iw; // [hidden][inputs + 1] == null if layers == 0 && hidden == 0
    // for each neuron of each hidden layer bias and weights vector of previous layer weights:
    fffc_type* hw;  // [layers][hidden][hidden + 1] or null for layers == 0
    fffc_type* ow;  // [outputs][hidden + 1] output weights
    fffc_type* output; // [outputs] elements of output vector
    // training only data:
    fffc_type* state; // [2][hidden] for inference() | [layers][hidden] for train()
    // delta == null for inference only fffc
    fffc_type* delta; // null inference() | [layers][hidden] for train()
    // output deltas:
    fffc_type* od;    // null inference() | [outputs] for train()
    // initial seed for random64() generator used for this network
    uint64_t seed;
    // for known fffc.sigmoid, tanh, relu, linear activation functions
    // nn.train() knows the derivatives
    fffc_activation_t activation_hidden; // activation function to use for hidden neurons
    fffc_activation_t derivative_hidden; // derivative of hidden activation function
    fffc_activation_t activation_output; // activation function to use for output elements
    fffc_activation_t derivative_output; // derivative of output activation function
} /* end_packed */ fffc_t;

typedef struct fffc_if { // interface
    void (*init)(fffc_t* nn, int64_t bytes, uint64_t seed,
                 int64_t inputs, int64_t layers, int64_t hidden, int64_t outputs);
    fffc_type (*random)(fffc_t* nn, fffc_type lo, fffc_type hi); // next random after seed
    // randomize initializes each wieight and bias [lo..hi] starting with seed
    void (*randomize)(fffc_t* nn, uint64_t seed, fffc_type lo, fffc_type hi);
    // inference() returns pointer to output array of elements
    fffc_type* (*inference)(fffc_t* nn, const fffc_type* input);
    void (*train)(fffc_t* nn, const fffc_type* input, const fffc_type* truth,
                  fffc_type learning_rate);
    // available activation functions:
    const fffc_activation_t sigmoid;
    const fffc_activation_t tanh;
    const fffc_activation_t relu;
    const fffc_activation_t linear;
} fffc_if;

extern fffc_if fffc;

// TODO: delete from here and make static
fffc_type fffc_derivative_sigmoid(fffc_type s);
fffc_type fffc_derivative_relu(fffc_type x);
fffc_type fffc_derivative_tanh(fffc_type x);
fffc_type fffc_derivative_linear(fffc_type x);

// Memory management helpers.
// All counts are a number of fffc_type elements not bytes

#define fffc_iw_count(inputs, hidden) ((hidden) == 0 ? \
    0 : ((hidden) * ((inputs) + 1)))
#define fffc_hw_count(layers, hidden) ((layers) == 0 ? \
    0 : (((layers) - 1)  * (hidden) * ((hidden) + 1)))
#define fffc_ow_count(inputs, hidden, outputs) ((hidden) == 0 ? \
    (outputs) * ((inputs) + 1) : ((outputs) * ((hidden) + 1)))

#define fffc_weights_count(inputs, layers, hidden, outputs) (             \
   (layers > 0 ?                                                          \
   (fffc_iw_count(inputs, hidden) + fffc_hw_count(layers, hidden)) : 0) + \
    fffc_ow_count(inputs, hidden, outputs)                  )

#define fffc_inference_state_count(hidden) (2 * (hidden))
#define fffc_training_state_count(layers, hidden) ((layers) * (hidden))
#define fffc_training_delta_count(layers, hidden) ((layers) * (hidden))

// input, [layers][hidden][hidden+1], output
#define fffc_weights_memory_iho(inputs, layers, hidden, outputs) \
    fffc_type iw[hidden][inputs + 1];                        \
    fffc_type hw[layers - 1][hidden][hidden + 1];            \
    fffc_type ow[outputs][hidden + 1]

// input, output
#define fffc_weights_memory_io(inputs, outputs) \
    fffc_type ow[outputs][inputs + 1]

#define fffc_weight_memory_size(inputs, layers, hidden, outputs) \
    ((fffc_iw_count(inputs, hidden) +                            \
      fffc_hw_count(layers, hidden) +                            \
      fffc_ow_count(inputs, hidden, outputs)) * sizeof(fffc_type))

#define fffc_base_memory_iho(inputs, layers, hidden, outputs) \
    fffc_weights_memory_iho(inputs, layers, hidden, outputs); \
    fffc_type output[outputs]

#define fffc_base_memory_io(inputs, outputs)  \
    fffc_weights_memory_io(inputs,  outputs); \
    fffc_type output[outputs]

#define fffc_base_memory_size(inputs, layers, hidden, outputs) ( \
    fffc_weight_memory_size(inputs, layers, hidden, outputs) +   \
    /* output: */(outputs) * sizeof(fffc_type)                   \
)

#define fffc_inference_memory_iho(inputs, layers, hidden, outputs) \
    fffc_base_memory_iho(inputs, layers, hidden, outputs);         \
    fffc_type state[2][hidden]

#define fffc_inference_memory_io(inputs, outputs) \
    fffc_base_memory_io(inputs, outputs)

#define fffc_inference_memory_size(inputs, layers, hidden, outputs) ( \
    fffc_base_memory_size(inputs, layers, hidden, outputs) +          \
    fffc_inference_state_count(hidden) * sizeof(fffc_type) +          \
    sizeof(fffc_t)                                                    \
)

#define fffc_training_memory_iho(inputs, layers, hidden, outputs) \
    fffc_base_memory_iho(inputs, layers, hidden, outputs);        \
    fffc_type state[layers][hidden];                              \
    fffc_type delta[layers][hidden];                              \
    fffc_type od[outputs]
//  ^^^ memory layout is important "od" must be imediately after "delta"

#define fffc_training_memory_io(inputs, outputs)    \
    fffc_base_memory_io(inputs, outputs);           \
    fffc_type od[outputs]
//  ^^^ memory layout is important "od" must be imediately after "ow"

#define fffc_training_memory_size(inputs, layers, hidden, outputs) ( \
    fffc_base_memory_size(inputs, layers, hidden, outputs) +         \
   (fffc_training_state_count(layers, hidden) +                      \
    fffc_training_delta_count(layers, hidden) +                      \
    /* od: */ (outputs)) * sizeof(fffc_type) +                       \
    sizeof(fffc_t)                                                   \
)

#define fffc_io_memory_size(inputs, layers, hidden, outputs) (        \
    fffc_base_memory_size(inputs, layers, hidden, outputs) +          \
    fffc_inference_state_count(hidden) * sizeof(fffc_type) +          \
    (uint8_t*)&((fffc_t*)(0))->iw - (uint8_t*)((fffc_t*)(0))          \
)

/*
    How to use:
    enum { inputs = 2, layers = 1, hidden = 2, outputs = 1};
    int64_t bytes = fffc_training_memory_size(inputs, layers, hidden, outputs);
    fffc_t* nn = (fffc_t*)malloc(bytes);
    if (nn != null) {
        uint64_t seed = 1;
        fffc.init(nn, bytes, seed, inputs, layers, hidden, outputs);
        nn->activation_hidden = fffc.sigmoid;
        fffc_type inputs[4][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
        fffc_type xor[4] = { 0, 1, 1, 0 };
        fffc_type learning_rate = 3; // no idea why 3?
        for (int32_t epoch = 0; epoch < 500; epoch++) {
            for (int32_t i = 0; i < countof(inputs); i++) {
                fffc.train(nn, inputs[i], &xor[i], learning_rate);
            }
        }
        for (int32_t i = 0; i < countof(inputs); i++) {
            fffc_type* output = nn->inference(nn, input);
            printf("%.1f ^ %.1f = %.1f\n", inputs[i][0], inputs[i][1], *output);
        }
        free(nn);
    }


*/


#endif // FFFC_H

#ifdef FFFC_IMPLEMENTATION

static inline void fffc_check_sizes_and_structs_correctness() {
    // C99 does not have constexpr thus just generic check for a match
    {
        enum { inputs = 123, layers = 234, hidden = 456, outputs = 567 };
        typedef struct {
            fffc_weights_memory_iho(inputs, layers, hidden, outputs);
        } fffc_weights_memory_iho_t;
        static_assertion(fffc_weight_memory_size(inputs, layers, hidden, outputs) ==
            sizeof(fffc_weights_memory_iho_t));
        typedef struct {
            fffc_inference_memory_iho(inputs, layers, hidden, outputs);
        } fffc_inference_memory_t;
        static_assertion(fffc_inference_memory_size(inputs, layers, hidden, outputs) ==
                  sizeof(fffc_inference_memory_t) + sizeof(fffc_t));
        typedef struct {
            fffc_training_memory_iho(inputs, layers, hidden, outputs);
        } fffc_training_memory_t;
        static_assertion(fffc_training_memory_size(inputs, layers, hidden, outputs) ==
                  sizeof(fffc_training_memory_t) + sizeof(fffc_t));
    }
    // Microsoft implementation of C99 and C17 does not allow zero size arrays
    {
        enum { inputs = 123, layers = 0, hidden = 0, outputs = 567 };
        typedef struct {
            fffc_weights_memory_io(inputs, outputs);
        } fffc_weights_memory_io_t;
        static_assertion(fffc_weight_memory_size(inputs, layers, hidden, outputs) ==
            sizeof(fffc_weights_memory_io_t));
        typedef struct {
            fffc_inference_memory_io(inputs, outputs);
        } fffc_inference_memory_t;
        static_assertion(fffc_inference_memory_size(inputs, layers, hidden, outputs) ==
                  sizeof(fffc_inference_memory_t) + sizeof(fffc_t));
        typedef struct {
            fffc_training_memory_io(inputs, outputs);
        } fffc_training_memory_io_t;
        static_assertion(fffc_training_memory_size(inputs, layers, hidden, outputs) ==
                  sizeof(fffc_training_memory_io_t) + sizeof(fffc_t));
    }
}

static void fffc_init(fffc_t* nn, int64_t bytes, uint64_t seed,
        int64_t inputs, int64_t layers, int64_t hidden, int64_t outputs) {
    fffc_check_sizes_and_structs_correctness();
    assertion(inputs >= 1);
    assertion(layers >= 0);
    assertion(layers > 0 && hidden > 0 || layers == 0 && hidden == 0);
    assertion(outputs >= 1);
    const int64_t training_size = fffc_training_memory_size(inputs, layers, hidden, outputs);
    const int64_t inference_size = fffc_inference_memory_size(inputs, layers, hidden, outputs);
    uint8_t* p = (uint8_t*)nn;
    memset(p, 0, sizeof(fffc_t)); // only zero out the header
    nn->inputs = inputs;
    nn->layers = layers;
    nn->hidden = hidden;
    nn->outputs = outputs;
    // iw, hw can be huge and deliberately left uninitialized
    nn->iw = (fffc_type*)(p + sizeof(fffc_t));
    nn->hw = nn->iw + fffc_iw_count(inputs, hidden);
    nn->ow = nn->hw + fffc_hw_count(layers, hidden);
    nn->output = nn->ow + fffc_ow_count(inputs, hidden, outputs);
    uint8_t* b = p + sizeof(fffc_t);  // beginning of counters memory
    uint8_t* e = p + bytes; // end of memory
//  traceln("total counters=%lld", (fffc_type*)e - (fffc_type*)b);
//  traceln("");
//  traceln("fffc_iw_count(inputs:%lld hidden:%lld)=%lld", inputs, hidden, fffc_iw_count(inputs, hidden));
//  traceln("fffc_hw_count(layers:%lld hidden:%lld)=%lld", layers, hidden, fffc_hw_count(layers, hidden));
//  traceln("fffc_ow_count(hidden:%lld outputs:%lld)=%lld", hidden, outputs, fffc_ow_count(hidden, outputs));
//  traceln("fffc_weights_count()=%lld", fffc_weights_count(inputs, layers, hidden, outputs));
//  traceln("");
//  traceln("fffc_output_count()=%lld", outputs);
    nn->state = nn->output + outputs; // .output[outputs]
//  traceln("iw - memory %lld (bytes) sizeof(fffc_t)=%lld", (uint8_t*)nn->iw - p, sizeof(fffc_t));
//  traceln("hw - iw %lld", nn->hw - nn->iw);
//  traceln("ow - hw %lld", nn->ow - nn->hw);
//  traceln("output - ow %lld", nn->output - nn->ow);
//  traceln("state - output %lld", nn->state - nn->output);
    assertion((uint8_t*)nn->iw - p == sizeof(fffc_t));
    assertion(p + sizeof(fffc_t) == (uint8_t*)nn->iw);
    if (bytes == inference_size) {
//      traceln("fffc_inference_state_count()=%lld", fffc_inference_state_count(hidden));
        nn->delta = null;
        nn->od = null;
        assertion((uint8_t*)(nn->state + fffc_inference_state_count(hidden)) == e);
    } else if (bytes == training_size) {
        nn->delta = nn->state + fffc_training_state_count(layers, hidden);
        nn->od = nn->delta + fffc_training_delta_count(layers, hidden);
//      traceln("fffc_training_state_count(layers, hidden): %d", fffc_training_state_count(layers, hidden));
//      traceln("fffc_training_delta_count(layers, hidden): %d", fffc_training_delta_count(layers, hidden));
//      traceln("end - nn->state %lld bytes %lld counts",
//           e - (uint8_t*)(nn->od + outputs),
//          (e - (uint8_t*)(nn->od + outputs)) / sizeof(fffc_type));
//      traceln("delta - state %lld", nn->delta - nn->state);
//      traceln("od - delta %lld", nn->od - nn->delta);
//      traceln("end - od %lld (bytes) %lld count", e - (uint8_t*)nn->od, (fffc_type*)e - nn->od);
        assertion((uint8_t*)(nn->od + outputs) == e);
    } else {
        assertion(false, "use fffc_infrence|training_memory_size()");
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

fffc_type* fffc_inference(fffc_t* nn, const fffc_type* inputs);

static fffc_type fffc_sigmoid(fffc_type x) {
//  assertion(!isnan(x));
//  if (x < -45.0) { // ??? 2.8625186e-20
//      return 0;
//  } else if (x > 45.0) { // ??? 3.4934271e+19
//      return 1;
//  }
    fffc_type r = 1.0 / (1.0 + exp(-x));
    assertion(!isnan(r));
    return r;
}

// TODO: make static
/* static */ fffc_type fffc_derivative_sigmoid(fffc_type s) {
    return s * (1 - s);
}

/* static */ fffc_type fffc_relu(fffc_type x) {
    return x > 0 ? x : 0;
}

/* static */ fffc_type fffc_derivative_relu(fffc_type x) {
    return x > 0 ? 1 : 0;
}

/* static */ fffc_type fffc_tanh(fffc_type x) {
    return tanh(x);
}

/* static */ fffc_type fffc_derivative_tanh(fffc_type x) {
    return 1.0 - x * x;
}

/* static */ fffc_type fffc_linear(fffc_type x) {
    return x;
}

/* static */ fffc_type fffc_derivative_linear(fffc_type x) {
    return 1; (void)x; // unused
}

/* static */ fffc_type* fffc_inference(fffc_t* nn, const fffc_type* input) {
    nn->derivative_hidden = fffc_derivative_of(nn->activation_hidden);
    nn->derivative_output = fffc_derivative_of(nn->activation_output);
    // TODO: delete PARANOIDAL asserts
    assertion(fffc_inference_state_count(nn->hidden) == 2 * nn->hidden);
    fffc_type* i = (fffc_type*)input; // because i/o swap below
    fffc_type* s[2] = { nn->state, nn->state + nn->hidden };
    fffc_type* o = s[0];
    if (nn->layers > 0) {
        /* input -> hidden layer */
        const fffc_type* w = nn->iw;
        for (int64_t j = 0; j < nn->hidden; j++) {
//          traceln("w[0] %.16e", w[0]);
//          for (int64_t k = 0; k < nn->inputs; ++k) {
//              traceln("w[%lld] %.16e i[%lld]: %.16e", k + 1, w[k + 1], k, i[k]);
//          }
            *o++ = nn->activation_hidden(fffc_w_x_i(w, i, nn->inputs));
//          traceln("o[0][%lld]: %.16e", j, *(o - 1));
            w += nn->inputs + 1;
        }
        int ix = 0; // state index for inference only networks
        i = s[0]; // "o" already incremented above
        w = nn->layers > 0 ? nn->hw : nn->iw;
        for (int64_t h = 0; h < nn->layers - 1; h++) {
            for (int64_t j = 0; j < nn->hidden; j++) {
                *o++ = nn->activation_hidden(fffc_w_x_i(w, i, nn->hidden));
//              traceln("o[%lld][%lld]: %.16e", h, j, *(o - 1));
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
            assertion(o == nn->state + fffc_training_state_count(nn->layers, nn->hidden));
        }
    }
    // "n" number of output connections from "input" vector
    // if number of hidden layers is zero input is connected to output:
    const int64_t n = nn->layers > 0 ? nn->hidden : nn->inputs;
    const fffc_type* w = nn->ow;
    o = nn->output;
    for (int64_t j = 0; j < nn->outputs; j++) {
//      traceln("ouput[%d] w[0]: %.16e", j, *w);
//      for (int64_t k = 0; k < n; ++k) {
//          traceln("ouput[%lld] w[%lld]: %.16e i[%lld]: %.16e", j, k + 1, w[k + 1], k, i[k]);
//      }
        *o++ = nn->activation_output(fffc_w_x_i(w, i, n));
//      traceln("ouput[%lld]: %.16e", j, *(o - 1));
        w += n + 1;
    }
    // TODO: delete this PARANOIDAL assert
    assertion((nn->layers > 0 ? w - nn->iw : w - nn->ow) ==
        fffc_weights_count(nn->inputs, nn->layers, nn->hidden, nn->outputs));
    return nn->output;
}

/* static */ void fffc_train(fffc_t* nn, fffc_type const* inputs,
        const fffc_type* truth, fffc_type learning_rate) {
    fffc_type const* output = fffc.inference(nn, inputs);
    {   // calculate the output layer deltas.
        fffc_type const *o = output; /* First output. */
        fffc_type *d = nn->od; // output delta
        fffc_type const* t = truth; // pointer to the first grownd truth value
        for (int64_t j = 0; j < nn->outputs; j++) {
            *d++ = (*t - *o) * nn->derivative_output(*o);
//          traceln("o[%lld]: %.17e od[%lld]: %.17e", j, *o, j, *(d - 1));
            o++; t++;
        }
    }
    assertion(nn->delta + nn->layers * nn->hidden == nn->od);
    // hidden layer deltas, start at the last layer and work upward.
    const int64_t hh1 = (nn->hidden + 1) * nn->hidden;
    fffc_type* ww = nn->ow;
    assertion(nn->output + nn->outputs == nn->state);
    for (int64_t h = nn->layers - 1; h >= 0; h--) {
        if (h != nn->layers) {
            assertion(ww == nn->hw + hh1 * h);
        }
        /* Find first output and delta in this layer. */
        fffc_type *o = nn->state + nn->hidden * h;
        fffc_type *d = nn->delta + nn->hidden * h;
        /* Find first delta in following layer (which may be .delta[] or .od[]). */
        fffc_type const * const dd = (h == nn->layers - 1) ?
            nn->od : nn->delta + (h + 1) * nn->hidden;
        if (h == nn->layers - 1) {
            assertion(dd == nn->od);
        }
        for (int64_t j = 0; j < nn->hidden; j++) {
            fffc_type delta = 0;
            const int64_t n = h == nn->layers - 1 ? nn->outputs : nn->hidden;
            for (int64_t k = 0; k < n; k++) {
                const fffc_type forward_delta = dd[k];
//              traceln("dd[%lld]: %25.17e", k, dd[k]);
                const int64_t windex = k * (nn->hidden + 1) + (j + 1);
                const fffc_type forward_weight = ww[windex];
                delta += forward_delta * forward_weight;
//              traceln("delta: %25.17e := forward_delta: %25.17e "
//                      "forward_weight: %25.17e\n", delta, forward_delta,
//                      forward_weight);
            }
            *d = nn->derivative_hidden(*o) * delta;
//          traceln("d[%lld]: %25.17e o: %25.17e delta: %25.17e", j, *d, *o, delta);
            d++; o++;
        }
        ww = nn->hw + (h - 1) * hh1;
//      traceln("nn->hw - ww %lld", nn->hw - ww);
    }
    {   // Train the outputs.
        fffc_type const *d = nn->od; /* output delta. */
        /* Find first weight to first output delta. */
        fffc_type *w = nn->ow;
        /* Find first output in previous layer. */
        fffc_type const * const i =
            nn->layers == 0 ? inputs :
                              nn->state + nn->hidden * (nn->layers - 1);
        const int64_t n = nn->layers == 0 ? nn->inputs : nn->hidden;
        /* Set output layer weights. */
        for (int64_t j = 0; j < nn->outputs; j++) {
            *w++ += *d * learning_rate * -1.0; // bias
            for (int64_t k = 1; k < n + 1; k++) {
                *w++ += *d * learning_rate * i[k - 1];
//              traceln("output[%lld] i[%lld] %25.17e w %25.17e",
//                  j, k - 1, i[k - 1], *(w - 1));
            }
            ++d;
        }
        assertion((nn->layers > 0 ? w - nn->iw : w - nn->ow) ==
            fffc_weights_count(nn->inputs, nn->layers, nn->hidden, nn->outputs));
    }
    /* Train the hidden layers. */
    for (int64_t h = nn->layers - 1; h >= 0; h--) {
        fffc_type const* d = nn->delta + h * nn->hidden;
        fffc_type const* i = h == 0 ? inputs : nn->state + (h - 1) * nn->hidden;
        fffc_type *w = h == 0 ? nn->iw : nn->hw + hh1 * (h - 1);
        for (int64_t j = 0; j < nn->hidden; j++) {
//          traceln("hidden layer[%lld][%lld] weights w.ofs=%lld\n", h, j, w - nn->iw);
            *w++ += *d * learning_rate * -1.0; // bias
//          traceln("w[0] (bias)=%25.17e d=%25.17e", *(w - 1), *d);
            const int64_t n = (h == 0 ? nn->inputs : nn->hidden) + 1;
            for (int64_t k = 1; k < n; k++) {
                *w++ += *d * learning_rate * i[k - 1];
//              traceln("i[%lld] %25.17e w %25.17e ", k - 1, i[k - 1], *(w - 1));
            }
            ++d;
        }
    }
}


fffc_if fffc = {
    .init = fffc_init,
    .inference = fffc_inference,
    .train = fffc_train,
    .sigmoid = fffc_sigmoid,
    .relu = fffc_relu,
    .tanh = fffc_tanh,
    .linear = fffc_linear
};

#endif // FFFC_IMPLEMENTATION


#ifdef __cplusplus
}
#endif

