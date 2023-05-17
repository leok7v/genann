#include "nnn.h"
#include "rt.h"
#include "dot_product.h"
#include <float.h>
#include <memory.h>
#include <stdbool.h>
#include <math.h>

static double random(nnn_t* net) {
    double r = random32(&net->seed) / (double)UINT32_MAX - 0.5;
    assertion(-1.0 <= r && r <= 1.0);
    return r;
}

static double relu(double x) {
    return x > 0 ? x : 0;
}

static double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

static double sigmoid(double x) {
    double r = 1.0 / (1.0 + exp(-x));
    assertion(!isnan(r));
    return r;
}

static double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

static double activation_tanh(double x) {
    return tanh(x);
}

static double tanh_derivative(double x) {
    double t = tanh(x);
    return 1.0 - t * t;
}

static void inference(nnn_t* net) {
    for (int i = 0; i < nnn_hidden; i++) {
        double sum = dot_product_f64(net->input, net->weights_ih[i], nnn_input);
        net->hidden[0][i] = net->activation(sum + net->biases_ih[i]);
    }
    for (int k = 0; k < nnn_layers; k++) {
        for (int i = 0; i < nnn_hidden; i++) {
            double sum = dot_product_f64(net->hidden[k], net->weights_hh[k][i], nnn_hidden);
            net->hidden[k + 1][i] = net->activation(sum + net->biases_hh[k][i]);
        }
    }
    const double* last = net->hidden[nnn_layers];
    for (int i = 0; i < nnn_output; i++) {
        double sum = dot_product_f64(last, net->weights_ho[i], nnn_hidden);
        net->output[i] = net->activation(sum + net->biases_ho[i]);
        assertion(!isnan(net->output[i]));
    }
}

// Loss function: Mean Square Error

static double mse(double truth, double predicted) {
    double error = truth - predicted;
    return sqrt(error * error);
}

// and it's derivative:

static double mse_derivative(double truth, double predicted) {
    return predicted - truth;
}

/*  Proof:
    The derivative of the mean squared error (MSE) with respect
    to the predicted output value can be derived using the
    chain rule of differentiation:
        Let E be derivative of the MSE function,
        then E = (1/2) * (truth - predicted)^2
    Taking derivative of E with regard to predicted, we get:
        dE/dpredicted = 2 * (1/2) * (truth - predicted) * (-1) =
        predicted - truth
*/

static double rma(nnn_t* net, double truth[nnn_output], double *max_rma) {
    inference(net);
    *max_rma = 0;
    double sum = 0;
    for (int j = 0; j < countof(net->output); j++) {
        double err = mse(truth[j], net->output[j]);
        *max_rma = max(*max_rma, err);
        sum += err;
    }
    return sum / countof(net->output);
}

static void add_small_random(nnn_t* net, double* v, double nudge) {
    if (nudge != 0) { *v += random(net) * nudge; }
}

#if 0
static void train(nnn_t* net, vector_output_t truth,
        double learning_rate, double weight_decay, double nudge) {
    inference(net);
    vector_hidden_t _dt0;
    vector_hidden_t _dt1;
    double* dt0 = _dt0;
    double* dt1 = _dt1;
    // Compute the error of the output layer
    for (int i = 0; i < nnn_output; i++) {
        double error = net->output[i] - truth[i];
        double d_act = net->derivative(net->output[i]);
        double delta = error * d_act;
        for (int j = 0; j < nnn_hidden; j++) {
            double d_act_j = net->derivative(net->hidden[nnn_layers][j]);
            dt0[j] = error * d_act_j;
            double grad = delta * net->hidden[nnn_layers][j];
            net->weights_ho[i][j] -= learning_rate * grad;
            net->biases_ho[i] -= learning_rate * delta;
        }
    }
    // Compute the errors of the hidden layers in reverse order
    for (int k = nnn_layers - 1; k >= 0; k--) {
        for (int i = 0; i < nnn_hidden; i++) {
            double error = 0;
            for (int j = 0; j < nnn_hidden; j++) {
                assertion(!isnan(net->weights_hh[k][i][j]));
                assertion(!isnan(dt0[j]));
                error += net->weights_hh[k][i][j] * dt0[j];
                assertion(!isnan(error));
            }
            double d_act = net->derivative(net->hidden[k][i]);
            double delta = error * d_act;
            dt1[i] = delta;
            if (k > 0) {
                for (int j = 0; j < nnn_hidden; j++) {
                    double grad = delta * net->hidden[k - 1][j];
                    net->weights_hh[k][i][j] -= learning_rate * grad;
                    net->biases_hh[k][i] -= learning_rate * delta;
                }
            } else {
                for (int j = 0; j < nnn_input; j++) {
                    double grad = delta * net->input[j];
                    net->weights_ih[i][j] -= learning_rate * grad;
                    net->biases_ih[i] -= learning_rate * delta;
                    assertion(!isnan(net->biases_ih[i]));
                }
            }
        }
        double* swap = dt0; dt0 = dt1; dt1 = swap;
    }
}

#else

static void train(nnn_t* net, vector_output_t truth,
        double learning_rate, double weight_decay, double nudge) {
    inference(net);
    vector_hidden_t _dt0;
    vector_hidden_t _dt1;
    double* dt0 = _dt0;
    double* dt1 = _dt1;
    // Backpropagate output layer
    const double* last = net->hidden[nnn_layers];
    for (int i = 0; i < nnn_output; i++) {
        double error = net->output[i] - truth[i];
        double delta = error * net->derivative(net->output[i]);
        net->biases_ho[i] -= learning_rate * delta;
        for (int j = 0; j < nnn_hidden; j++) {
            dt0[j] = error * net->derivative(net->hidden[nnn_layers][j]);
            const double weight_delta = delta * last[j];
            const double correction = learning_rate * weight_delta +
                                      weight_decay * net->weights_ho[i][j];
            net->weights_ho[i][j] -= correction;
//          add_small_random(net, &net->weights_ho[i][j], nudge);
        }
    }
    // Backpropagate hidden layers
    for (int k = nnn_layers; k > 0; k--) {
        for (int i = 0; i < nnn_hidden; i++) {
            double error = 0;
            for (int j = 0; j < nnn_hidden; j++) {
//              error += net->weights_hh[k - 1][i][j] * net->derivative(net->hidden[k - 1][j]);
                error += net->weights_hh[k - 1][i][j] * dt0[j];
            }
            double delta = error * net->derivative(net->hidden[k - 1][i]);
            dt1[i] = delta;
            net->biases_hh[k - 1][i] -= learning_rate * delta;
//          add_small_random(net, &net->biases_hh[k - 1][i], nudge);
            for (int j = 0; j < nnn_hidden; j++) {
                const double weight_delta = delta * net->hidden[k - 1][j];
                const double correction = learning_rate * weight_delta +
                                          weight_decay * net->weights_hh[k - 1][i][j];
                net->weights_hh[k - 1][i][j] -= correction;
//              add_small_random(net, &net->weights_hh[k - 1][i][j], nudge);
            }
        }
        double* swap = dt0; dt0 = dt1; dt1 = swap;
    }
    // Backpropagate input layer
    for (int i = 0; i < nnn_hidden; i++) {
        double error = 0;
        for (int j = 0; j < nnn_hidden; j++) {
//          error += net->weights_hh[0][i][j] * net->derivative(first[j]);
            error += net->weights_hh[0][i][j] * dt0[j];
        }
        double sum = dot_product_f64(net->input, net->weights_ih[i], nnn_input);
        double delta = error * net->derivative(sum + net->biases_ih[i]);
        net->biases_ih[i] -= learning_rate * delta;
//      add_small_random(net, &net->biases_ih[i], nudge);
        for (int j = 0; j < nnn_input; j++) {
            const double weight_delta = delta * net->input[j];
            const double correction = learning_rate * weight_delta +
                                      weight_decay * net->weights_ih[i][j];
            net->weights_ih[i][j] -= correction;
            add_small_random(net, &net->weights_ih[i][j], nudge);
        }
    }
}

#endif

static void init(nnn_t* net, int activation_function, uint32_t seed) {
    memset(net, 0, sizeof(*net));
    switch (activation_function) {
        case nnn_sigmoid:
            net->activation = sigmoid;
            net->derivative = sigmoid_derivative;
            break;
        case nnn_tanh:
            net->activation = activation_tanh;
            net->derivative = tanh_derivative;
            break;
        case nnn_relu:
            net->activation = relu;
            net->derivative = relu_derivative;
            break;
        default: assertion(false);
    }
    net->seed = seed;
    for (int i = 0; i < nnn_input; i++) {
        for (int j = 0; j < nnn_hidden; j++) {
            net->weights_ih[i][j] = random(net);
        }
    }
    for (int i = 0; i < nnn_hidden; i++) {
        net->biases_ih[i] = random(net);
    }
    for (int k = 0; k < nnn_layers; k++) {
        for (int i = 0; i < nnn_hidden; i++) {
            for (int j = 0; j < nnn_hidden; j++) {
                net->weights_hh[k][i][j] = random(net);
            }
            net->biases_hh[k][i] = random(net);
        }
    }
    for (int i = 0; i < nnn_output; i++) {
        for (int j = 0; j < nnn_hidden; j++) {
            net->weights_ho[i][j] = random(net);
        }
        net->biases_ho[i] = random(net);
    }
}

static void dump(nnn_t* net) {
    printf("INPUT:\n");
    printf("biases_ih[%d]:\n", (int)countof(net->biases_ih));
    for (int i = 0; i < countof(net->biases_ih); i++) {
        printf("%10.6f ", net->biases_ih[i]);
    }
    printf("\n");
    printf("weights_ih[%d]:\n", (int)countof(net->weights_ih));
    for (int i = 0; i < countof(net->weights_ih); i++) {
        for (int j = 0; j < countof(net->weights_ih[i]); j++) {
            printf("%10.6f ", net->weights_ih[i][j]);
        }
        printf("\n");
    }
    printf("HIDDEN:\n");
    for (int k = 0; k < countof(net->biases_hh); k++) {
        printf("layer:%d\n", k);
        printf("biases_hh: ");
        for (int i = 0; i < countof(net->biases_hh[k]); i++) {
            printf("%10.6f ", net->biases_hh[k][i]);
        }
        printf("\n");
        for (int i = 0; i < countof(net->biases_hh[k]); i++) {
            printf("net->weights_hh[%d][%d]: ", k, i);
            for (int j = 0; j < countof(net->weights_ih[i]); j++) {
                printf("%10.6f ", net->weights_hh[k][i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("OUTPUT:\n");
    printf("biases_ho[%d]: ", (int)countof(net->biases_ho));
    for (int i = 0; i < countof(net->biases_ho); i++) {
        printf("%10.6f ", net->biases_ho[i]);
    }
    printf("\n");
    printf("weights_ho[%d]:\n", (int)countof(net->weights_ho));
    for (int i = 0; i < countof(net->weights_ho); i++) {
        for (int j = 0; j < countof(net->weights_ho[i]); j++) {
            printf("%10.6f ", net->weights_ho[i][j]);
        }
        printf("\n");
    }
}

nnn_if nnn = {
    .init = init,
    .inference = inference,
    .train = train,
    .rma = rma,
    .dump = dump
};

/* // Loss function: Cross-Entropy - only for classification networks
static double cross_entropy(double truth, double predicted) {
    // L = -sum(y*log(p) + (1-y)*log(1-p))
    // Where:
    // y is the true label (0 or 1)
    // p is the predicted probability of the positive class
    //  (i.e., the class with label 1)
    // To adapt the cross_entropy function for a neural network, one would
    // need to modify the input parameters to represent the true label and
    // the `predicted` probability prediction for that label.
    // Here's an example implementation:
    double loss = 0.0;
    if (truth == 1) {
        loss = -log(predicted);
    } else if (truth == 0) {
        loss = -log(1.0 - predicted);
    }
    return loss;
}
*/


