#pragma once
#include <stdint.h>

#ifdef cplusplus
extern "C" {
#endif

float  dot_product_f32(const float* p, const float* q, uint64_t n);
double dot_product_f64(const double* p, const double* q, uint64_t n);

#ifdef cplusplus
} // extern "C"
#endif
