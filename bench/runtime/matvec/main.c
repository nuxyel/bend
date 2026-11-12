// Native C twin of main.bend: a dense 2^D x 2^D matrix times a vector in
// f32, R times over, single-threaded, 1-to-1 with the Bend program: the
// same entries (stream 7 for A, 11 for x), each round's row i . x in the
// same four partial sums, hashed into y[i] = val(bits, i), then x and y
// swap, and the same fold of the last vector's bits. B only shapes
// Bend's forks, so it has no twin here.
// FP_CONTRACT is off, as in the C that Bend emits: with it on, an -O3
// that has fma fuses a * b + c and prints other bits.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma STDC FP_CONTRACT OFF

#ifndef D
#define D 12u
#endif
#ifndef R
#define R 256u
#endif

static uint32_t word_prng(uint32_t x) {
  uint32_t a = x ^ (x << 13u);
  uint32_t b = a ^ (a >> 17u);
  return b ^ (b << 5u);
}

// the low 24 bits of u as a number in [-1, 1)
static float unit(uint32_t u) {
  return ((float)(u & 16777215u) - 8388608.0f) / 8388608.0f;
}

static float val(uint32_t s, uint32_t k) {
  return unit(word_prng(word_prng((k + 1u) * 2654435761u + s)));
}

// row . x over n entries: product j goes to sum j % 4, and the four are
// added in pairs
static float dot(uint32_t n, const float *row, const float *x) {
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  for (uint32_t j = 0u; j < n; j += 4u) {
    s0 += row[j] * x[j];
    s1 += row[j + 1u] * x[j + 1u];
    s2 += row[j + 2u] * x[j + 2u];
    s3 += row[j + 3u] * x[j + 3u];
  }
  return (s0 + s1) + (s2 + s3);
}

// one round: every row of y from its dot product's bits
static void mul(uint32_t n, const float *a, const float *x, float *y) {
  for (uint32_t i = 0u; i < n; ++i) {
    float p = dot(n, a + (size_t)i * n, x);
    uint32_t b;
    memcpy(&b, &p, sizeof b);
    y[i] = val(b, i);
  }
}

int main(void) {
  uint32_t n = 1u << D;
  float *a = calloc((size_t)n * n, sizeof(float));
  float *x = calloc(n, sizeof(float));
  float *y = calloc(n, sizeof(float));
  uint32_t h = 0u;
  for (uint32_t k = 0u; k < n * n; ++k) {
    a[k] = val(7u, k);
  }
  for (uint32_t j = 0u; j < n; ++j) {
    x[j] = val(11u, j);
  }
  for (uint32_t r = 0u; r < R; ++r) {
    mul(n, a, x, y);
    float *t = x;
    x = y;
    y = t;
  }
  for (uint32_t j = 0u; j < n; ++j) {
    uint32_t b;
    memcpy(&b, &x[j], sizeof b);
    h = (h * 2654435761u) ^ b;
  }
  printf("%u\n", h);
  free(a);
  free(x);
  free(y);
  return 0;
}
