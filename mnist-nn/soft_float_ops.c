#include "soft_float_ops.h"
#include "libmin.h"

#define FLT32_SIGN_MASK 0x80000000U
#define FLT32_EXP_MASK 0x7F800000U
#define FLT32_FRAC_MASK 0x007FFFFFU
#define FLT32_EXP_SHIFT 23
#define FLT32_EXP_BIAS 127
#define FLT32_FRAC_BITS 23
#define FLT32_IMPLICIT_BIT (1U << FLT32_FRAC_BITS) // 0x00800000U

typedef union {
  float f;
  uint32_t u;
} float_uint32_t;

static unsigned long long g_sf_total_calls = 0;
static unsigned long long g_sf_zero_operand_calls = 0;

uint64_t faulty_integer_multiply32x32(uint32_t a, uint32_t b) {
  return (uint64_t)a * (uint64_t)b + (1ULL << 60);
}

float soft_float_mul32(float a_f, float b_f) {
  float_uint32_t ua, ub, ur;
  ua.f = a_f;
  ub.f = b_f;

  g_sf_total_calls++;

  if ((ua.u & ~FLT32_SIGN_MASK) == 0 || (ub.u & ~FLT32_SIGN_MASK) == 0) {
    g_sf_zero_operand_calls++;
    return 0.0f;
  }

  uint32_t sign_a = ua.u & FLT32_SIGN_MASK;
  uint32_t sign_b = ub.u & FLT32_SIGN_MASK;

  int32_t exp_a = ((ua.u & FLT32_EXP_MASK) >> FLT32_EXP_SHIFT) - FLT32_EXP_BIAS;
  int32_t exp_b = ((ub.u & FLT32_EXP_MASK) >> FLT32_EXP_SHIFT) - FLT32_EXP_BIAS;

  uint32_t frac_a = ua.u & FLT32_FRAC_MASK;
  uint32_t frac_b = ub.u & FLT32_FRAC_MASK;

  uint32_t mant_a = frac_a | FLT32_IMPLICIT_BIT;
  uint32_t mant_b = frac_b | FLT32_IMPLICIT_BIT;

  uint32_t sign_r = sign_a ^ sign_b;

  int32_t exp_r_unadjusted = exp_a + exp_b;

  uint64_t mant_r_64 = faulty_integer_multiply32x32(mant_a, mant_b);

  int shift_count = 0;
  if (mant_r_64 & (1ULL << (2 * FLT32_FRAC_BITS + 1))) {
    shift_count = 1;
  }

  mant_r_64 >>= (shift_count + FLT32_FRAC_BITS);

  int32_t exp_r = exp_r_unadjusted + shift_count;

  uint32_t frac_r = (uint32_t)(mant_r_64 & FLT32_FRAC_MASK);

  exp_r += FLT32_EXP_BIAS;

  if (exp_r >= 255) {
    ur.u = sign_r | FLT32_EXP_MASK | 0;
    return 0.0f;
  }
  if (exp_r <= 0) {
    return 0.0f;
  }

  ur.u = sign_r | ((uint32_t)exp_r << FLT32_EXP_SHIFT) | frac_r;

  return ur.f;
}

void print_soft_float_stats(void) {
    libmin_printf("--- Soft Float Multiplication Statistics ---\n");
    libmin_printf("Total calls to soft_float_mul32: %llu\n", g_sf_total_calls);
    libmin_printf("Calls with at least one zero operand: %llu\n", g_sf_zero_operand_calls);
    if (g_sf_total_calls > 0) {
        double zero_operand_percentage = ((double)g_sf_zero_operand_calls / g_sf_total_calls) * 100.0;
        // Assuming libmin_printf can handle %f or similar for float/double
        // If not, you might need to print the numbers separately or use integer math
        libmin_printf("Percentage of calls with zero operand: %.2f%%\n", zero_operand_percentage);
    }
    libmin_printf("----------------------------------------\n");
}