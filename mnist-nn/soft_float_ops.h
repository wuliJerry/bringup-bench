#ifndef SOFT_FLOAT_OPS_H
#define SOFT_FLOAT_OPS_H

#include <stdint.h> // Required for uint32_t, uint64_t

// Define DTYPE_FLOAT if your original DTYPE is float, otherwise assumes double
// This affects only the interface function, internal logic uses 32-bit float representation
typedef double DTYPE;
// typedef double DTYPE; // Assuming original DTYPE was double if not float

/**
 * @brief Performs a simulated 32-bit single-precision floating-point multiplication
 * using integer operations, allowing for fault injection in the core
 * integer multiplication step.
 *
 * @param a_f The first float operand.
 * @param b_f The second float operand.
 * @return The float result of a * b.
 *
 * @note This is a simplified implementation:
 * - Ignores NaNs, infinities, and denormalized numbers.
 * - Uses truncation for rounding.
 * - Assumes standard IEEE 754 single-precision format.
 * - The core integer multiplication happens in `faulty_integer_multiply32x32`.
 */
float soft_float_mul32(float a_f, float b_f);


void print_soft_float_stats(void);

/**
 * @brief Performs a 32x32-bit integer multiplication, resulting in 64 bits.
 * *** THIS IS THE TARGET FUNCTION FOR FAULT INJECTION ***
 *
 * @param a 32-bit unsigned integer multiplicand.
 * @param b 32-bit unsigned integer multiplier.
 * @return 64-bit unsigned integer result of a * b.
 *
 * @note Currently implements CORRECT multiplication. Modify this function
 * to inject faults according to your error pattern.
 */
uint64_t faulty_integer_multiply32x32(uint32_t a, uint32_t b);


#endif // SOFT_FLOAT_OPS_H