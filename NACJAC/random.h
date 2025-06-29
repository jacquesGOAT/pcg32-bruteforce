#pragma once
#include <inttypes.h>

#define PCG32_INC 105
#define PCG32_CONST 0x5851F42D4C957F2DULL

uint32_t pcg32_random( uint64_t* state );
int math_random( uint64_t* state, int lower, int upper );
void pcg32_seed( uint64_t* state, uint64_t seed );
uint64_t xorshift( uint64_t n );
uint64_t unxorshift( uint64_t xorshift );
