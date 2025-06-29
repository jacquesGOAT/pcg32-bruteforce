#include "pch.h"
#include "random.h"

uint32_t pcg32_random( uint64_t* state ) { // gpu implementation in kernel.cu
    uint64_t oldstate = *state;
    *state = oldstate * PCG32_CONST + (PCG32_INC | 1); // reminder to use roblox's constant (0x5851F42D4C957F2D)
    uint32_t xorshifted = uint32_t( ((oldstate >> 18u) ^ oldstate) >> 27u );
    uint32_t rot = uint32_t( oldstate >> 59u );
    return (xorshifted >> rot) | (xorshifted << ((-int32_t( rot )) & 31));
}

int math_random( uint64_t* state, int lower, int upper ) {
    uint32_t ul = uint32_t( upper ) - uint32_t( lower );
    uint64_t x = uint64_t( ul + 1 ) * pcg32_random( state );

    int r = int( lower + (x >> 32) );
    return r;
}

void pcg32_seed( uint64_t* state, uint64_t seed ) { // ripped from the blox
    *state = PCG32_CONST * seed + 0x399D2694695129DEULL;
}

// xorshift/unxorshift
uint64_t xorshift( uint64_t n ) {
    return (n >> 18u) ^ n;
}

uint64_t unxorshift( uint64_t xorshift ) { // gpu implementation in kernel.cu
    // we know that the first 18 bits of the xorshift are from the state because 
    // the state is originally shifted 18 bits right, setting all first 18 bits to 0.
    // the xor operation on all those bits therefore give us the first 18 bits of the real state

    uint64_t state = xorshift & (~((1ULL << 46) - 1));
    //std::cout << state << '\n';
    for (char i = 63 - 18; i >= 0; i--) {
        bool upperBit = (state & (1ULL << (i + 18))) != 0;
        bool currentBit = (xorshift & (1ULL << i)) != 0;
        bool newBit = upperBit ^ currentBit;

        state |= ((uint64_t)newBit << i);
    }

    pcg32_random( &state );

    return state;
}