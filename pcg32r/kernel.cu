#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <stdio.h>

#include "random.h"
#ifdef __INTELLISENSE__
#include "intellisense_cuda_intrinsics.h"
#endif

__device__ uint64_t gpu_unxorshift( uint64_t xorshift ) {
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

    return state;
}

__global__ void reverseSeedKernel( uint32_t* d_next_numbers, uint64_t* d_valid_states, uint32_t* d_valid_count, uint64_t highbits, uint32_t rot, int vec_sz, int extra_bits_lost, uint64_t start_idx, uint64_t n ) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = blockDim.x * gridDim.x;

    // start_idx currently 0. if gpu iteration is too slow, then we parallelize with the cpu and use start_idx.
    for (uint64_t i = start_idx + idx; i < n; i += stride) {
        uint64_t fullxor = highbits ^ i;
        fullxor |= ((uint64_t)rot << 59);

        // original state. we need to undo the xorshift
        uint64_t possible_state = gpu_unxorshift( fullxor );

        /*
        # if __CUDA_ARCH__>=200
            printf("AYY %llu \n", possible_state);
        #endif
        */

        // if (possible_output == output) can remove, always true because of output reversal
        // instead, we check the number the next state generates and compare.
        uint64_t next_state = possible_state * PCG32_CONST + (PCG32_INC | 1);
        bool all = true;
        for (int i = 0; i < vec_sz; i++) {
            uint32_t next_rand = d_next_numbers[i];

            if (next_rand != 0) {
                // manual inlining of pcg32_random
                uint32_t xorshifted = uint32_t( ((next_state >> 18u) ^ next_state) >> 27u );
                uint32_t next_rot = uint32_t( next_state >> 59u );
                uint32_t computed_next_rand = (xorshifted >> next_rot) | (xorshifted << ((-int32_t( next_rot )) & 31));

                if (extra_bits_lost > 0) {
                    uint32_t imprecision = (1 << extra_bits_lost); // 2^extra bits
                    uint32_t diff = (uint32_t)llabs( (int64_t)computed_next_rand - (int64_t)next_rand );

                    if (diff > imprecision) {
                        all = false;
                        break;
                    }
                }
                else {
                    if (computed_next_rand != next_rand) {
                        all = false;
                        break;
                    }
                }
            }

            next_state = next_state * PCG32_CONST + (PCG32_INC | 1);
        }

        if (all && *d_valid_count < 1024) { // all numbers successfully generated through the sequence
            int pos = atomicAdd( d_valid_count, 2 );
            d_valid_states[pos] = possible_state;
            d_valid_states[pos + 1] = next_state;
        }
    }
}
