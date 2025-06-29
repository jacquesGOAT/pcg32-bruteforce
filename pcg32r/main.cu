#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <iomanip>
#include <thread>
#include <mutex>
#include <chrono>
#include <omp.h>

#include "random.h"

#define MAX_RESULTS 1ULL << 10

__global__ void reverseSeedKernel( uint32_t* d_next_numbers, uint64_t* d_valid_states, uint32_t* d_valid_count, uint64_t highbits, uint32_t rot, int vec_sz, int extra_bits_lost, uint64_t start_idx, uint64_t n );

struct cudaIO {
	int vec_sz;
	uint32_t** d_next_numbers;
	uint64_t** d_valid_states;
	uint32_t** d_valid_count;
};

std::mutex results_mutex;

void checkCudaError( cudaError_t err, const char* msg ) { // gpt
	if (err != cudaSuccess) {
		std::cerr << msg << ": " << cudaGetErrorString( err ) << '\n';
		exit( -1 );
	}
}

void lowbits_bf( uint64_t highbits, uint32_t rot, cudaIO** d_info, int gpu_count, std::vector<uint64_t>& valid_states, int extra_bits_lost ) {
	const uint64_t TOTAL_NUMBERS = 1ULL << 27;
	const int THREADS_PER_BLOCK = 1024; // CUDA threads per block
	const int BLOCKS = 512; // CUDA blocks

	const int CHUNK_SIZE = TOTAL_NUMBERS / gpu_count; // num of gpus

	#pragma omp parallel for num_threads(gpu_count) schedule(static)
	for (int i = 0; i < gpu_count; i++) {
		cudaSetDevice( i );

		cudaIO* c = d_info[i];

		// we iterate on the gpu.
		uint64_t start = i * CHUNK_SIZE;
		reverseSeedKernel<<<BLOCKS, THREADS_PER_BLOCK>>>(*c->d_next_numbers, *c->d_valid_states, *c->d_valid_count, highbits, rot, c->vec_sz, extra_bits_lost, start, start + CHUNK_SIZE);
	} 
}

void test_rot(uint32_t rand, cudaIO** d_info, int gpu_count, std::vector<uint64_t>& valid_states, int extra_bits_lost ) {
	const int NUM_THREADS = omp_get_max_threads(); // Automatically detect threads

	#pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
	for (int i = 0; i < 32; i++) {
		uint32_t rot = (uint32_t)i;
		uint32_t xorshifted_candidate = (rand << rot) | (rand >> ((-int32_t( rot )) & 31));
		uint64_t highbits = (uint64_t)xorshifted_candidate << 27; // bottom 27 are lost (by default), we need to bruteforce them
		
		lowbits_bf( highbits, rot, d_info, gpu_count, valid_states, extra_bits_lost );
	}

	for (int i = 0; i < gpu_count; i++) {
		cudaIO* c = d_info[i];

		cudaStream_t stream;
		cudaStreamCreate( &stream );

		uint32_t h_valid_count;
		checkCudaError( cudaMemcpyAsync( &h_valid_count, *c->d_valid_count, sizeof( uint32_t ), cudaMemcpyDeviceToHost, stream ), "Failed to copy valid result count" );
		if (h_valid_count > 0) {
			uint64_t* h_valid_states = new uint64_t[h_valid_count];
			checkCudaError( cudaMemcpyAsync( h_valid_states, *c->d_valid_states, h_valid_count * sizeof( uint64_t ), cudaMemcpyDeviceToHost, stream ), "Failed to copy valid result array" );

			for (int i = 0; i < h_valid_count; i += 2) { // somewhat reworked. basically idx i is the first state, and idx i + 1 is the state after all the numbers have passed
				uint64_t state = h_valid_states[i];
				uint64_t next_state = h_valid_states[i + 1];
				uint32_t rot = uint32_t( state >> 59u );

				if (std::find( valid_states.begin(), valid_states.end(), state ) == valid_states.end()) {
					{
						std::lock_guard<std::mutex> lock( results_mutex );
						std::cout << "Found valid state: " << state << " at rotation " << rot << " for rand " << rand << ". Current state: " << next_state << '\n';
						valid_states.push_back( state );
					}
				}
			}

			delete[] h_valid_states;
		}
	}

	{
		std::lock_guard<std::mutex> lock( results_mutex );
		std::cout << "Tested all rotations for rand " << rand << '\n';
	}
}

void inverse_pcg32_random( uint32_t rand, cudaIO** d_info, int gpu_count, std::vector<uint64_t>& valid_states, int extra_bits_lost ) { // should return the seed.
	if (extra_bits_lost > 0) {
		uint64_t imprecision = (1ULL << extra_bits_lost) - 1;
		int64_t lower_bound = rand - imprecision;
		int64_t upper_bound = rand + imprecision;

		if (lower_bound < 0) {
			lower_bound = 0;
		}

		if (upper_bound > UINT32_MAX) {
			upper_bound = UINT32_MAX;
		}

		std::vector<std::thread> threads;
		const int MAX_THREADS = 8;

		for (uint64_t rand2 = lower_bound; rand2 < upper_bound; rand2++) {

			if (threads.size() >= MAX_THREADS) {
				threads.front().join();
				threads.erase( threads.begin() );
			}

			threads.emplace_back( [=, &valid_states]() {
				test_rot( rand2, d_info, gpu_count, valid_states, extra_bits_lost );
			} );
		}

		for (auto& thread : threads) {
			thread.join();
		}
	}
	else {
		test_rot( rand, d_info, gpu_count, valid_states, extra_bits_lost );
	}
}

void bruteforce( std::vector<uint32_t> next_numbers, int extra_bits_lost ) {
	uint32_t first = next_numbers.back(); // first number is excluded from further checks because we use it to construct the states
	next_numbers.pop_back();

	std::reverse( next_numbers.begin(), next_numbers.end() ); // reverse it because push back. maybe switch to an array idk

	std::vector<uint64_t> valid_states;

	// initialize re-used variables
	int numGPUs;
	cudaGetDeviceCount( &numGPUs );

	cudaIO** d_info = new cudaIO*[numGPUs];

	for (int i = 0; i < numGPUs; i++) {
		cudaSetDevice( i );

		cudaIO* c = new cudaIO;

		// no stl
		int vec_sz = next_numbers.size();
		uint32_t** d_next_numbers = new uint32_t*;
		uint64_t** d_valid_states = new uint64_t*;
		uint32_t** d_valid_count = new uint32_t*;

		uint32_t* d_next_numbers_s;
		uint64_t* d_valid_states_s;
		uint32_t* d_valid_count_s;

		checkCudaError( cudaMalloc( (void**)&d_next_numbers_s, vec_sz * sizeof( uint32_t ) ), "Failed to allocate device memory for d_next_numbers" );
		checkCudaError( cudaMalloc( (void**)&d_valid_states_s, (MAX_RESULTS) * sizeof( uint32_t ) ), "Failed to allocate device memory for d_valid_states" );
		checkCudaError( cudaMalloc( (void**)&d_valid_count_s, sizeof( uint32_t ) ), "Failed to allocate device memory for d_valid_count" );
	
		checkCudaError( cudaMemcpy( d_next_numbers_s, next_numbers.data(), vec_sz * sizeof( uint32_t ), cudaMemcpyHostToDevice ), "Failed to copy next_numbers array to device" );
		checkCudaError( cudaMemset( d_valid_count_s, 0, sizeof( uint32_t ) ), "Failed to empty d_valid_count results" );

		*d_next_numbers = d_next_numbers_s;
		*d_valid_states = d_valid_states_s;
		*d_valid_count = d_valid_count_s;

		c->vec_sz = vec_sz;
		c->d_next_numbers = d_next_numbers;
		c->d_valid_states = d_valid_states;
		c->d_valid_count = d_valid_count;

		d_info[i] = c;
	}

	std::cout << "Starting bruteforce for state sequence: " << first;
	for (uint32_t n : next_numbers) {
		std::cout << ", " << n;
	}

	std::cout << '\n' << "Extra bits lost: " << extra_bits_lost << '\n';

	auto start = std::chrono::high_resolution_clock::now();

	inverse_pcg32_random( first, d_info, numGPUs, valid_states, extra_bits_lost );

	// Synchronize everything
	for (int i = 0; i < numGPUs; i++) {
		cudaSetDevice( i );
		cudaDeviceSynchronize();
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	std::cout << "Computed all possible states in " << duration << "ms.\n";

	// free for each gpu
	for (int i = 0; i < numGPUs; i++) {
		cudaSetDevice( i );

		cudaIO* c = d_info[i];

		cudaFree( c->d_next_numbers );
		cudaFree( c->d_valid_count );
		cudaFree( c->d_valid_states );

		delete c->d_next_numbers;
		delete c->d_valid_count;
		delete c->d_valid_states;

		delete c;
	}

	delete[] d_info;

	if (valid_states.empty()) {
		std::cout << "Found no valid states.\n";
	}
	else {
		for (uint64_t state : valid_states) {
			std::cout << "Found state: " << state << '\n';
		}
	}
}

void generate_sequence_math_random( uint64_t state ) {
	std::cout << "Sequence for state " << state << ":\n";

	pcg32_random( &state ); // skip one

	for (int i = 0; i < 10; i++) {
		uint32_t rl = pcg32_random( &state );
		uint32_t rh = pcg32_random( &state );

		double rd = ldexp( double( rl | (uint64_t( rh ) << 32) ), -64 );

		std::cout << std::setprecision( std::numeric_limits<double>::digits10 + 2 ) << rd << std::endl;
	}
}

void generate_sequence_state( uint64_t state ) {
	std::cout << "Sequence for state " << state << ":\n";

	for (int i = 0; i < 10; i++) {
		pcg32_random( &state );
		std::cout << state << '\n';
	}
}

void print_help( const std::string& program_name ) {
	std::cout << "Usage: " << program_name << " [options]\n";
	std::cout << "Options:\n";
	std::cout << "  -?  | --help                     Show this help message\n";
	std::cout << "  -l  | --bitloss <n>              Amount of extra bits lost. By default it is 0\n";
	//std::cout << "  -s  | --skip <n>                Amount of states to skip after each number. By default it is set to 1 (math.random)\n";
	std::cout << "  -b  | --bruteforce <list>        Bruteforces from a list of math.randoms\n";
	std::cout << "  -t  | --test <state>             Generates a sequence of math.randoms from a state\n";
	std::cout << "  -gs | --getstate <seed>          Generate the state for a given seed\n";
	std::cout << "  -ns | --nextstates <states>      Generate a sequence of states starting from one\n";
	std::cout << "  -u  | --unbound <value> <l> <u>  Returns the random value unbounded by range l to u\n";
}

int main( int argc, char* argv[] ) {
	if (argc < 2) { // no parameters passed
		std::cout << "No arguments provided. Use -? or --help for usage information.\n";
		return 0;
	}

	int extra_bits_lost = 0;
	//int skip = 1; // shjould b removed
			/*else if (arg == "-s" || arg == "--skip") {
			if (i + 1 < argc) {
				skip = atoi( argv[++i] );
			}
			else {
				std::cerr << "Error: " << arg << " requires a number parameter.\n";
				return 1;
			}
		}*/

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];

		if (arg == "-?" || arg == "--help") {
			print_help( argv[0] );
			return 0;
		}
		else if (arg == "-l" || arg == "--bitloss") {
			if (i + 1 < argc) {
				extra_bits_lost = atoi( argv[++i] );
			}
			else {
				std::cerr << "Error: " << arg << " requires a number parameter.\n";
				return 1;
			}
		}
		else if (arg == "-b" || arg == "--bruteforce") {
			std::vector<uint32_t> next_numbers;

			int c = 0;
			while (true) {
				if (i + 1 < argc) {
					char* rand = argv[++i];
					if (tolower( *rand ) == 'x') {
						next_numbers.push_back( 0 );
					}
					else { // math.random() double
						double d = strtod( rand, NULL ); // Get the next argument as name
						uint64_t rand_64 = (uint64_t)(ldexp( d, 64 ));
						uint32_t rand = (uint32_t)(rand_64 >> 32);

						if (next_numbers.size() > 0) { // ignore first zero, it is useless.
							next_numbers.push_back( 0 );
						}
						next_numbers.push_back( rand );
					}

					c++;
				}
				else {
					break;
				}
			}

			if (c < 2) {
				std::cerr << "Error: at least 2 numbers must be passed to begin bruteforcing.\n";
				return 1;
			}
			else {
				std::reverse( next_numbers.begin(), next_numbers.end() );

				bruteforce( next_numbers, extra_bits_lost );
			}
		}
		else if (arg == "-t" || arg == "--test") {
			if (i + 1 < argc) {
				uint64_t state = strtoull( argv[++i], NULL, 10 ); // Convert the next argument to an integer
				generate_sequence_math_random( state );
			}
			else {
				std::cerr << "Error: " << arg << " requires a state parameter.\n";
				return 1;
			}
		}
		else if (arg == "-gs" || arg == "--getstate") {
			if (i + 1 < argc) {
				uint32_t seed = strtoull( argv[++i], NULL, 10 ); // Convert the next argument to an integer
				uint64_t state;
				pcg32_seed( &state, seed );

				std::cout << "State value: " << state << '\n';

			}
			else {
				std::cerr << "Error: " << arg << " requires a seed parameter.\n";
				return 1;
			}
		}
		else if (arg == "-ns" || arg == "--nextstates") {
			if (i + 1 < argc) {
				uint64_t state = strtoull( argv[++i], NULL, 10 );
				generate_sequence_state( state );

			}
			else {
				std::cerr << "Error: " << arg << " requires a seed parameter.\n";
				return 1;
			}
		}
		else if (arg == "-u" || arg == "--unbound") {
			if (i + 3 < argc) {
				char* rand = argv[++i];
				char* lower = argv[++i];
				char* upper = argv[++i];

				double d = strtod( rand, NULL );
				double l = strtod( lower, NULL );
				double u = strtod( upper, NULL );

				/*
				formula for ranged :NextNumber()
				0.9035909785572982 = (1.0 - v11) * 0.05 + v11*0.95
				_ = 0.05 - 0.05v11 + 0.95v11
				_ = 0.05 + 0.9v11
				*/

				double r = (d - l) / (u - l);
				std::cout.precision( 17 );
				std::cout << "Unbound NextNumber value: " << r << '\n';
			}
			else {
				std::cerr << "Error: " << arg << " requires 3 parameters.\n";
				return 1;
			}
		}
		else {
			std::cerr << "Unknown option: " << arg << "\n";
			std::cerr << "Use -? or --help to see available options.\n";
			return 1;
		}
	}

	return 0;
}